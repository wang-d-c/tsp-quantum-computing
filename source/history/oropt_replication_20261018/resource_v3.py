"""Bounded process-tree resource sampling and explicitly scoped containment.

No system-wide process enumeration, sample list, package installation, or
solver decision logic is implemented here.  Caller schedules samples (20ms by
default) and owns an independent deadline watchdog.  Sampling can miss short-
lived children and sub-interval RSS peaks; summed RSS double-counts shared pages
and is not proportional/unique physical memory.  Unavailable metrics are None
with errors, never invented zeros.  CPU observations are user+system only, not
the parent's children_* fields (which would double count and differ by OS).

On Windows, attach WindowsJob BEFORE a READY/GO handshake permits the worker
to spawn children.  Kill-on-close then contains its ordinary descendants even
if the root exits.  Attaching to an already-running arbitrary process cannot
retroactively contain children it created earlier.  POSIX fallback cleanup is
identity-checked best effort, not a kernel containment guarantee.

Primary API references used to implement the optional Windows boundary:
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_accounting_information
https://psutil.readthedocs.io/stable/
"""

from __future__ import annotations

from collections import Counter
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
import time

try:
    import psutil
except ImportError as exc:  # No silent zero-valued fallback.
    raise ImportError("resource_v3 requires an installed psutil; no resources were measured") from exc


class ResourceUnavailable(RuntimeError):
    """A requested containment/measurement facility could not be established."""


@dataclass(slots=True)
class OnlineMoments:
    """Constant-space moments; None means no valid observations."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("statistics require finite observations")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def as_dict(self) -> dict:
        return dict(count=self.count, mean=self.mean if self.count else None,
                    variance=self.m2 / (self.count - 1) if self.count > 1 else None,
                    minimum=self.minimum, maximum=self.maximum)


def _pid(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("pid must be an explicitly named positive integer")
    return value


def _identity(pid: int, expected_create_time: float | None = None):
    process = psutil.Process(_pid(pid))
    created = float(process.create_time())
    if expected_create_time is not None and not math.isclose(created, expected_create_time, rel_tol=0., abs_tol=1e-6):
        raise ResourceUnavailable("PID creation time differs; refusing a reused process identity")
    return process, created


def _safe_mutation_target(pid: int) -> None:
    _pid(pid)
    if pid == os.getpid():
        raise ValueError("refusing to terminate or contain the calling process")
    try:
        if any(parent.pid == pid for parent in psutil.Process().parents()):
            raise ValueError("refusing to terminate or contain an ancestor of the caller")
    except psutil.Error as exc:
        raise ResourceUnavailable("cannot verify the caller's ancestry safely") from exc


class ProcessMonitor:
    """Sample one specified process and its discovered descendants, O(cap) RAM.

    sample_path is opened exclusively and every record is immediately flushed.
    summary RSS mean is arithmetic across complete samples, not a continuous
    time average.  CPU totals preserve last observations of terminated children
    but can miss their final sub-interval CPU use; WindowsJob.accounting gives
    separate kernel job totals including terminated members when available.
    """

    def __init__(self, pid: int, *, interval_seconds: float = .02,
                 sample_path: str | Path | None = None, max_tracked_processes: int = 128):
        if isinstance(interval_seconds, bool) or not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")
        if isinstance(max_tracked_processes, bool) or not isinstance(max_tracked_processes, int) or max_tracked_processes < 1:
            raise ValueError("max_tracked_processes must be a positive integer")
        try:
            self.root, self.root_create_time = _identity(pid)
        except (psutil.Error, OSError) as exc:
            raise ResourceUnavailable(f"cannot monitor PID {pid}: {type(exc).__name__}") from exc
        self.pid = pid
        self.interval_seconds = float(interval_seconds)
        self.max_tracked_processes = max_tracked_processes
        self._tracked = {(pid, self.root_create_time): None}
        self._started = time.perf_counter()
        self._previous_stamp = None
        self._previous_cpu_total = None
        self._rss = OnlineMoments()
        self._cpu_rate = OnlineMoments()
        self._lag = OnlineMoments()
        self._gap = OnlineMoments()
        self._duration = OnlineMoments()
        self._error_counts = Counter()
        self._sample_count = 0
        self._live_sample_count = 0
        self._incomplete_samples = 0
        self._tracking_overflow_samples = 0
        self._cpu_complete_samples = 0
        self._closed = False
        self._write_seconds = 0.0
        self._monitor_process_cpu_seconds = 0.0
        self.sample_path = None if sample_path is None else str(Path(sample_path).resolve())
        self._stream = None if sample_path is None else Path(sample_path).open("x", encoding="utf-8", newline="\n")

    def sample(self) -> dict:
        if self._closed:
            raise RuntimeError("monitor is closed")
        stamp = time.perf_counter()
        cpu_started = time.process_time()
        errors = []
        error_total = 0

        def error(operation, exc):
            nonlocal error_total
            error_total += 1
            name = type(exc).__name__
            self._error_counts[f"{operation}:{name}"] += 1
            if len(errors) < 16:
                errors.append(dict(operation=operation, error=name))

        root_alive = False
        discovery_complete = True
        try:
            root, _ = _identity(self.pid, self.root_create_time)
            root_alive = root.is_running() and root.status() != psutil.STATUS_ZOMBIE
            children = root.children(recursive=True) if root_alive else []
            for child in children:
                try:
                    identity = (child.pid, float(child.create_time()))
                    if identity not in self._tracked:
                        if len(self._tracked) >= self.max_tracked_processes:
                            discovery_complete = False
                            self._tracking_overflow_samples += 1
                            error("tracking_limit", ResourceUnavailable("process cap reached"))
                            break
                        self._tracked[identity] = None
                except psutil.NoSuchProcess:
                    continue
                except (psutil.Error, OSError) as exc:
                    discovery_complete = False
                    error("child_identity", exc)
        except psutil.NoSuchProcess:
            root_alive = False
        except (psutil.Error, OSError, ResourceUnavailable) as exc:
            discovery_complete = False
            root_alive = None
            error("root_discovery", exc)
        rss_complete = discovery_complete
        cpu_complete = discovery_complete
        partial_rss = 0
        live = []
        for identity in self._tracked:
            pid, created = identity
            try:
                process, _ = _identity(pid, created)
                if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    continue
            except psutil.NoSuchProcess:
                continue
            except (psutil.Error, OSError, ResourceUnavailable) as exc:
                rss_complete = cpu_complete = False
                error("tracked_identity", exc)
                continue
            memory, process_cpu = None, None
            try:
                memory = int(process.memory_info().rss)
                partial_rss += memory
            except psutil.NoSuchProcess as exc:
                rss_complete = False
                error("rss_exit_race", exc)
            except (psutil.Error, OSError) as exc:
                rss_complete = False
                error("rss", exc)
            try:
                times = process.cpu_times()
                process_cpu = float(times.user + times.system)
                previous = self._tracked[identity]
                self._tracked[identity] = process_cpu if previous is None else max(previous, process_cpu)
            except psutil.NoSuchProcess as exc:
                cpu_complete = False
                error("cpu_exit_race", exc)
            except (psutil.Error, OSError) as exc:
                cpu_complete = False
                error("cpu", exc)
            live.append(dict(pid=pid, create_time=created, rss_bytes=memory,
                             cpu_user_system_seconds=process_cpu))
        observed_cpu = [value for value in self._tracked.values() if value is not None]
        total_cpu = sum(observed_cpu)
        if len(observed_cpu) != len(self._tracked):
            cpu_complete = False
        gap = None if self._previous_stamp is None else stamp - self._previous_stamp
        lag = 0.0 if gap is None else max(0.0, gap - self.interval_seconds)
        rate = None
        if gap is not None and gap > 0 and self._previous_cpu_total is not None and cpu_complete:
            rate = 100.0 * max(0.0, total_cpu - self._previous_cpu_total) / gap
            self._cpu_rate.add(rate)
        if rss_complete:
            self._rss.add(float(partial_rss))
        if gap is not None:
            self._gap.add(gap)
        self._lag.add(lag)
        self._sample_count += 1
        self._live_sample_count += bool(live)
        self._incomplete_samples += not (rss_complete and cpu_complete)
        self._cpu_complete_samples += cpu_complete
        self._previous_stamp = stamp
        self._previous_cpu_total = total_cpu if cpu_complete else None
        duration = time.perf_counter() - stamp
        self._duration.add(duration)
        row = dict(
            sample=self._sample_count, monotonic_seconds=stamp, elapsed_seconds=stamp - self._started,
            root_pid=self.pid, root_alive=root_alive, alive_process_count=len(live),
            tracked_process_count=len(self._tracked), process_tracking_cap=self.max_tracked_processes,
            discovery_complete=discovery_complete, rss_complete=rss_complete, cpu_complete=cpu_complete,
            rss_total_bytes=partial_rss if rss_complete else None, rss_known_partial_bytes=partial_rss,
            cpu_seconds_observed_tree=total_cpu if cpu_complete else None,
            cpu_seconds_known_partial=total_cpu if observed_cpu else None, cpu_percent_one_core=rate,
            sample_gap_seconds=gap, sample_lag_seconds=lag, sample_duration_seconds=duration,
            processes=live, errors=errors, errors_truncated=max(0, error_total - len(errors)),
        )
        if self._stream is not None:
            write_started = time.perf_counter()
            self._stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            self._stream.flush()
            self._write_seconds += time.perf_counter() - write_started
        self._monitor_process_cpu_seconds += time.process_time() - cpu_started
        return row

    def summary(self) -> dict:
        observed_cpu = [value for value in self._tracked.values() if value is not None]
        cpu_seconds = sum(observed_cpu)
        all_observed = len(observed_cpu) == len(self._tracked)
        return dict(
            pid=self.pid, root_create_time=self.root_create_time, psutil_version=psutil.__version__,
            sample_count=self._sample_count, live_sample_count=self._live_sample_count,
            interval_seconds=self.interval_seconds, sample_path=self.sample_path,
            measurement_status=("no_live_samples" if not self._live_sample_count else
                                "partial" if self._incomplete_samples else "sampled"),
            incomplete_samples=self._incomplete_samples, tracking_overflow_samples=self._tracking_overflow_samples,
            tracked_process_count=len(self._tracked), max_tracked_processes=self.max_tracked_processes,
            rss_peak_bytes=self._rss.maximum, rss_mean_bytes=self._rss.mean if self._rss.count else None,
            rss_complete_samples=self._rss.count,
            cpu_seconds_observed_tree=cpu_seconds if self._sample_count and all_observed else None,
            cpu_seconds_known_partial=cpu_seconds if observed_cpu else None,
            cpu_observed_process_count=len(observed_cpu),
            cpu_unobserved_process_count=len(self._tracked) - len(observed_cpu),
            cpu_complete_samples=self._cpu_complete_samples,
            cpu_percent_one_core_mean=self._cpu_rate.mean if self._cpu_rate.count else None,
            cpu_percent_one_core_peak=self._cpu_rate.maximum,
            sample_lag_max_seconds=self._lag.maximum, sample_lag_mean_seconds=self._lag.mean if self._lag.count else None,
            sample_gap=self._gap.as_dict(), sample_duration=self._duration.as_dict(),
            error_counts=dict(self._error_counts), jsonl_write_seconds=self._write_seconds,
            monitor_call_process_cpu_seconds=self._monitor_process_cpu_seconds,
            closed=self._closed,
            limitations=["RSS peak is sampled, not continuous; tree RSS may double-count shared pages",
                         "RSS mean is arithmetic over complete samples",
                         "Unseen short-lived children and CPU after the last sample may be missed",
                         "CPU percentage uses one core=100%; no system-wide utilization was queried",
                         "Sample gaps/lag show monitor scheduling delay, not its unobserved cause"],
        )

    def close(self) -> None:
        if not self._closed:
            if self._stream is not None:
                self._stream.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def kill_process_tree(pid: int, *, expected_create_time: float | None = None,
                      grace_seconds: float = .1, wait_seconds: float = 1.0) -> dict:
    """Terminate only an explicit identity and discovered descendants.

    WindowsJob is the preferred containment boundary.  This fallback snapshots
    descendants, verifies each identity before signaling, and reports failures;
    it cannot guarantee finding descendants orphaned before the snapshot.
    A positive grace uses terminate then kill; Windows terminate is not a
    graceful application shutdown.  Never call on the caller or its ancestors.
    """

    _safe_mutation_target(pid)
    if not math.isfinite(grace_seconds) or grace_seconds < 0 or not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError("cleanup waits must be finite and nonnegative")
    started = time.perf_counter()
    try:
        root, created = _identity(pid, expected_create_time)
    except psutil.NoSuchProcess:
        return dict(pid=pid, status="already_exited", targeted=[], surviving=[], errors=[],
                    elapsed_seconds=time.perf_counter() - started,
                    containment="best_effort; root already exited, unknown orphans cannot be discovered")
    children = root.children(recursive=True)
    identities = []
    for process in [*reversed(children), root]:
        try:
            identities.append((process.pid, float(process.create_time())))
        except psutil.NoSuchProcess:
            continue
    targeted, errors, objects = [], [], []
    for child_pid, child_created in identities:
        try:
            process, _ = _identity(child_pid, child_created)
            targeted.append(child_pid)
            objects.append(process)
            if grace_seconds:
                process.terminate()
            else:
                process.kill()
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, OSError, ResourceUnavailable) as exc:
            errors.append(dict(pid=child_pid, error=type(exc).__name__))
    _, alive = psutil.wait_procs(objects, timeout=grace_seconds if grace_seconds else wait_seconds)
    for process in alive:
        try:
            verified, _ = _identity(process.pid, process.create_time())
            verified.kill()
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, OSError, ResourceUnavailable) as exc:
            errors.append(dict(pid=process.pid, error=type(exc).__name__))
    _, alive = psutil.wait_procs(alive, timeout=wait_seconds)
    return dict(pid=pid, root_create_time=created, status="terminated" if not alive and not errors else "incomplete",
                targeted=targeted, surviving=[p.pid for p in alive], errors=errors,
                elapsed_seconds=time.perf_counter() - started,
                containment="identity-checked descendant snapshot; no pre-existing orphan guarantee")


class WindowsJob:
    """Unnamed kill-on-close Windows job, with thread-safe terminate/close.

    Attach before the worker's GO gate.  A construction failure raises rather
    than silently downgrading containment.  close() also kills all members;
    inspect accounting() first if final job CPU totals are required.
    """

    def __init__(self, pid: int, *, expected_create_time: float | None = None):
        if os.name != "nt":
            raise ResourceUnavailable("WindowsJob requires Windows; use an explicitly documented POSIX boundary")
        _safe_mutation_target(pid)
        _, created = _identity(pid, expected_create_time)
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                        ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                        ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                        ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "GetProcessTimes": ([wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes = arguments
            function.restype = result
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ResourceUnavailable(f"CreateJobObjectW failed: {ctypes.get_last_error()}")
        process_handle = None
        try:
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE, no breakaway.
            if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ResourceUnavailable(f"SetInformationJobObject failed: {ctypes.get_last_error()}")
            process_handle = kernel.OpenProcess(0x0100 | 0x0001 | 0x1000, False, pid)
            if not process_handle:
                raise ResourceUnavailable(f"OpenProcess failed: {ctypes.get_last_error()}")
            creation, exited, system, user = (ctypes.c_uint64() for _ in range(4))
            if not kernel.GetProcessTimes(process_handle, ctypes.byref(creation), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
                raise ResourceUnavailable(f"GetProcessTimes failed: {ctypes.get_last_error()}")
            handle_created = creation.value / 10_000_000 - 11_644_473_600
            if not math.isclose(handle_created, created, rel_tol=0., abs_tol=1e-5):
                raise ResourceUnavailable("OpenProcess handle creation time differs; refusing PID reuse")
            if not kernel.AssignProcessToJobObject(handle, process_handle):
                raise ResourceUnavailable(f"AssignProcessToJobObject failed: {ctypes.get_last_error()}")
        except Exception:
            kernel.CloseHandle(handle)
            raise
        finally:
            if process_handle:
                kernel.CloseHandle(process_handle)
        self.pid, self.root_create_time = pid, created
        self._kernel, self._handle, self._accounting_type = kernel, handle, Accounting
        self._lock = threading.Lock()

    def accounting(self) -> dict:
        with self._lock:
            if self._handle is None:
                raise ResourceUnavailable("job handle is closed")
            information = self._accounting_type()
            if not self._kernel.QueryInformationJobObject(self._handle, 1, ctypes.byref(information), ctypes.sizeof(information), None):
                raise ResourceUnavailable(f"QueryInformationJobObject failed: {ctypes.get_last_error()}")
            return dict(cpu_user_seconds=information.TotalUserTime / 10_000_000,
                        cpu_system_seconds=information.TotalKernelTime / 10_000_000,
                        cpu_seconds=(information.TotalUserTime + information.TotalKernelTime) / 10_000_000,
                        total_processes=int(information.TotalProcesses), active_processes=int(information.ActiveProcesses),
                        page_fault_count=int(information.TotalPageFaultCount),
                        scope="Windows kernel job accounting, including terminated members; not sampled CPU")

    def terminate(self, exit_code: int = 1) -> dict:
        requested = time.perf_counter()
        with self._lock:
            if self._handle is None:
                return dict(status="already_closed", requested_monotonic=requested, returned_monotonic=time.perf_counter())
            if not self._kernel.TerminateJobObject(self._handle, exit_code):
                raise ResourceUnavailable(f"TerminateJobObject failed: {ctypes.get_last_error()}")
        return dict(status="termination_requested", requested_monotonic=requested,
                    returned_monotonic=time.perf_counter(),
                    scope="kill request return does not prove all members have exited")

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                if not self._kernel.CloseHandle(self._handle):
                    raise ResourceUnavailable(f"CloseHandle failed: {ctypes.get_last_error()}")
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
