"""One cold, paid NN -> optional grouped search -> LKH experiment worker."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def scalar_length(coords, route):
    if len(route) != len(coords) or any(type(v) is not int for v in route):
        raise ValueError('noninteger or incomplete route')
    if sorted(route) != list(range(len(coords))):
        raise ValueError('route is not a permutation')
    total = 0
    for a, b in zip(route, route[1:] + route[:1]):
        dx = coords[a][0] - coords[b][0]
        dy = coords[a][1] - coords[b][1]
        total += math.floor(math.sqrt(dx * dx + dy * dy) + .5)
    return total


class Publisher:
    def __init__(self, directory, coords, deadline, origin):
        self.directory, self.coords, self.deadline = directory, coords, deadline
        self.origin = origin
        self.best = None
        self.route = None
        self.sequence = 0
        self.sharing_retries = 0

    def __call__(self, route, source):
        if time.perf_counter() >= self.deadline:
            return False
        values = [int(v) for v in route]
        cost = scalar_length(self.coords, values)
        if time.perf_counter() >= self.deadline:
            return False
        if self.best is not None and cost >= self.best:
            return False
        started = time.perf_counter()
        payload = dict(route=values, length=cost, source=source,
                       sequence=self.sequence + 1,
                       worker_write_started_perf_counter=started)
        temporary = self.directory / 'checkpoint.partial'
        with temporary.open('w', encoding='ascii') as stream:
            json.dump(payload, stream, separators=(',', ':'), allow_nan=False)
        for attempt in range(6):
            try:
                os.replace(temporary, self.directory / 'checkpoint.json')
                break
            except PermissionError as exc:
                if os.name != 'nt' or getattr(exc, 'winerror', None) not in (5, 32, 33):
                    raise
                self.sharing_retries += 1
                if attempt == 5 or time.perf_counter() >= self.deadline:
                    return False
                time.sleep(.001)
        self.best, self.route = cost, values
        self.sequence += 1
        return True


def tour_text(route):
    return ('TYPE : TOUR\nDIMENSION : ' + str(len(route)) + '\nTOUR_SECTION\n'
            + '\n'.join(str(int(v) + 1) for v in route) + '\n-1\nEOF\n')


def parse_tour(text, n):
    if 'TOUR_SECTION' not in text:
        raise ValueError('missing TOUR_SECTION')
    values = []
    for token in text.split('TOUR_SECTION', 1)[1].split():
        if token in ('-1', 'EOF'):
            break
        values.append(int(token) - 1)
    if len(values) != n or sorted(values) != list(range(n)):
        raise ValueError('incomplete tour')
    return values


def nearest_neighbor(coords, seed):
    import numpy as np
    points = np.asarray(coords, dtype=np.int64)
    n = len(points)
    used = np.zeros(n, dtype=bool)
    route = np.empty(n, dtype=np.int64)
    city = random.Random(seed).randrange(n)
    for position in range(n):
        route[position] = city
        used[city] = True
        if position + 1 < n:
            offsets = points - points[city]
            costs = np.floor(np.hypot(offsets[:, 0], offsets[:, 1]) + .5)
            costs[used] = np.inf
            city = int(np.argmin(costs))  # EUC_2D ties: smallest city ID.
    return route


def distance_matrix(coords):
    import numpy as np
    points = np.asarray(coords, dtype=np.int64)
    n = len(points)
    result = np.empty((n, n), dtype=np.int32)
    for first in range(0, n, 128):
        offsets = points[first:first + 128, None, :] - points[None, :, :]
        result[first:first + 128] = np.floor(
            np.hypot(offsets[:, :, 0], offsets[:, :, 1]) + .5).astype(np.int32)
    return result


def lkh_search(task, publisher, events):
    directory = Path(task['directory']) / 'lkh'
    directory.mkdir(exist_ok=False)
    origin = task['cold_started']
    preparation_started = time.perf_counter()
    tsp = directory / 'problem.tsp'
    with tsp.open('x', encoding='ascii') as stream:
        stream.write('NAME: grouped_pilot\nTYPE: TSP\nDIMENSION: '
                     + str(len(task['coords']))
                     + '\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n')
        for i, (x, y) in enumerate(task['coords']):
            stream.write(f'{i + 1} {x} {y}\n')
        stream.write('EOF\n')
    initial = directory / 'initial.tour'
    initial.write_text(tour_text(publisher.route), encoding='ascii')
    tour = directory / 'best.tour'
    par = directory / 'run.par'
    lines = [f'PROBLEM_FILE = {tsp.as_posix()}',
             f'OUTPUT_TOUR_FILE = {tour.as_posix()}',
             f'INITIAL_TOUR_FILE = {initial.as_posix()}',
             'INITIAL_TOUR_FRACTION = 1',
             'CANDIDATE_SET_TYPE = ALPHA', 'MAX_CANDIDATES = 5',
             'RUNS = 100000', 'MAX_TRIALS = 100000',
             'STOP_AT_OPTIMUM = NO', f'SEED = {task["seed"]}',
             'TIME_LIMIT = 86400', 'TOTAL_TIME_LIMIT = 86400', 'TRACE_LEVEL = 1']
    par.write_text('\n'.join(lines) + '\n', encoding='ascii')
    until = task['deadline'] - .20
    events.append(dict(stage='lkh_start', cold_seconds=time.perf_counter() - origin,
                       input_length=publisher.best))
    started = time.perf_counter()
    if started >= until:
        return dict(skipped=True, reason='no_remaining_budget',
                    preparation_seconds=started - preparation_started)
    seen = None
    exports = 0
    with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
        process = subprocess.Popen([task['lkh_binary'], str(par)], cwd=directory,
                                   stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while time.perf_counter() < until:
                try:
                    stat = tour.stat()
                    signature = (stat.st_mtime_ns, stat.st_size)
                    if signature != seen:
                        route = parse_tour(tour.read_text(encoding='ascii'), len(task['coords']))
                        # Publisher validates independently against coordinates.
                        if time.perf_counter() < until:
                            publisher(route, 'lkh')
                            exports += 1
                            seen = signature
                except (OSError, ValueError, UnicodeError):
                    pass
                if process.poll() is not None:
                    if process.returncode != 0:
                        raise RuntimeError(f'LKH exited {process.returncode}')
                    break
                time.sleep(.003)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
    ended = time.perf_counter()
    events.append(dict(stage='lkh_end', cold_seconds=ended - origin,
                       best_length=publisher.best))
    result = dict(started_cold_seconds=started - origin,
                  preparation_seconds=started - preparation_started,
                  elapsed_seconds=ended - started, observed_tour_files=exports,
                  returncode=process.returncode,
                  stop_guard_seconds=.20,
                  initial_tour_file_sha256=__import__('hashlib').sha256(initial.read_bytes()).hexdigest())
    write_new(directory / 'phase.json', result)
    return result


def main():
    task = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    directory = Path(task['directory'])
    origin, deadline = task['cold_started'], task['deadline']
    write_new(directory / 'ready.json', dict(pid=os.getpid(), method=task['method'],
              deadline=deadline, cold_started=origin, before_any_child=True))
    if json.loads(sys.stdin.readline()) != {'go': True, 'deadline': deadline}:
        raise ValueError('invalid supervisor handshake')
    publisher = Publisher(directory, task['coords'], deadline, origin)
    events = []
    stats = dict(method=task['method'], settings=task['settings'])
    try:
        began_nn = time.perf_counter()
        route = nearest_neighbor(task['coords'], task['seed'])
        publisher(route, 'nearest_neighbor')
        stats['nearest_neighbor_seconds'] = time.perf_counter() - began_nn
        stats['nearest_neighbor_length'] = publisher.best
        stats['nearest_neighbor_start_city'] = int(route[0])
        stats['nearest_neighbor_route_sha256'] = hashlib.sha256(
            json.dumps([int(v) for v in route], separators=(',', ':')).encode('ascii')).hexdigest()
        events.append(dict(stage='nearest_neighbor', cold_seconds=time.perf_counter() - origin,
                           length=publisher.best))
        preprocessing_started = time.perf_counter()
        if task['method'] in ('B', 'C'):
            settings = task['settings']
            local_deadline = origin + task['budget'] * settings['preprocessing_fraction_target']
            stats['local_deadline_cold_seconds'] = local_deadline - origin
            distances = distance_matrix(task['coords'])
            events.append(dict(stage='distance_matrix', cold_seconds=time.perf_counter() - origin))
            from group_search import optimize_grouped

            def emit(candidate, stage, counts):
                publisher(candidate, stage)

            stats['grouped_search'] = optimize_grouped(
                route, distances, task['method'], local_deadline, emit,
                group_size=settings['group_size'], candidate_count=settings['candidate_count'])
            publisher(route, 'grouped_final')
            del distances
        stats['group_preprocessing_seconds'] = time.perf_counter() - preprocessing_started
        stats['pre_lkh_length'] = publisher.best
        stats['pre_lkh_cold_seconds'] = time.perf_counter() - origin
        stats['preprocessing_target_overrun_seconds'] = (
            max(0., stats['pre_lkh_cold_seconds'] - task['budget']
                * task['settings']['preprocessing_fraction_target'])
            if task['method'] in ('B', 'C') else 0.)
        events.append(dict(stage='pre_lkh', cold_seconds=stats['pre_lkh_cold_seconds'],
                           length=publisher.best))
        stats['lkh'] = lkh_search(task, publisher, events)
        stats.update(events=events, best_length=publisher.best,
                     publications=publisher.sequence, sharing_retries=publisher.sharing_retries,
                     finished_cold_seconds=time.perf_counter() - origin)
        write_new(directory / 'worker_result.json', stats)
        write_new(directory / 'done.json', dict(at=time.perf_counter()))
    except BaseException:
        write_new(directory / 'error.json', dict(traceback=traceback.format_exc(), at=time.perf_counter()))
        raise


if __name__ == '__main__':
    main()
