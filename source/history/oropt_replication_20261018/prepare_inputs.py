"""Freeze new held-out TSP inputs before any formal solver job.

This is common, offline input preparation. It is outside every method's 30 s
budget and provides no route, distance matrix, candidate list, compiled kernel,
or other method-specific solver work. Each formal run must pay those costs.

Only new files are created. An existing output directory is never overwritten.
The four public instances are copied byte for byte and parsed using Decimal:
nonintegral coordinates, non-EUC_2D distance types, or unexpected dimensions
cause an error. No coordinate rounding, scaling, or distance conversion occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PUBLIC_DIRECTORY = Path(
    "D:/My project/outputs/paper2_vsr_lkh/VSR-LKH-V2/Instances"
)
PUBLIC_INSTANCES = (("pcb442", 442), ("rat575", 575),
                    ("nrw1379", 1379), ("pr2392", 2392))
FIRST_COORDINATE_SEED = 20261018230001
CITY_COUNTS = (1000, 3000, 5000, 8000)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new_json(path: Path, value: object) -> str:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
    return sha_bytes(encoded)


def parse_public(path: Path, expected_name: str, expected_n: int):
    raw = path.read_bytes()
    text = raw.decode("ascii")
    headers: dict[str, str] = {}
    rows: dict[int, list[int]] = {}
    in_coordinates = False
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line == "EOF":
            break
        if line == "NODE_COORD_SECTION":
            if in_coordinates:
                raise ValueError(f"{path}:{line_number}: duplicate coordinate section")
            in_coordinates = True
            continue
        if not in_coordinates:
            if ":" not in line:
                raise ValueError(f"{path}:{line_number}: unexpected header")
            key, value = line.split(":", 1)
            headers[key.strip().upper()] = value.strip()
            continue
        tokens = line.split()
        if len(tokens) != 3:
            raise ValueError(f"{path}:{line_number}: expected node ID and two coordinates")
        try:
            city = int(tokens[0])
            decimals = [Decimal(token) for token in tokens[1:]]
        except (ValueError, InvalidOperation) as exc:
            raise ValueError(f"{path}:{line_number}: invalid node record") from exc
        if any(not value.is_finite() or value != value.to_integral_value()
               for value in decimals):
            raise ValueError(f"{path}:{line_number}: coordinate is not an exact integer")
        if city in rows:
            raise ValueError(f"{path}:{line_number}: duplicate node ID")
        rows[city] = [int(value) for value in decimals]
    if not in_coordinates:
        raise ValueError(f"{path}: NODE_COORD_SECTION missing")
    if headers.get("NAME") != expected_name:
        raise ValueError(f"{path}: unexpected NAME {headers.get('NAME')!r}")
    if headers.get("TYPE") != "TSP" or headers.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError(f"{path}: only symmetric EUC_2D TSP is supported")
    if int(headers.get("DIMENSION", "-1")) != expected_n:
        raise ValueError(f"{path}: unexpected DIMENSION")
    if set(rows) != set(range(1, expected_n + 1)):
        raise ValueError(f"{path}: city IDs must be exactly 1 through DIMENSION")
    coords = [rows[city] for city in range(1, expected_n + 1)]
    return raw, headers, coords


def prepare(output: Path, public_directory: Path):
    # Parse and validate all public input bytes before creating the output tree.
    public = []
    for instance_id, n in PUBLIC_INSTANCES:
        original = public_directory / f"{instance_id}.tsp"
        raw, headers, coords = parse_public(original, instance_id, n)
        public.append((instance_id, n, original, raw, headers, coords))

    output.mkdir(parents=True, exist_ok=False)
    raw_directory = output / "raw_tsp"
    raw_directory.mkdir(exist_ok=False)
    maps = []
    synthetic_index = 0

    def register(item, raw_tsp_sha256=None):
        path = output / f"{item['id']}.json"
        digest = write_new_json(path, item)
        summary = {key: value for key, value in item.items() if key != "coords"}
        summary.update(path=path.name, absolute_path=str(path.resolve()),
                       sha256=digest, raw_tsp_sha256=raw_tsp_sha256)
        maps.append(summary)
        print(json.dumps(dict(id=item["id"], n=item["n"], bytes=path.stat().st_size,
                              sha256=digest), ensure_ascii=False), flush=True)

    for n in CITY_COUNTS:
        for family, short_name in (("uniform", "u"), ("clustered", "c")):
            seed = FIRST_COORDINATE_SEED + synthetic_index
            synthetic_index += 1
            rng = np.random.default_rng(seed)
            if family == "uniform":
                coords = rng.integers(0, 100001, size=(n, 2), dtype=np.int64)
                generation = "default_rng(seed).integers(0,100001,(n,2),dtype=int64)"
            else:
                centers = rng.integers(10000, 90001, size=(8, 2), dtype=np.int64)
                labels = rng.integers(0, 8, size=n)
                points = centers[labels] + rng.normal(0.0, 1500.0, size=(n, 2))
                coords = np.rint(np.clip(points, 0, 100000)).astype(np.int64)
                generation = ("8 centers integers(10000,90001,(8,2)); "
                              "labels integers(0,8,n); centers[labels] + "
                              "normal(0,1500,(n,2)); clip[0,100000]; rint; int64")
            register(dict(id=f"fresh_{short_name}{n}", n=n, family=family,
                          stratum=f"synthetic_{family}", coordinate_seed=seed,
                          edge_weight_type="EUC_2D", coords=coords.tolist(),
                          source_provenance=dict(kind="new_synthetic_instance",
                              generator_script=str(Path(__file__).resolve()),
                              numpy_version=np.__version__, bit_generator="PCG64",
                              generation=generation, prior_four_maps_reused=False)))

    for instance_id, n, original, raw, headers, coords in public:
        raw_path = raw_directory / f"{instance_id}.tsp"
        with raw_path.open("xb") as stream:
            stream.write(raw)
        raw_sha = sha_bytes(raw)
        register(dict(id=instance_id, n=n, family="tsplib", stratum="tsplib",
                      coordinate_seed=None, edge_weight_type="EUC_2D", coords=coords,
                      source_provenance=dict(kind="public_TSPLIB_instance",
                          original_local_path=str(original.resolve()),
                          original_repository="VSR-LKH-V2/Instances",
                          original_header=headers,
                          raw_tsp_path=f"raw_tsp/{instance_id}.tsp",
                          raw_tsp_sha256=raw_sha,
                          byte_for_byte_raw_copy=True,
                          parser="Decimal; exact integer coordinates required",
                          coordinate_transform="none; node IDs 1..n become list positions 0..n-1",
                          optimum_or_reference_tour_used=False)), raw_sha)

    manifest = dict(schema=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    map_count=len(maps), maps=maps,
                    registered_before_first_formal_job=True,
                    preparation_outside_per_run_budget=True,
                    common_preparation="input creation/parsing/copying/hashing only",
                    no_solver_work_performed=True,
                    previous_four_maps_included=False,
                    synthetic_coordinate_seeds=list(range(FIRST_COORDINATE_SEED,
                                                          FIRST_COORDINATE_SEED + 8)),
                    source_script_sha256=sha_bytes(Path(__file__).read_bytes()),
                    distance_rule="EUC_2D: floor(sqrt(dx*dx+dy*dy)+0.5); closed tours",
                    python_version=sys.version, numpy_version=np.__version__)
    write_new_json(output / "manifest.json", manifest)
    print(f"MANIFEST {output / 'manifest.json'}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "datasets")
    parser.add_argument("--public-directory", type=Path, default=PUBLIC_DIRECTORY)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.public_directory.resolve())


if __name__ == "__main__":
    main()
