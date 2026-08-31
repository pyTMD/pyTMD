#!/usr/bin/env python3
"""
Benchmark FES netCDF: full load vs chunked open + ``tmd.crop``.

Not collected by pytest (no ``test_`` prefix). Run from the pyTMD repo::

    PYTHONPATH=. python test/perf_fes_windowed_read.py \\
        --tide-dir /path/to/parent/of/fes2022b \\
        --repeats 3

Prints a markdown table suitable for issue/PR notes.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np

import pyTMD.io.FES as FES


def _time_call(fn, repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _force_load(ds) -> tuple[int, tuple[int, ...]]:
    """Materialize first data variable; return nbytes and shape."""
    name = next(iter(ds.data_vars))
    arr = np.asarray(ds[name].values)
    return int(arr.nbytes), tuple(int(s) for s in arr.shape)


def _fmt(seconds: list[float]) -> str:
    med = statistics.median(seconds)
    if len(seconds) > 1:
        return f"{med:.3f} s (min {min(seconds):.3f}, max {max(seconds):.3f})"
    return f"{med:.3f} s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tide-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TIDE_MODEL_PATH",
                str(Path.home() / "data" / "tide_models"),
            )
        ),
        help="Directory containing fes2022b/ (or set TIDE_MODEL_PATH)",
    )
    parser.add_argument(
        "--subdir",
        default="fes2022b/ocean_tide_20241025",
        help="Relative path under tide-dir to constituent netCDFs",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=[149.5, 152.5, -24.5, -21.5],
        help="Regional box (caller-buffered). Default ≈ Fitzroy + 0.5° halo",
    )
    parser.add_argument(
        "--constituents",
        nargs="+",
        default=["m2", "s2", "k1", "o1", "n2", "k2", "p1", "q1"],
        help="Constituent stems for multi-file open_mfdataset",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    ocean = args.tide_dir / args.subdir
    if not ocean.is_dir():
        raise SystemExit(
            f"Missing FES directory: {ocean}\n"
            "Pass --tide-dir (parent of fes2022b/) or set TIDE_MODEL_PATH."
        )

    m2 = ocean / "m2_fes2022.nc"
    if not m2.is_file():
        raise SystemExit(f"Missing {m2}")

    bounds = list(args.bounds)
    files = []
    for c in args.constituents:
        p = ocean / f"{c}_fes2022.nc"
        if p.is_file():
            files.append(p)
    if not files:
        raise SystemExit("No constituent files found for multi-file bench")

    rows: list[tuple[str, str, str, str]] = []

    # --- single file ---
    def open_full_one():
        ds = FES.open_fes_netcdf(m2, group="z")
        _force_load(ds)

    def open_crop_one():
        ds = FES.open_fes_netcdf(m2, group="z", chunks={})
        ds = ds.tmd.crop(bounds, buffer=0)
        _force_load(ds)

    # warm filesystem once
    open_full_one()
    open_crop_one()

    t_full = _time_call(open_full_one, args.repeats)
    t_win = _time_call(open_crop_one, args.repeats)
    ds_full = FES.open_fes_netcdf(m2, group="z")
    nbytes_full, shape_full = _force_load(ds_full)
    ds_win = FES.open_fes_netcdf(m2, group="z", chunks={}).tmd.crop(
        bounds, buffer=0
    )
    nbytes_win, shape_win = _force_load(ds_win)
    speedup = statistics.median(t_full) / max(statistics.median(t_win), 1e-12)

    rows.append(
        (
            f"single file (`{m2.name}`)",
            _fmt(t_full),
            _fmt(t_win),
            f"{speedup:.1f}×; shape {shape_full}→{shape_win}; "
            f"{nbytes_full / 1e6:.1f}→{nbytes_win / 1e6:.2f} MB",
        )
    )

    # --- multi file ---
    def open_full_many():
        ds = FES.open_mfdataset(files, format="netcdf", group="z")
        for name in ds.data_vars:
            _ = np.asarray(ds[name].values)

    def open_crop_many():
        ds = FES.open_mfdataset(files, format="netcdf", group="z", chunks={})
        ds = ds.tmd.crop(bounds, buffer=0)
        for name in ds.data_vars:
            _ = np.asarray(ds[name].values)

    open_full_many()
    open_crop_many()
    t_full_m = _time_call(open_full_many, args.repeats)
    t_win_m = _time_call(open_crop_many, args.repeats)
    speedup_m = statistics.median(t_full_m) / max(
        statistics.median(t_win_m), 1e-12
    )
    rows.append(
        (
            f"open_mfdataset ({len(files)} constituents)",
            _fmt(t_full_m),
            _fmt(t_win_m),
            f"{speedup_m:.1f}×; files={[p.stem.split('_')[0] for p in files]}",
        )
    )

    print("## FES windowed-read benchmark")
    print()
    print(f"- tide dir: `{ocean}`")
    print(f"- bounds: `{bounds}` (xmin, xmax, ymin, ymax)")
    print(f"- repeats: {args.repeats} (median reported)")
    print("- path: chunked open + ``Dataset.tmd.crop``")
    print(f"- pyTMD FES: `{Path(FES.__file__).resolve()}`")
    print()
    print("| Case | Full open | Chunked open + crop | Notes |")
    print("|------|-----------|---------------------|-------|")
    for case, full, win, notes in rows:
        print(f"| {case} | {full} | {win} | {notes} |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
