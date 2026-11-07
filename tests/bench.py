#!/usr/bin/env python3
"""Measures speed on bench.aep from bench_project.jsx.  bench.py run LABEL | compare A B | canvas

Needs a plugin built with -DFEEDBACK_LOG=ON, which takes a fresh GUID salt from each render, so AE's
disk cache never serves frames and everything is measured.
"""
import json
import re
import shutil
import sys

import numpy as np

from run_tests import LOG, OUT, read_tiff, render

BENCHES = ["simple", "blur", "grade", "map", "trailmap", "sub", "add", "late", "full", "wander", "expr", "small"]
COPIES = 3
LAST = 299


def new_log(start):
    with open(LOG, "rb") as f:
        f.seek(start)
        return f.read().decode(errors="replace").splitlines()


def renders(lines):
    pattern = r"^frame=(\d+) computed=(\d+).*fetchMs=([\d.]+) gpuMs=([\d.]+).*took=([\d.]+)ms err=0"
    return [(int(m[1]), int(m[2]), float(m[3]), float(m[4]), float(m[5])) for m in map(lambda l: re.search(pattern, l), lines) if m]


def prerender_ms(lines):
    return sum(float(m[1]) for m in map(lambda l: re.search(r"^prerender .*ms=([\d.]+)", l), lines) if m)


def cmd_run(label):
    base = OUT / f"bench_{label}"
    results = {}
    for name in BENCHES:
        cold = []
        for copy in range(COPIES):
            start = LOG.stat().st_size
            folder = base / "z"
            render("bench.aep", f"{name}_{copy}", folder, LAST, LAST)
            r = [x for x in renders(new_log(start)) if x[0] == LAST]
            if r:
                best = max(r, key=lambda x: x[4])
                cold.append({"computed": best[1], "fetch": best[2], "gpu": best[3], "took": best[4]})
            shutil.rmtree(folder, ignore_errors=True)
        # Frames rendered forward, without the frame left over from the rebuild above.
        start = LOG.stat().st_size
        folder = base / name
        play_wall = render("bench.aep", f"{name}_0", folder, 0, LAST - 1)
        shutil.rmtree(folder, ignore_errors=True)  # uncompressed frames: gigabytes per comp
        lines = new_log(start)
        r = renders(lines)
        results[name] = {
            "cold": cold,
            "play_took": sum(x[4] for x in r),
            "play_gpu": sum(x[3] for x in r),
            "play_fetch": sum(x[2] for x in r),
            "play_frames": len(r),
            "play_prerender": prerender_ms(lines),
            "play_wall": play_wall,
        }
    (OUT / f"bench_{label}.json").write_text(json.dumps(results, indent=1))
    show(label, results)


def show(label, results):
    print(f"{label:>10}: {'rebuild ms':>11} {'gpu':>8} {'fetch':>8} {'other':>8} | {'play ms':>9} {'gpu':>8} {'fetch':>8} {'prerender':>9}")
    for name, r in results.items():
        if not r["cold"]:
            print(f"{name:>10}: no rebuild measured")
            continue
        c = min(r["cold"], key=lambda x: x["took"])
        print(f"{name:>10}: {c['took']:>11.0f} {c['gpu']:>8.0f} {c['fetch']:>8.0f} {c['took'] - c['gpu'] - c['fetch']:>8.0f} | "
              f"{r['play_took']:>9.0f} {r['play_gpu']:>8.0f} {r['play_fetch']:>8.0f} {r['play_prerender']:>9.0f}")


def cmd_compare(a, b):
    ra, rb = (json.loads((OUT / f"bench_{x}.json").read_text()) for x in (a, b))
    print(f"{'':>10}  rebuild ms: {a} -> {b}             play ms: {a} -> {b}")
    for name in ra:
        if name not in rb or not (ra[name]["cold"] and rb[name]["cold"]):
            continue
        ca, cb = (min(x[name]["cold"], key=lambda c: c["took"])["took"] for x in (ra, rb))
        pa, pb = ra[name]["play_took"], rb[name]["play_took"]
        print(f"{name:>10}  {ca:>9.0f} -> {cb:>7.0f} ({100 * (cb - ca) / ca:+5.0f}%)    {pa:>8.0f} -> {pb:>8.0f} ({100 * (pb - pa) / pa:+5.0f}%)")


def cmd_canvas():
    """How much of the canvas holds anything: the most that tracking a changed area could skip."""
    print(f"{'':>10}  skippable canvas: mean, best frame  (GPU share of a played frame, so saving at most)")
    results = json.loads(sorted(OUT.glob("bench_*.json"), key=lambda p: p.stat().st_mtime)[-1].read_text())
    for name in BENCHES:
        folder = OUT / "bench_canvas" / name
        start = LOG.stat().st_size
        render("bench.aep", f"{name}_1", folder, 0, LAST - 1)
        sizes = {}
        for line in new_log(start):
            m = re.search(r"^prerender frame=(\d+) .*canvas=(\d+)x(\d+)", line)
            if m:
                sizes[int(m[1])] = int(m[2]) * int(m[3])
        skip = []
        for frame in range(15, LAST, 15):
            ys, xs = np.nonzero(read_tiff(folder / f"f_{frame:05d}.tif")[..., 3])
            if frame in sizes and len(xs):
                skip.append(1 - (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / sizes[frame])
        shutil.rmtree(folder, ignore_errors=True)
        r = results.get(name)
        share = r["play_gpu"] / max(r["play_took"] + r["play_prerender"], 1) if r else 0
        if skip:
            print(f"{name:>10}  {100 * np.mean(skip):>5.0f}%  {100 * max(skip):>5.0f}%   GPU share {100 * share:>3.0f}%, "
                  f"saving at most {100 * np.mean(skip) * share:>3.0f}%")


if __name__ == "__main__":
    {"run": cmd_run, "compare": cmd_compare, "canvas": cmd_canvas}[sys.argv[1]](*sys.argv[2:])
