#!/usr/bin/env python3
"""Renders the projects from build_project.jsx with aerender and checks the results.

  run_tests.py render LABEL     render everything into tests/out/LABEL, recording timings
  run_tests.py check LABEL      check one set of renders on its own terms
  run_tests.py compare A B      compare two sets of renders pixel by pixel
  run_tests.py perf LABEL       PreRender and render costs from that set's log

Needs numpy, and a plugin built with -DFEEDBACK_LOG=ON for the log-based results.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
AERENDER = "/Applications/Adobe After Effects 2026/aerender"
LOG = Path.home() / "Library/Logs/AEFeedback.log"

COMPS_96 = ["trail", "behind", "screen_sub", "add_sub", "grade_blur", "self_disp", "keyed", "keyed_edit"]
COMPS_72 = ["two_layers", "duplicate_fx", "neutral_ref", "neutral_luma", "neutral_red", "neutral_alpha_transparent",
            "neutral_alpha_half", "map_solid_opaque", "map_source_default", "map_solid_masked", "map_effects"]
NEUTRAL = ["neutral_ref", "neutral_luma", "neutral_red", "neutral_alpha_transparent", "neutral_alpha_half"]
# Rendered again alone in a new process: a full rebuild, with nothing cached.
COLD = {"trail": 60, "behind": 60, "screen_sub": 60, "add_sub": 60, "grade_blur": 60, "self_disp": 60, "keyed": 70,
        "duplicate_fx": 40, "perf_long": 299, "limit_700": 699,
        "preroll_800": 699, "preroll_100": 299}


def render(project, comp, folder, start=None, end=None):
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [AERENDER, "-project", str(OUT / project), "-comp", comp, "-RStemplate", "Best Settings",
           "-OMtemplate", "TIFF Sequence with Alpha", "-output", str(folder / "f_[#####].tif")]
    if start is not None:
        cmd += ["-s", str(start), "-e", str(end)]
    began = time.time()
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    took = time.time() - began
    frames = len(list(folder.glob("f_*.tif")))
    print(f"  {project} {comp} {start if start is not None else 'all'}: {took:.1f}s, {frames} files, exit {done.returncode}",
          flush=True)
    if done.returncode != 0 or frames == 0:
        print(done.stdout[-1500:])
    return took


def cmd_render(label):
    base = OUT / label
    base.mkdir(parents=True, exist_ok=True)
    log_start = LOG.stat().st_size if LOG.exists() else 0
    timing = {}
    for comp in COMPS_96 + COMPS_72 + ["perf_long", "limit_700", "preroll_800", "preroll_100"]:
        timing[comp] = render("feedback_16.aep", comp, base / comp)
    for depth in (8, 32):
        for comp in NEUTRAL:
            timing[f"{comp}_{depth}"] = render(f"feedback_{depth}.aep", comp, base / f"{comp}_{depth}")
    for comp, frame in COLD.items():
        timing[f"{comp}_cold"] = render("feedback_16.aep", comp, base / f"{comp}_cold", frame, frame)
    (base / "timing.json").write_text(json.dumps(
        {"timing": timing, "log": [log_start, LOG.stat().st_size if LOG.exists() else 0]}, indent=1))


def read_tiff(path):
    """AE's uncompressed 8-bit RGBA TIFFs. Pillow drops the alpha, which isn't labelled as one."""
    data = path.read_bytes()
    order = "<" if data[:2] == b"II" else ">"
    number = lambda fmt, at: np.frombuffer(data, order + fmt, 1, at)[0]
    tags = {}
    ifd = int(number("u4", 4))
    for i in range(int(number("u2", ifd))):
        at = ifd + 2 + i * 12
        tag, kind, count = int(number("u2", at)), int(number("u2", at + 2)), int(number("u4", at + 4))
        fmt = "u2" if kind == 3 else "u4"
        size = 2 if kind == 3 else 4
        start = at + 8 if size * count <= 4 else int(number("u4", at + 8))
        tags[tag] = [int(v) for v in np.frombuffer(data, order + fmt, count, start)]
    width, height, channels = tags[256][0], tags[257][0], tags[277][0]
    if tags.get(259, [1])[0] != 1 or tags[258][0] != 8 or channels != 4:
        raise ValueError(f"{path}: unexpected TIFF layout")
    pixels = b"".join(data[o:o + n] for o, n in zip(tags[273], tags[279]))
    return np.frombuffer(pixels, np.uint8).reshape(height, width, channels).astype(np.int32)


def load(label, comp, frame):
    return read_tiff(OUT / label / comp / f"f_{frame:05d}.tif")


def difference(a, b):
    d = np.abs(a - b)
    return int(d.max()), int((d.max(axis=-1) > 0).sum())


RESULTS = []


def report(ok, text):
    RESULTS.append(ok)
    print(("PASS  " if ok is True else "FAIL  " if ok is False else "INFO  ") + text)


def cmd_check(label):
    print("Rendering a frame alone matches playing up to it:")
    for comp, frame in COLD.items():
        try:
            peak, count = difference(load(label, comp, frame), load(label, comp + "_cold", frame))
        except FileNotFoundError as e:
            report(False, f"{comp}: missing {e.filename}")
            continue
        text = f"{comp}@{frame}: peak {peak}/255 across {count} pixels"
        limited = comp in ("limit_700", "preroll_100")
        report(None if limited else peak <= 1, text + (" (Alpha 1.0, beyond Max Pre-roll Frames)" if limited else ""))

    print("A later keyframe edit leaves earlier frames alone:")
    before = [difference(load(label, "keyed", f), load(label, "keyed_edit", f))[0] for f in range(0, 49)]
    after = [difference(load(label, "keyed", f), load(label, "keyed_edit", f))[0] for f in range(50, 96)]
    report(max(before) <= 1, f"frames 0-48 match to within rounding (peak {max(before)})")
    report(min(after) > 0, f"frames 50-95 all differ (smallest peak {min(after)})")

    print("Max Pre-roll Frames sets how far a rebuild reaches:")
    for comp, frame, limit in (("limit_700", 699, 600), ("preroll_800", 699, 800), ("preroll_100", 299, 100)):
        played = (load(label, comp, frame)[..., 3] > 8).sum()
        alone = (load(label, comp + "_cold", frame)[..., 3] > 8).sum()
        report(None, f"{comp}@{frame}, limit {limit}: {played} visible pixels played through, {alone} rendered alone")

    print("Two layers with identical settings keep their own images:")
    for frame in (30, 71):
        im = load(label, "two_layers", frame)
        w = im.shape[1] // 2
        visible = im[..., 3] > 8
        left, right = visible[:, :w], visible[:, w:]
        lb = im[:, :w][left].mean(axis=0) if left.any() else np.zeros(4)
        rb = im[:, w:][right].mean(axis=0) if right.any() else np.zeros(4)
        report(bool(lb[0] > lb[2] and rb[2] > rb[0]),
               f"frame {frame}: left mean RGB {lb[:3].round(0)}, right mean RGB {rb[:3].round(0)}")

    print("Displacement is neutral (compared with Max = 0 at the last frame):")
    for suffix, name in (("_8", "8-bit"), ("", "16-bit"), ("_32", "32-bit")):
        ref = load(label, "neutral_ref" + suffix, 71)
        for comp in NEUTRAL[1:]:
            peak, count = difference(ref, load(label, comp + suffix, 71))
            # 8-bit has no exact 50%: the nearest value, 128/255, still shifts a little.
            report(peak <= 1 if comp == "neutral_alpha_transparent" or suffix != "_8" else None,
                   f"{name} {comp}: peak {peak}/255 across {count} pixels")

    print("The Layer control reads masks and effects:")
    ref = load(label, "neutral_ref", 71)
    for comp, text in (("map_solid_masked", "mask respected"), ("map_effects", "effect respected")):
        peak, count = difference(load(label, comp, 71), ref)
        report(peak <= 1, f"{text}: peak {peak}/255 across {count} pixels")
    peak, count = difference(load(label, "map_source_default", 71), load(label, "map_solid_opaque", 71))
    report(peak <= 1, f"Source ignores the mask: peak {peak}/255 against an unmasked map")


def cmd_compare(a, b):
    print(f"{a} against {b}:")
    for folder in sorted(p.name for p in (OUT / a).iterdir() if p.is_dir()):
        if not (OUT / b / folder).is_dir():
            continue
        names = sorted(p.name for p in (OUT / a / folder).glob("f_*.tif"))
        worst, moved, where = 0, 0, ""
        for name in names:
            peak, count = difference(read_tiff(OUT / a / folder / name), read_tiff(OUT / b / folder / name))
            if peak > worst:
                worst, moved, where = peak, count, name
        report(worst == 0, f"{folder}: {len(names)} frames, peak {worst}/255 ({moved} pixels in {where})")


def cmd_perf(label):
    info = json.loads((OUT / label / "timing.json").read_text())
    with open(LOG, "rb") as f:
        f.seek(info["log"][0])
        lines = f.read(info["log"][1] - info["log"][0]).decode(errors="replace").splitlines()
    pre = [(int(m[1]), float(m[3]), int(m[2])) for m in
           (re.search(r"^prerender .*window=(\d+).*checkouts=(\d+).*ms=([\d.]+)", l) for l in lines) if m]
    ren = [float(m[1]) for m in (re.search(r"^frame=.*took=([\d.]+)ms", l) for l in lines) if m]
    print(f"{label}: {len(pre)} PreRenders totalling {sum(p[1] for p in pre) / 1e3:.1f}s, "
          f"{len(ren)} renders totalling {sum(ren) / 1e3:.1f}s, CPU renders {sum('CPU render' in l for l in lines)}")
    for lo, hi in ((0, 9), (10, 49), (50, 149), (150, 299), (300, 700)):
        q = [p for p in pre if lo <= p[0] <= hi]
        if q:
            print(f"  window {lo}-{hi}: {len(q)} calls, median {statistics.median(p[1] for p in q):.2f} ms, "
                  f"median {statistics.median(p[2] for p in q)} checkouts")
    t = info["timing"]
    print("  wall clock: " + ", ".join(f"{k} {t[k]:.1f}s" for k in ("trail", "perf_long", "limit_700", "perf_long_cold")))


def main():
    action = sys.argv[1]
    {"render": cmd_render, "check": cmd_check, "compare": cmd_compare, "perf": cmd_perf}[action](*sys.argv[2:])
    if False in RESULTS:
        print(f"\n{RESULTS.count(False)} failed, {RESULTS.count(True)} passed")
        sys.exit(1)
    if RESULTS:
        print(f"\nAll {RESULTS.count(True)} checks passed")


if __name__ == "__main__":
    main()
