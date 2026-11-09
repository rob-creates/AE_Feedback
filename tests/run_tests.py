#!/usr/bin/env python3
"""Renders the projects from build_project.jsx with aerender and checks the results.

  run_tests.py render LABEL     render everything into tests/out/LABEL, recording timings
  run_tests.py check LABEL      check one set of renders on its own terms
  run_tests.py compare A B      compare two sets of renders pixel by pixel
  run_tests.py perf LABEL       PreRender and render costs from that set's log

Needs numpy, and a plugin built with -DFEEDBACK_LOG=ON for the log-based results.
"""
import colorsys
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

COMPS = ["trail", "behind", "screen_sub", "add_sub", "grade_blur", "self_disp", "levels_inverted",
         "levels_inverted_sub", "keyed", "keyed_edit",
         "alpha_min", "alpha_min_ref", "alpha_min_fade", "alpha_min_sub", "empties_midway",
         "shape_trail", "shape_sub", "shape_mapped", "shape_as_map",
         "two_layers", "duplicate_fx", "neutral_ref", "neutral_luma", "neutral_red", "neutral_alpha_transparent",
         "neutral_alpha_half", "map_solid_opaque", "map_source_default", "map_solid_masked", "map_effects",
         "empty_layer", "empty_mapped", "layer_own", "layer_none", "layer_own_effects",
         "perf_long", "limit_600", "limit_800", "limit_100",
         "trim_48", "trim_48_plain", "trim_24", "trim_24_plain", "trim_start", "trim_start_plain",
         "trim_stretch", "trim_stretch_plain",
         *[f"blend_{m}{v}" for m in ("multiply", "lighten", "darken", "overlay", "add_subtract", "difference")
           for v in ("", "_sub", "_fade")],
         "transform_position", "transform_scale", "transform_scale_xy", "transform_rotate", "transform_sub",
         "edge_luminance", "edge_colour", "edge_trail"]
NEUTRAL = ["neutral_ref", "neutral_luma", "neutral_red", "neutral_alpha_transparent", "neutral_alpha_half"]
# Rendered again alone in a new process: a full rebuild, with nothing cached.
COLD = {"trail": 60, "behind": 60, "screen_sub": 60, "add_sub": 60, "grade_blur": 60, "self_disp": 60, "keyed": 70,
        "alpha_min": 60, "alpha_min_fade": 60, "alpha_min_sub": 60,
        "duplicate_fx": 40, "perf_long": 299, "limit_600": 699,
        "limit_800": 699, "limit_100": 299, "trim_48": 60, "trim_start": 60, "trim_stretch": 60,
        "blend_difference": 3, "blend_overlay_sub": 29, "transform_rotate": 12, "transform_sub": 60, "edge_trail": 60}
DEEP = 3999  # deep_alpha1's last frame


def render(project, comp, folder, start=None, end=None):
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [AERENDER, "-project", str(OUT / project), "-comp", comp, "-RStemplate", "Best Settings",
           "-OMtemplate", "TIFF Sequence with Alpha", "-output", str(folder / "f_[#####].tif")]
    if start is not None:
        cmd += ["-s", str(start), "-e", str(end)]
    # A logging build mixes this into its GUIDs, so AE's disk cache can't serve an earlier render's frames.
    env = dict(os.environ, FEEDBACK_GUID_SALT=str(time.time_ns()))
    began = time.time()
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
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
    for comp in COMPS:
        timing[comp] = render("feedback_16.aep", comp, base / comp)
    for comp in ("software_trail", "software_plain"):
        timing[comp] = render("feedback_software.aep", comp, base / comp, 0, 23)
    for depth in (8, 32):
        for comp in NEUTRAL:
            timing[f"{comp}_{depth}"] = render(f"feedback_{depth}.aep", comp, base / f"{comp}_{depth}")
    for comp, frame in COLD.items():
        timing[f"{comp}_cold"] = render("feedback_16.aep", comp, base / f"{comp}_cold", frame, frame)
    timing["deep_alpha1"] = render("feedback_16.aep", "deep_alpha1", base / "deep_alpha1", DEEP, DEEP)
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


def visible(image):
    """How many pixels can be seen."""
    return int((image[..., 3] > 8).sum())


def difference(a, b):
    d = np.abs(a - b)
    return int(d.max()), int((d.max(axis=-1) > 0).sum())


RESULTS = []


def report(ok, text):
    ok = None if ok is None else bool(ok)  # numpy's own booleans aren't True or False
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
        if comp in ("limit_600", "limit_100"):
            report(None, text + " (Alpha Max 1.0, beyond Trail Rebuild Limit)")
        elif comp == "duplicate_fx":
            # A rebuild stops once the effect's own Alpha has faded older frames to nothing, but the
            # second copy's input is a trail already, which fades slower. The faintest tail differs.
            report(None, text + " (fed by another trail, so a rebuild drops a sliver of the oldest tail)")
        else:
            # Add sums the same steps in another order, which can round a pixel the other way.
            report(peak <= {"add_sub": 2}.get(comp, 1), text)

    print("A later keyframe edit leaves earlier frames alone:")
    before = [difference(load(label, "keyed", f), load(label, "keyed_edit", f))[0] for f in range(0, 49)]
    after = [difference(load(label, "keyed", f), load(label, "keyed_edit", f))[0] for f in range(50, 96)]
    report(max(before) <= 1, f"frames 0-48 match to within rounding (peak {max(before)})")
    report(min(after) > 0, f"frames 50-95 all differ (smallest peak {min(after)})")

    print("Trail Rebuild Limit sets how far a rebuild reaches:")
    for comp, frame, limit in (("limit_600", 699, 600), ("limit_800", 699, 800), ("limit_100", 299, 100)):
        played = visible(load(label, comp, frame))
        alone = visible(load(label, comp + "_cold", frame))
        report(None, f"{comp}@{frame}, limit {limit}: {played} visible pixels played through, {alone} rendered alone")

    counts = [visible(load(label, "limit_100", f)) for f in range(299)]
    worst = min(b / a for a, b in zip(counts, counts[1:]) if a)
    report(worst > 0.99, f"playing forward, the limit doesn't cut the trail: worst frame-to-frame change x{worst:.3f}")

    print("Alpha Max 1.0, 4000 frames in, rendered alone:")
    took = json.loads((OUT / label / "timing.json").read_text())["timing"]["deep_alpha1"]
    try:
        seen = visible(load(label, "deep_alpha1", DEEP))
        report(seen > 0, f"rendered in {took:.1f}s with {seen} visible pixels")
    except FileNotFoundError:
        report(False, f"no frame after {took:.1f}s")

    print("Two layers with identical settings keep their own images:")
    for frame in (30, 71):
        im = load(label, "two_layers", frame)
        w = im.shape[1] // 2
        shown = im[..., 3] > 8
        left, right = shown[:, :w], shown[:, w:]
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

    print("Map Layer reads masks and effects:")
    ref = load(label, "neutral_ref", 71)
    for comp, text in (("map_solid_masked", "mask respected"), ("map_effects", "effect respected")):
        peak, count = difference(load(label, comp, 71), ref)
        report(peak <= 1, f"{text}: peak {peak}/255 across {count} pixels")
    peak, count = difference(load(label, "map_source_default", 71), load(label, "map_solid_opaque", 71))
    report(peak <= 1, f"Source ignores the mask: peak {peak}/255 against an unmasked map")

    print("Map Layer:")
    # A trail displaced by its own alpha differs from one displaced by the layer's image.
    peak, count = difference(load(label, "layer_own", 71), load(label, "layer_none", 71))
    report(peak > 8, f"its own layer reads the layer, None reads the trail: differ by peak {peak}/255 across {count} pixels")
    peak, _ = difference(load(label, "layer_none", 71), load(label, "neutral_ref", 71))
    report(peak > 8, f"None still displaces, using the trail: differs from no shift by peak {peak}/255")
    peak, count = difference(load(label, "layer_own_effects", 71), load(label, "layer_none", 71))
    report(peak <= 1, f"its own layer with Effects & Masks uses the trail, as None: peak {peak}/255 across {count} pixels")

    print("Without a GPU:")
    peak, count = difference(load(label, "software_trail", 23), load(label, "software_plain", 23))
    report(peak <= 1, f"the layer passes through with no trail: peak {peak}/255 across {count} pixels")

    print("A trimmed layer's trail starts at its in-point:")
    for comp, first in (("trim_48", 48), ("trim_24", 24), ("trim_start", 48), ("trim_stretch", 48)):
        peak, count = difference(load(label, comp, first), load(label, comp + "_plain", first))
        report(peak <= 1, f"{comp}@{first} shows only the layer: peak {peak}/255 across {count} pixels")
        later, plain = visible(load(label, comp, first + 6)), visible(load(label, comp + "_plain", first + 6))
        report(later > plain, f"{comp}@{first + 6} has a trail: {later} visible pixels against {plain} for the layer")
    short, whole = visible(load(label, "trim_48", 52)), visible(load(label, "trail", 52))
    report(short < whole, f"4 frames in, the trail is shorter than an untrimmed one: {short} visible pixels against {whole}")

    print("A layer with nothing in it:")
    for comp, last in (("empty_layer", 47), ("empty_mapped", 47)):
        frames = len(list((OUT / label / comp).glob("f_*.tif")))
        report(bool(frames == last + 1 and load(label, comp, last)[..., 3].max() == 0),
               f"{comp}: {frames} of {last + 1} frames rendered, all transparent")
    # Before the layer empties, soon after, and once the trail has faded.
    seen = [visible(load(label, "empties_midway", f)) for f in (29, 35, 95)]
    report(seen[0] > seen[1] > 0 and seen[2] == 0, f"empties_midway: the trail fades after the input stops: {seen} visible pixels")

    print("Inverted Levels, with and without sub-steps:")
    centre = lambda comp, f: load(label, comp, f)[170:190, 310:330, :3].reshape(-1, 3).mean(axis=0)
    # Worked out step by step: the trail inverted once a frame, faded, and laid over the dot. Sub-steps
    # split the fade and the dot between them. Dividing the inversion itself used to flatten it to grey.
    dot = np.array([1, 0.1, 0.1, 1.0])  # premultiplied RGBA
    for comp, n in (("levels_inverted", 1), ("levels_inverted_sub", 2)):
        trail = dot.copy()
        for f in range(1, 62):
            for step in range(n):
                if step == n - 1:
                    trail[:3] = (1 - np.maximum(trail[:3] / trail[3], 0)) * trail[3]
                trail = trail * 0.9 ** (1 / n)
                trail = trail + dot * (1 - trail[3])
            if f in (60, 61):
                want = np.clip(trail[:3] / trail[3], 0, 1) * 255
                seen = centre(comp, f)
                report(np.abs(seen - want).max() <= 2, f"{comp}@{f}: centre RGB {seen.round(0)}, expected {want.round(0)}")

    print("Blend modes: the dot over its trail, the dot's hue turned half way:")
    dot = np.array([0.8, 0.4, 0.2])
    h, l, sat = colorsys.rgb_to_hls(*dot)
    trail = np.array(colorsys.hls_to_rgb((h + 0.5) % 1, l, sat))
    def blend_colours(bottom, top, mode):
        return {"multiply": bottom * top, "lighten": np.maximum(bottom, top), "darken": np.minimum(bottom, top),
                "overlay": np.where(bottom <= 0.5, 2 * bottom * top, 1 - 2 * (1 - bottom) * (1 - top)),
                "add_subtract": bottom + 2 * top - 1, "difference": np.abs(bottom - top)}[mode]
    expected = {mode: blend_colours(trail, dot, mode)
                for mode in ("multiply", "lighten", "darken", "overlay", "add_subtract", "difference")}
    for mode, want in expected.items():
        want = np.clip(want, 0, 1) * 255
        seen = load(label, "blend_" + mode, 1)[180, 320, :3]
        report(np.abs(seen - want).max() <= 2, f"{mode}: centre RGB {seen}, expected {want.round(0)}")

    # The dot over its own trail, worked out step by step. With sub-steps, each takes a share of the dot
    # and the current frame goes over the top whole. At full strength each time, the look would build
    # up with the count: up to 94/255 apart in a simulation, against about 2 with shares.
    print("Blend modes keep their look with sub-steps:")
    def blended(mode, n, alpha=0.9, frames=30):
        def over(bottom, top, share=1.0):
            top = top * share
            b, t = bottom[3], top[3]
            straight = lambda c: c[:3] / c[3] if c[3] > 1e-5 else np.zeros(3)
            rgb = (1 - t) * bottom[:3] + (1 - b) * top[:3] + t * b * blend_colours(straight(bottom), straight(top), mode)
            return np.r_[np.clip(rgb, -1024, 1024), min(t + b - t * b, 1)]
        step, image, dot4 = alpha ** (1 / n), np.zeros(4), np.r_[dot, 1.0]
        share = alpha * (1 - step) / (1 - alpha) if n > 1 else 1.0
        for f in range(frames):
            for _ in range(1 if f == 0 else n):
                image = over(image * step, dot4, share)
        shown = over(image, dot4) if n > 1 else image
        return np.clip(shown[:3] / shown[3], 0, 1) * 255
    for mode in expected:
        whole, split = (load(label, f"blend_{mode}{v}", 29)[180, 320, :3] for v in ("_fade", "_sub"))
        want_whole, want_split = blended(mode, 1), blended(mode, 4)
        report(np.abs(whole - want_whole).max() <= 2 and np.abs(split - want_split).max() <= 2 and
               np.abs(whole - split).max() <= 5,
               f"{mode}: centre RGB {whole} without sub-steps, {split} with 3; worked out {want_whole.round(0)} and {want_split.round(0)}")

    print("Transform carries each frame's dot, fading by Alpha Max 0.9 a frame:")
    def alpha_at(comp, frame, x, y):
        return int(load(label, comp, frame)[y, x, 3])
    for k in range(1, 6):
        seen, want = alpha_at("transform_position", 11, 200 + 20 * k, 180), 255 * 0.9 ** k
        report(abs(seen - want) <= 4, f"Position 20 to the right, {k} frames back at x={200 + 20 * k}: alpha {seen}, expected {want:.0f}")
    for comp, y, text in (("transform_scale", 180, "Uniform Scale 150% (Width 300% left out)"),
                          ("transform_scale_xy", 220, "Scale Width 150%, Height 100%")):
        for k in range(1, 5):
            x = 320 + 40 * 1.5 ** k
            seen, want = alpha_at(comp, 11, int(x), y), 255 * 0.9 ** k
            report(abs(seen - want) <= 4, f"{text}, {k} frames back at ({x:.0f}, {y}): alpha {seen}, expected {want:.0f}")
    for k, (x, y) in enumerate([(320, 280), (220, 180), (320, 80)], 1):
        # Older copies land here too, but behind the dot, which covers them each frame before it turns.
        seen, want = alpha_at("transform_rotate", 12, x, y), 255 * 0.9 ** k
        report(abs(seen - want) <= 4, f"Rotation 90, a quarter turn clockwise {k} times at ({x}, {y}): alpha {seen}, expected {want:.0f}")

    print("Edges: the dot's outline over its trail, in front of the dot:")
    for comp, least in (("edge_luminance", 40), ("edge_colour", 200)):
        im = load(label, comp, 1)
        centre, ring, beyond = im[180, 320], im[180, 361], im[180, 370]
        report(centre[0] > 240 and centre[1] < 40 and ring[3] > least and beyond[3] == 0,
               f"{comp}: centre RGBA {centre} is the dot, outline alpha {ring[3]}, beyond it {beyond[3]}")
    report(im[180, 361, 0] > 3 * im[180, 361, 1], f"edge_colour keeps the dot's colour: outline RGB {im[180, 361, :3]}")

    print("Alpha Min:")
    # A pixel the dot has left follows the curve exactly: Max 0.9, Min 0.1.
    seen = [int(load(label, "alpha_min_fade", f)[180, 320, 3]) for f in range(56, 72)]
    expected = [seen[0] / 255]
    for _ in seen[1:]:
        expected.append(max(min(expected[-1] * 0.81 / 0.8 - 0.09 / 0.8, 1), 0))
    worst = max(abs(a - b * 255) for a, b in zip(seen, expected))
    report(worst <= 2 and seen[-1] == 0, f"fades along the curve to nothing: worst {worst:.1f}/255, {seen}")
    peak, count = difference(load(label, "alpha_min_fade", 95), load(label, "alpha_min_sub", 95))
    report(None, f"sub-steps against none: peak {peak}/255 across {count} pixels")
    choked = visible(load(label, "alpha_min", 95))
    loose = visible(load(label, "alpha_min_ref", 95))
    report(0 < choked < loose, f"chokes a blurred trail: {choked} visible pixels against {loose}")


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
    print("  wall clock: " + ", ".join(f"{k} {t[k]:.1f}s" for k in ("trail", "perf_long", "limit_600", "perf_long_cold")))


def summarize():
    if False in RESULTS:
        print(f"\n{RESULTS.count(False)} failed, {RESULTS.count(True)} passed")
        sys.exit(1)
    if RESULTS:
        print(f"\nAll {RESULTS.count(True)} checks passed")


def main():
    action = sys.argv[1]
    {"render": cmd_render, "check": cmd_check, "compare": cmd_compare, "perf": cmd_perf}[action](*sys.argv[2:])
    summarize()


if __name__ == "__main__":
    main()
