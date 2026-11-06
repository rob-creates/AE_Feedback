#!/usr/bin/env python3
"""Compares live_session.jsx's frames with aerender's.  live_check.py LABEL"""
import sys

import numpy as np

from run_tests import OUT, read_tiff, render


def live(step, frame):
    return read_tiff(OUT / "live" / step / f"f_{frame:05d}.tif")


def main(label):
    failures = 0

    def check(name, a, b, tolerance=1):
        nonlocal failures
        d = np.abs(a - b)
        ok = d.max() <= tolerance
        failures += not ok
        print(("PASS  " if ok else "FAIL  ") + f"{name}: peak {d.max()}/255 across {(d.max(axis=-1) > 0).sum()} pixels")

    ref = lambda comp, frame: read_tiff(OUT / label / comp / f"f_{frame:05d}.tif")

    print("Out-of-order requests in one session:")
    for i, frame in enumerate([60, 20, 61, 95, 60, 59]):
        check(f"request {i + 1}, frame {frame}", live(f"order_{i}", frame), ref("trail", frame))

    print("Nested precomp, and returning from half resolution:")
    check("nested@60", live("nested", 60), ref("trail", 60))
    check("full again@60", live("full_again", 60), ref("trail", 60))

    half = live("half", 60)
    full = ref("trail", 60)
    h, w = half.shape[:2]
    shrunk = full[: h * 2, : w * 2].reshape(h, 2, w, 2, 4).mean(axis=(1, 3))
    print(f"INFO  half-resolution frame {w}x{h}: mean difference from a shrunk full frame "
          f"{np.abs(half - shrunk).mean():.2f}/255, peak {np.abs(half - shrunk).max():.0f}")

    print("Editing a keyframe during the session:")
    check("before the edit@70", live("keyed_before", 70), ref("keyed", 70))
    check("after the edit@70", live("keyed_after", 70), ref("keyed_edit", 70))
    check("earlier frame after the edit@30", live("keyed_early", 30), ref("keyed", 30))

    print("Editing the layer's mask during the session:")
    folder = OUT / label / "edited_mask"
    render("feedback_live_edited.aep", "trail", folder, 0, 60)
    edited = read_tiff(folder / "f_00060.tif")
    check("before the edit@60", live("mask_before", 60), ref("trail", 60))
    check("after the edit@60", live("mask_after", 60), edited)
    changed = np.abs(live("mask_before", 60) - edited).max()
    print(f"INFO  the edit itself changed the frame by a peak of {changed}/255")

    print(f"\n{failures} failed" if failures else "\nAll live checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(sys.argv[1])
