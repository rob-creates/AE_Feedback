#!/usr/bin/env python3
"""Compares live_session.jsx's frames with aerender's.  live_check.py LABEL"""
import sys

import numpy as np

from run_tests import difference, load, render, report, summarize, visible, OUT


def live(step, frame):
    return load("live", step, frame)


def check(name, a, b):
    peak, count = difference(a, b)
    report(peak <= 1, f"{name}: peak {peak}/255 across {count} pixels")


def shrink(full, factor):
    """A full-resolution frame at reduced resolution. A partial pixel at the edge is left out."""
    h, w = full.shape[0] // factor, full.shape[1] // factor
    return full[: h * factor, : w * factor].reshape(h, factor, w, factor, 4).mean(axis=(1, 3))


def main(label):
    ref = lambda comp, frame: load(label, comp, frame)

    print("Out-of-order requests in one session:")
    for i, frame in enumerate([60, 20, 61, 95, 60, 59]):
        check(f"request {i + 1}, frame {frame}", live(f"order_{i}", frame), ref("trail", frame))

    print("Nested precomp, and returning from half resolution:")
    check("nested@60", live("nested", 60), ref("trail", 60))
    check("full again@60", live("full_again", 60), ref("trail", 60))

    half = live("half", 60)
    shrunk = shrink(ref("trail", 60), 2)
    report(None, f"half-resolution frame {half.shape[1]}x{half.shape[0]}: mean difference from a shrunk full frame "
                 f"{np.abs(half - shrunk).mean():.2f}/255, peak {np.abs(half - shrunk).max():.0f}")

    print("Reduced resolution, against a shrunk full-resolution frame:")
    for step, comp, factor in [("half_" + c, c, 2) for c in ("shape_trail", "shape_sub", "shape_mapped", "shape_as_map",
                                                             "screen_sub", "map_solid_opaque", "alpha_min")] + [
            ("third_shape_trail", "shape_trail", 3), ("half_run_shape_sub", "shape_sub", 2)]:
        shrunk = shrink(ref(comp, 60), factor)
        small = live(step, 60)[: shrunk.shape[0], : shrunk.shape[1]]
        seen, expected = visible(small), visible(shrunk)
        report(expected > 0 and abs(seen - expected) < 0.15 * expected,
               f"{step}: {seen} visible pixels, expected about {expected}, "
               f"mean difference {np.abs(small - shrunk).mean():.2f}/255")

    print("Transform and comp-space trails at half resolution, against a shrunk full-resolution frame:")
    for comp, frame in (("transform_rotate", 12), ("transform_scale", 11), ("transform_position", 11),
                        ("comp_space_2d", 20), ("layer_3d", 20)):
        shrunk = shrink(ref(comp, frame), 2)
        small = live("half_" + comp, frame)[: shrunk.shape[0], : shrunk.shape[1]]
        seen, expected = small[..., 3] > 32, shrunk[..., 3] > 32
        overlap = (seen & expected).sum() / max((seen | expected).sum(), 1)
        report(overlap > 0.7, f"{comp}: the dots overlap by {overlap:.0%}")

    print("Editing a keyframe during the session:")
    check("before the edit@70", live("keyed_before", 70), ref("keyed", 70))
    check("after the edit@70", live("keyed_after", 70), ref("keyed_edit", 70))
    check("earlier frame after the edit@30", live("keyed_early", 30), ref("keyed", 30))

    print("Editing the layer's mask during the session:")
    render("feedback_live_edited.aep", "trail", OUT / label / "edited_mask", 0, 60)
    edited = ref("edited_mask", 60)
    check("before the edit@60", live("mask_before", 60), ref("trail", 60))
    check("after the edit@60", live("mask_after", 60), edited)
    report(None, f"the edit itself changed the frame by a peak of {difference(live('mask_before', 60), edited)[0]}/255")

    print("Changing Map Layer's stage during the session:")
    check("Effects & Masks@60 shifts nothing", live("stage_all", 60), ref("neutral_ref", 60))
    check("Source@60 ignores the mask", live("stage_source", 60), ref("map_solid_opaque", 60))
    check("a duplicate keeps Effects & Masks@60", live("stage_duplicate", 60), ref("neutral_ref", 60))

    print("Changing Map Layer during the session:")
    check("None before picking@60", live("pick_none_before", 60), ref("layer_none", 60))
    check("its own layer@60", live("pick_own", 60), ref("layer_own", 60))
    check("None again@60", live("pick_none_after", 60), ref("layer_none", 60))
    check("its own layer at Effects & Masks@60", live("own_effects", 60), ref("layer_none", 60))
    check("then at Source@60", live("own_source", 60), ref("layer_own", 60))
    check("then at Effects & Masks again@60", live("own_effects_again", 60), ref("layer_none", 60))

    print("Moving a layer's in-point during the session:")
    check("before the move@60", live("trim_before", 60), ref("trim_48", 60))
    check("after the move@60", live("trim_after", 60), ref("trim_24", 60))

    print("Moving a comp-space trail's layer during the session:")
    render("feedback_live_edited.aep", "comp_space_2d", OUT / label / "edited_motion", 0, 20)
    check("before the move@20", live("comp_motion_before", 20), ref("comp_space_2d", 20))
    check("after the move@20", live("comp_motion_after", 20), ref("edited_motion", 20))
    report(None, f"the move itself changed the frame by a peak of {difference(live('comp_motion_before', 20), ref('edited_motion', 20))[0]}/255")

    print("Changing what's beneath an adjustment layer, or inside a precomp, during the session:")
    for comp, step in (("adjust_below", "below"), ("precomp_outer", "inner")):
        render("feedback_live_edited.aep", comp, OUT / label / f"edited_{step}", 0, 40)
        check(f"{comp} before the change@40", live(f"{step}_before", 40), ref(comp, 40))
        check(f"{comp} after the change@40", live(f"{step}_after", 40), ref(f"edited_{step}", 40))

    print("A project saved by an earlier version of the effect:")
    render("old_project.aep", "trail", OUT / label / "old_project", 0, 60)
    check("opens and renders the same@60", ref("old_project", 60), ref("trail", 60))

    summarize()


if __name__ == "__main__":
    main(sys.argv[1])
