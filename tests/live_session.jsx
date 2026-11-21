// Renders frames out of order within one After Effects session, editing the project part way
// through, so the effect's caches are in play. live_check.py compares the results with aerender's.
// Writes to tests/out/live/<step>; run with a project closed or empty.
#include "common.jsx"

(function () {
    var HERE = new File($.fileName).parent.fsName;
    var OUT = HERE + "/out";

    function queue(c, frame, step, settings, frames) {
        var rq = app.project.renderQueue;
        while (rq.numItems > 0) rq.item(1).remove();
        var item = rq.items.add(c);
        item.applyTemplate("Best Settings");
        if (settings) item.setSettings(settings);
        item.timeSpanStart = (frame - (frames || 1) + 1) / FPS;
        item.timeSpanDuration = (frames || 1) / FPS;
        var om = item.outputModule(1);
        om.applyTemplate("TIFF Sequence with Alpha");
        var folder = new Folder(OUT + "/live/" + step);
        if (!folder.exists) folder.create();
        om.file = new File(OUT + "/live/" + step + "/f_[#####].tif");
        rq.render();
    }

    app.beginSuppressDialogs();
    app.open(new File(OUT + "/feedback_16.aep"));

    // Out of order, including revisiting a frame and stepping back.
    var trail = comp("trail");
    var order = [60, 20, 61, 95, 60, 59];
    for (var i = 0; i < order.length; i++) queue(trail, order[i], "order_" + i);

    // A nested precomp should match its source.
    var nest = app.project.items.addComp("nested", 640, 360, 1, 96 / FPS, FPS);
    nest.layers.add(comp("trail"));
    queue(nest, 60, "nested");

    // A half-resolution request, then full again.
    queue(trail, 60, "half", { "Resolution": [2, 2] });
    queue(trail, 60, "full_again");

    // Half resolution with sub-steps, maps and shape layers.
    var halves = ["shape_trail", "shape_sub", "shape_mapped", "shape_as_map", "screen_sub", "map_solid_opaque", "alpha_min"];
    for (var h = 0; h < halves.length; h++) queue(comp(halves[h]), 60, "half_" + halves[h], { "Resolution": [2, 2] });
    queue(comp("shape_trail"), 60, "third_shape_trail", { "Resolution": [3, 3] });
    // Transform's centre, offset and moves at half resolution.
    queue(comp("transform_rotate"), 12, "half_transform_rotate", { "Resolution": [2, 2] });
    queue(comp("transform_position"), 11, "half_transform_position", { "Resolution": [2, 2] });
    // The trail in comp space at half resolution.
    queue(comp("comp_space_2d"), 20, "half_comp_space_2d", { "Resolution": [2, 2] });
    queue(comp("layer_3d"), 20, "half_layer_3d", { "Resolution": [2, 2] });
    queue(comp("transform_scale"), 11, "half_transform_scale", { "Resolution": [2, 2] });
    // From the start and playing through, so that frames resume from one another.
    queue(comp("shape_sub"), 60, "half_run_shape_sub", { "Resolution": [2, 2] }, 61);

    // Editing a keyframe must change later frames only, right away.
    var keyed = comp("keyed");
    queue(keyed, 70, "keyed_before");
    var fx = keyed.layer(1).property("Effects").property(1);
    find(fx, "Blur Radius").setValueAtTime(49 / FPS, 20);
    queue(keyed, 70, "keyed_after");
    queue(keyed, 30, "keyed_early");

    // Moving the layer's own mask changes its bounds, which PreRender remembers per frame.
    queue(trail, 60, "mask_before");
    var shape = trail.layer(1).Masks.property(1).property("maskShape");
    var moved = shape.keyValue(2);
    var verts = moved.vertices;
    for (var v = 0; v < verts.length; v++) verts[v] = [verts[v][0] - 400, verts[v][1] - 150];
    moved.vertices = verts;
    shape.setValueAtKey(2, moved);
    queue(trail, 60, "mask_after");

    // Changing Map Layer's stage must take effect right away.
    var masked = comp("map_solid_masked");
    queue(masked, 60, "stage_all");

    // A duplicate keeps the setting of the original.
    queue(masked.duplicate(), 60, "stage_duplicate");

    find(masked.layer("dot").property("Effects").property(1), MAP_LAYER).setLayerInputStage(0);
    queue(masked, 60, "stage_source");

    // Picking and clearing Map Layer must take effect right away, using bounds it hadn't read before.
    var pick = comp("layer_none");
    var pickFx = pick.layer("dot").property("Effects").property(1);
    queue(pick, 60, "pick_none_before");
    find(pickFx, MAP_LAYER).setValue(pick.layer("dot").index);
    queue(pick, 60, "pick_own");
    find(pickFx, MAP_LAYER).setValue(0);
    queue(pick, 60, "pick_none_after");

    // Its own layer reads the trail at Effects & Masks and the layer at Source, so switching the
    // stage alone changes whether there's a map.
    var own = comp("layer_own_effects");
    var ownLayer = find(own.layer("dot").property("Effects").property(1), MAP_LAYER);
    queue(own, 60, "own_effects");
    ownLayer.setLayerInputStage(0);
    queue(own, 60, "own_source");
    ownLayer.setLayerInputStage(-1);
    queue(own, 60, "own_effects_again");

    // Moving an in-point changes every frame after it, though nothing in the layer's history did.
    var trimmed = comp("trim_48");
    queue(trimmed, 60, "trim_before");
    trimmed.layer("dot").inPoint = 24 / FPS;
    queue(trimmed, 60, "trim_after");

    // Moving the layer itself changes a comp-space trail, though nothing in the effect's own history did.
    var travelling = comp("comp_space_2d");
    queue(travelling, 20, "comp_motion_before");
    travelling.layer("dot").property("Position").setValueAtKey(2, [560, 60]);
    queue(travelling, 20, "comp_motion_after");

    // Changing what's beneath an adjustment layer, or inside a precomp, changes every frame of the trail.
    var below = comp("adjust_below");
    queue(below, 40, "below_before", null, 41);
    below.layer("dot").source.mainSource.color = [0.1, 0.2, 1];
    queue(below, 40, "below_after");
    var outer = comp("precomp_outer");
    queue(outer, 40, "inner_before", null, 41);
    comp("precomp_inner").layer("dot").source.mainSource.color = [0.1, 0.2, 1];
    queue(outer, 40, "inner_after");

    // Moving a layer between frames, where only a sub-step sees it, changes the trail, though nothing at
    // frame times moved.
    var betweenComp = comp("subframe_motion");
    queue(betweenComp, 20, "between_before", null, 21);
    betweenComp.layer("dot").property("Position").setValueAtKey(3, [200, 60]);
    queue(betweenComp, 20, "between_after");
    // Changing a keyframe beneath an adjustment layer, at frames 10 to 15 only, changes frame 40's trail.
    var keyBelow = comp("adjust_key_below");
    queue(keyBelow, 40, "key_below_before", null, 41);
    keyBelow.layer("flash").property("Opacity").setValueAtKey(3, 10);
    queue(keyBelow, 40, "key_below_after");

    // AE's history of the effect's controls doesn't change with an expression, so frames made before
    // one is added, or before what it reads changes, mustn't be shown or resumed from after.
    var stamped = comp("expression_stamp");
    var opacity = find(stamped.layer("dot").property("Effects").property(1), "Opacity");
    queue(stamped, 12, "expr_before", null, 13);
    opacity.expression = "Math.round(time * " + FPS + ") % 4 == 0 ? 1 : 0";
    queue(stamped, 12, "expr_added", null, 13);
    var control = stamped.layers.addNull();
    control.name = "control";
    var slider = control.property("Effects").addProperty("ADBE Slider Control").property(1);
    slider.setValue(3);
    opacity.expression = "Math.round(time * " + FPS + ") % thisComp.layer(\"control\").effect(1)(1) == 0 ? 1 : 0";
    queue(stamped, 12, "expr_read_3", null, 13);
    slider.setValue(5);
    queue(stamped, 13, "expr_read_5", null, 2);

    // Saved so that a fresh aerender can show what those last frames should be.
    app.project.save(new File(OUT + "/feedback_live_edited.aep"));
    app.endSuppressDialogs(false);
})();
