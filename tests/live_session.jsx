// Renders frames out of order within one After Effects session, editing the project part way
// through, so the effect's caches are in play. live_check.py compares the results with aerender's.
// Output folders tests/out/live/<step> must already exist; run with a project closed or empty.
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

    // Moving an in-point changes every frame after it, though nothing in the layer's history did.
    var trimmed = comp("trim_48");
    queue(trimmed, 60, "trim_before");
    trimmed.layer("dot").inPoint = 24 / FPS;
    queue(trimmed, 60, "trim_after");

    // Saved so that a fresh aerender can show what those last frames should be.
    app.project.save(new File(OUT + "/feedback_live_edited.aep"));
    app.endSuppressDialogs(false);
})();
