// Renders frames out of order within one After Effects session, editing the project part way
// through, so the effect's caches are in play. live_check.py compares the results with aerender's.
// Output folders tests/out/live/<step> must already exist; run with a project closed or empty.
(function () {
    var HERE = new File($.fileName).parent.fsName;
    var OUT = HERE + "/out";
    var FPS = 24;

    function comp(name) {
        for (var i = 1; i <= app.project.numItems; i++) {
            var it = app.project.item(i);
            if (it instanceof CompItem && it.name === name) return it;
        }
        throw new Error("No comp " + name);
    }

    function queue(c, frame, step, settings) {
        var rq = app.project.renderQueue;
        while (rq.numItems > 0) rq.item(1).remove();
        var item = rq.items.add(c);
        item.applyTemplate("Best Settings");
        if (settings) item.setSettings(settings);
        item.timeSpanStart = frame / FPS;
        item.timeSpanDuration = 1 / FPS;
        var om = item.outputModule(1);
        om.applyTemplate("TIFF Sequence with Alpha");
        om.file = new File(OUT + "/live/" + step + "/f_[#####].tif");
        rq.render();
    }

    function find(group, name) {
        for (var i = 1; i <= group.numProperties; i++) {
            var p = group.property(i);
            if (p.name === name) return p;
            if (p.propertyType !== PropertyType.PROPERTY) {
                var q = find(p, name);
                if (q) return q;
            }
        }
        return null;
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

    // Saved so that a fresh aerender can show what those last frames should be.
    app.project.save(new File(OUT + "/feedback_live_edited.aep"));
    app.endSuppressDialogs(false);
})();
