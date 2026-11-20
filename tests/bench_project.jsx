// Builds tests/out/bench.aep, larger comps for measuring speed. Run inside After Effects, like
// build_project.jsx, after make_footage.sh. Each comp has copies whose Alpha Max differs too little to see, so that AE's
// disk cache can't serve one from another's render.
#include "common.jsx"

(function () {
    var FRAMES = 300, COPIES = 3;
    var OUT = new File($.fileName).parent.fsName + "/out";

    function dot(cx, cy, r) {
        var verts = [];
        for (var i = 0; i < 24; i++) {
            var a = i / 24 * Math.PI * 2;
            verts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
        }
        var s = new Shape();
        s.vertices = verts;
        s.closed = true;
        return s;
    }

    function noiseLayer(c, name) {
        var layer = c.layers.addSolid([0.5, 0.5, 0.5], name, c.width, c.height, 1, c.duration);
        var noise = layer.property("Effects").addProperty("ADBE Fractal Noise");
        noise.property("Evolution").setValueAtTime(0, 0);
        noise.property("Evolution").setValueAtTime(c.duration, 720);
        return layer;
    }

    function movingDot(c) {
        var layer = c.layers.addSolid([1, 0.1, 0.1], "dot", c.width, c.height, 1, c.duration);
        var shape = layer.Masks.addProperty("Mask").property("maskShape");
        shape.setValueAtTime(0, dot(c.width * 0.1, c.height * 0.2, c.height * 0.08));
        shape.setValueAtTime(c.duration, dot(c.width * 0.9, c.height * 0.8, c.height * 0.08));
        return layer;
    }

    function feedback(layer, values, nudge) {
        var fx = layer.property("Effects").addProperty("RPAY Feedback");
        for (var k in values) find(fx, k).setValue(values[k]);
        find(fx, "Alpha Max").setValue((values["Alpha Max"] || 0.85) + nudge);
        return fx;
    }

    // A shape wandering around the comp under a noise expression, as real footage often does.
    function wanderingShape(c) {
        var layer = c.layers.addShape();
        var contents = layer.property("Contents").addProperty("ADBE Vector Group").property("Contents");
        contents.addProperty("ADBE Vector Shape - Ellipse").property("Size").setValue([200, 200]);
        contents.addProperty("ADBE Vector Graphic - Fill").property("Color").setValue([1, 0.1, 0.1]);
        // Inside the layer: the effect works in layer space, so moving the layer itself leaves no trail.
        layer.property("Contents").property(1).property("ADBE Vector Transform Group").property("ADBE Vector Position").expression =
            "[noise([time * 0.3, 1]), noise([time * 0.3, 2])] * [thisComp.width / 2, thisComp.height / 2]";
        return layer;
    }

    // Each clip is imported once, and shared by every comp that uses it.
    var clips = {};
    function clip(name) {
        if (!clips[name]) clips[name] = app.project.importFile(new ImportOptions(new File(OUT + "/clips/" + name)));
        return clips[name];
    }

    var BENCHES = {
        simple: function (c, n) { feedback(movingDot(c), { "Alpha Max": 0.99 }, n); },
        blur: function (c, n) { feedback(movingDot(c), { "Alpha Max": 0.98, "Blur Radius": 30 }, n); },
        grade: function (c, n) { feedback(movingDot(c), { "Alpha Max": 0.98, "Hue Shift": 3, "Gamma": 1.02 }, n); },
        map: function (c, n) {
            var map = noiseLayer(c, "map");
            map.enabled = false;
            var fx = feedback(movingDot(c), { "Alpha Max": 0.98, "Max Horizontal": 15, "Max Vertical": 15 }, n);
            find(fx, MAP_LAYER).setValue(c.layer("map").index);
        },
        trailmap: function (c, n) {
            var layer = movingDot(c);
            var fx = feedback(layer, { "Alpha Max": 0.98, "Max Horizontal": 6, "Max Vertical": 6 }, n);
            find(fx, MAP_LAYER).setValue(layer.index);
            find(fx, MAP_LAYER).setLayerInputStage(-1);
        },
        sub: function (c, n) { feedback(movingDot(c), { "Alpha Max": 0.98, "Sub-steps": 7 }, n); },
        add: function (c, n) { feedback(movingDot(c), { "Alpha Max": 0.97, "Composite": 3, "Sub-steps": 3 }, n); },
        late: function (c, n) {
            var layer = movingDot(c);
            layer.startTime = 0;
            layer.inPoint = c.duration * 0.8;
            feedback(layer, { "Alpha Max": 0.98 }, n);
        },
        full: function (c, n) { feedback(noiseLayer(c, "noise"), { "Alpha Max": 0.9, "Max Horizontal": 4 }, n); },
        wander: function (c, n) { feedback(wanderingShape(c), { "Alpha Max": 0.97 }, n); },
        expr: function (c, n) {
            var fx = feedback(wanderingShape(c), { "Alpha Max": 0.97 }, n);
            var names = ["Blur Radius", "Hue Shift", "Saturation", "Lightness", "Gamma", "Max Horizontal", "Max Vertical"];
            var bases = [1, 2, 100, 0, 1, 2, 2];
            for (var e = 0; e < names.length; e++)
                find(fx, names[e]).expression = bases[e] + " + noise(time) * 0.2";
        },
        small: function (c, n) {
            var layer = c.layers.addSolid([1, 0.1, 0.1], "dot", 200, 200, 1, c.duration);
            layer.Masks.addProperty("Mask").property("maskShape").setValue(dot(100, 100, 80));
            layer.property("Position").setValueAtTime(0, [c.width * 0.1, c.height * 0.2]);
            layer.property("Position").setValueAtTime(c.duration, [c.width * 0.9, c.height * 0.8]);
            feedback(layer, { "Alpha Max": 0.97 }, n);
        }
    };
    // A small shape layer moved by its own Position: its trail is kept in comp space.
    function movingShape(c) {
        var layer = c.layers.addShape();
        var contents = layer.property("Contents").addProperty("ADBE Vector Group").property("Contents");
        contents.addProperty("ADBE Vector Shape - Ellipse").property("Size").setValue([80, 80]);
        contents.addProperty("ADBE Vector Graphic - Fill").property("Color").setValue([1, 0.1, 0.1]);
        layer.property("Position").setValueAtTime(0, [c.width * 0.1, c.height * 0.2]);
        layer.property("Position").setValueAtTime(c.duration, [c.width * 0.9, c.height * 0.8]);
        return layer;
    }
    BENCHES.shape = function (c, n) { feedback(movingShape(c), { "Alpha Max": 0.9 }, n); };
    BENCHES.shape_sub = function (c, n) { feedback(movingShape(c), { "Alpha Max": 0.9, "Sub-steps": 3 }, n); };
    // Like a user's comp: a 128-pixel ellipse moved by Position, overlaid on its trail and displaced by
    // a precomp of two Turbulent Noise layers.
    function noisePrecomp(c, n) {
        var pre = app.project.items.addComp("noise_pre_" + c.name, c.width, c.height, 1, c.duration, c.frameRate);
        for (var i = 0; i < 2; i++) {
            var l = pre.layers.addSolid([0.5, 0.5, 0.5], "turb" + i, c.width, c.height, 1, c.duration);
            var t = l.property("Effects").addProperty("Turbulent Noise");
            t.property("Evolution").setValueAtTime(0, i * 90 + n);
            t.property("Evolution").setValueAtTime(c.duration, 720 + i * 90);
            if (i) { l.blendingMode = BlendingMode.ADD; l.property("Opacity").setValue(50); }
        }
        return pre;
    }
    function userComp(c, n, settings, mapKind) {
        var map = null;
        if (mapKind === "noise") map = c.layers.add(noisePrecomp(c, n));
        if (mapKind === "footage") map = c.layers.add(clip("map.mp4"));
        if (mapKind === "solid") map = c.layers.addSolid([0.5, 0.5, 0.5], "flat", c.width, c.height, 1, c.duration);
        if (map) map.enabled = false;
        var layer = c.layers.addShape();
        var contents = layer.property("Contents").addProperty("ADBE Vector Group").property("Contents");
        contents.addProperty("ADBE Vector Shape - Ellipse").property("Size").setValue([128, 128]);
        contents.addProperty("ADBE Vector Graphic - Fill").property("Color").setValue([1, 0.1, 0.1]);
        layer.property("Position").setValueAtTime(0, [c.width * 0.1, c.height * 0.2]);
        layer.property("Position").setValueAtTime(c.duration, [c.width * 0.9, c.height * 0.8]);
        var values = { "Alpha Max": 0.9, "Composite": 8 };
        if (map) { values["Max Horizontal"] = 32; values["Max Vertical"] = 32; values["Horizontal Channel"] = 1; values["Vertical Channel"] = 2; }
        for (var k in settings) values[k] = settings[k];
        var fx = feedback(layer, values, n);
        if (map) find(fx, MAP_LAYER).setValue(map.index);
    }
    BENCHES.user = function (c, n) { userComp(c, n, { "Sub-steps": 1 }, "noise"); };
    BENCHES.user_nosub = function (c, n) { userComp(c, n, { "Sub-steps": 0 }, "noise"); };
    BENCHES.user_flatmap = function (c, n) { userComp(c, n, { "Sub-steps": 1 }, "solid"); };
    BENCHES.user_nomap = function (c, n) { userComp(c, n, { "Sub-steps": 1 }, null); };
    BENCHES.user_static = function (c, n) { userComp(c, n, { "Sub-steps": 1, "Static Maps": true }, "noise"); };
    // Footage from make_footage.sh: a keyed clip with a slow fade, so a rebuild reaches back about 300
    // frames, as it plays and slowed by stretching; and a clip as the map.
    function footageIn(c, n, stretch) {
        var layer = c.layers.add(clip("input.mov"));
        layer.stretch = stretch;
        layer.inPoint = 0;
        layer.outPoint = c.duration;
        feedback(layer, { "Alpha Max": 0.98, "Composite": 4 }, n);
    }
    BENCHES.footage_in = function (c, n) { footageIn(c, n, 100); };
    BENCHES.footage_stretch = function (c, n) { footageIn(c, n, 400); };
    BENCHES.footage_map = function (c, n) { userComp(c, n, { "Sub-steps": 1 }, "footage"); };
    BENCHES.footage_map_static = function (c, n) { userComp(c, n, { "Sub-steps": 1, "Static Maps": true }, "footage"); };
    var SIZES = { small: [3840, 2160] };

    app.beginSuppressDialogs();
    app.newProject();
    app.project.bitsPerChannel = 16;
    app.project.gpuAccelType = GpuAccelType.METAL;
    for (var name in BENCHES) {
        for (var i = 0; i < COPIES; i++) {
            var size = SIZES[name] || [1920, 1080];
            var c = app.project.items.addComp(name + "_" + i, size[0], size[1], 1, FRAMES / FPS, FPS);
            BENCHES[name](c, i * 0.0001);
        }
    }
    app.project.save(new File(OUT + "/bench.aep"));
    app.endSuppressDialogs(false);
    return app.project.numItems;
})();
