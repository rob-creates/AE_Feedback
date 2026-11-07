// Builds the test projects in tests/out/ (which must exist), one per bit depth. Run inside After Effects:
//   osascript -e 'tell application "Adobe After Effects 2026" to DoScriptFile "<path to this file>"'
#include "common.jsx"

(function () {
    var W = 640, H = 360;
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

    function newComp(name, frames, w, h) {
        return app.project.items.addComp(name, w || W, h || H, 1, frames / FPS, FPS);
    }

    // A solid masked to a dot travelling from `from` to `to` over the comp, in layer space.
    function movingDot(comp, name, colour, w, h, from, to, r) {
        var layer = comp.layers.addSolid(colour, name, w, h, 1, comp.duration);
        var mask = layer.Masks.addProperty("Mask");
        var shape = mask.property("maskShape");
        shape.setValueAtTime(0, dot(from[0], from[1], r));
        shape.setValueAtTime(comp.duration, dot(to[0], to[1], r));
        return layer;
    }

    function useMap(fx, layer) {
        find(fx, MAP_LAYER).setValue(layer.index);
    }

    function addEllipse(layer, size) {
        var contents = layer.property("Contents").addProperty("ADBE Vector Group").property("Contents");
        contents.addProperty("ADBE Vector Shape - Ellipse").property("Size").setValue([size, size]);
        contents.addProperty("ADBE Vector Graphic - Fill").property("Color").setValue(RED);
    }

    function feedback(layer, values) {
        var fx = layer.property("Effects").addProperty("RPAY Feedback");
        for (var k in values) {
            var p = find(fx, k);
            if (!p) throw new Error("No parameter " + k);
            p.setValue(values[k]);
        }
        return fx;
    }

    var RED = [1, 0.1, 0.1], BLUE = [0.1, 0.2, 1], GREY = [0.5, 0.5, 0.5];
    var ACROSS = [[80, 100], [560, 260]];

    function dotComp(name, frames, settings) {
        var comp = newComp(name, frames);
        var layer = movingDot(comp, "dot", RED, W, H, ACROSS[0], ACROSS[1], 30);
        return { comp: comp, layer: layer, fx: feedback(layer, settings) };
    }

    // A dot trimmed to begin at `inFrame`, with or without the effect, beside the same thing bare.
    function trimComps(name, inFrame, startFrame, stretch) {
        for (var bare = 0; bare < 2; bare++) {
            var comp = newComp(name + (bare ? "_plain" : ""), 96);
            var layer = movingDot(comp, "dot", RED, W, H, ACROSS[0], ACROSS[1], 30);
            layer.stretch = stretch;
            layer.startTime = startFrame / FPS;
            layer.inPoint = inFrame / FPS;
            if (!bare) feedback(layer, { "Alpha Max": 0.85 });
        }
    }

    function build() {
        dotComp("trail", 96, { "Alpha Max": 0.85 });
        trimComps("trim_48", 48, 0, 100);
        trimComps("trim_24", 24, 0, 100);
        trimComps("trim_start", 48, 24, 100);
        trimComps("trim_stretch", 48, 0, 200);
        dotComp("behind", 96, { "Alpha Max": 0.85, "Composite": 2 });
        dotComp("screen_sub", 96, { "Alpha Max": 0.9, "Composite": 4, "Sub-steps": 3 });
        dotComp("add_sub", 96, { "Alpha Max": 0.8, "Composite": 3, "Sub-steps": 1 });
        dotComp("grade_blur", 96, {
            "Alpha Max": 0.9, "Blur Radius": 4, "Hue Shift": 12, "Saturation": 90, "Gamma": 1.1, "Lightness": 1
        });
        dotComp("self_disp", 96, { "Alpha Max": 0.9, "Max Horizontal": 12, "Max Vertical": -8 });

        // Keyframes: `keyed` and `keyed_edit` agree up to frame 48 and differ after it.
        var keyed = dotComp("keyed", 96, {});
        var keyedEdit = dotComp("keyed_edit", 96, {});
        var alphaA = find(keyed.fx, "Alpha Max"), blurA = find(keyed.fx, "Blur Radius");
        var alphaB = find(keyedEdit.fx, "Alpha Max"), blurB = find(keyedEdit.fx, "Blur Radius");
        var t = 48 / FPS;
        alphaA.setValueAtTime(0, 0.7); alphaA.setValueAtTime(t, 0.9);
        alphaB.setValueAtTime(0, 0.7); alphaB.setValueAtTime(t, 0.9);
        blurA.setValueAtTime(0, 0); blurA.setValueAtTime(t, 3); blurA.setValueAtTime(t + 1 / FPS, 3);
        blurB.setValueAtTime(0, 0); blurB.setValueAtTime(t, 3); blurB.setValueAtTime(t + 1 / FPS, 20);

        // Two different layers with identical settings, sizes and motion, side by side.
        var two = newComp("two_layers", 72);
        var left = movingDot(two, "red", RED, 320, H, [60, 60], [260, 300], 25);
        var right = movingDot(two, "blue", BLUE, 320, H, [60, 60], [260, 300], 25);
        left.property("Position").setValue([160, H / 2]);
        right.property("Position").setValue([480, H / 2]);
        feedback(left, { "Alpha Max": 0.9 });
        feedback(right, { "Alpha Max": 0.9 });

        // The same layer's effect twice over.
        var dup = dotComp("duplicate_fx", 72, { "Alpha Max": 0.8 });
        feedback(dup.layer, { "Alpha Max": 0.8, "Hue Shift": 90 });

        // Displacement: each of these should match `neutral_ref`.
        dotComp("neutral_ref", 72, { "Alpha Max": 0.9 });
        function mapped(name, makeMap, channel) {
            var d = dotComp(name, 72, {
                "Alpha Max": 0.9, "Max Horizontal": 100, "Max Vertical": 100,
                "Horizontal Channel": channel, "Vertical Channel": channel
            });
            var map = makeMap(d.comp);
            map.enabled = false;
            useMap(d.fx, map);
        }
        function greySolid(comp) { return comp.layers.addSolid(GREY, "map", W, H, 1, comp.duration); }
        function redSolid(comp) { return comp.layers.addSolid(RED, "map", W, H, 1, comp.duration); }
        // A map with real transparency: a layer's own masks don't reach the effect, a precomp's do.
        function precompMap(opacity, size) {
            return function (comp) {
                var inner = newComp(comp.name + "_map", 72);
                var solid = inner.layers.addSolid(RED, "s", W, H, 1, inner.duration);
                solid.property("Opacity").setValue(opacity);
                if (size) solid.Masks.addProperty("Mask").property("maskShape").setValue(dot(5, 5, size));
                return comp.layers.add(inner);
            };
        }
        mapped("neutral_luma", greySolid, 5);
        mapped("neutral_red", greySolid, 1);
        mapped("neutral_alpha_transparent", precompMap(100, 2), 4);
        mapped("neutral_alpha_half", precompMap(50, 0), 4);

        // Map Layer's stage. At Effects & Masks, a mask leaves the map almost all transparent
        // and Fill turns its red to the neutral 50%, so neither should shift anything.
        function maskedSolid(comp) {
            var s = redSolid(comp);
            s.Masks.addProperty("Mask").property("maskShape").setValue(dot(5, 5, 2));
            return s;
        }
        function setStage(name, stage) {
            find(comp(name).layer("dot").property("Effects").property(1), MAP_LAYER).setLayerInputStage(stage);
        }
        mapped("map_solid_opaque", redSolid, 4);
        mapped("map_source_default", maskedSolid, 4); // left at Source, which ignores the mask
        mapped("map_solid_masked", maskedSolid, 4);
        setStage("map_solid_masked", -1);
        mapped("map_effects", function (comp) {
            var s = redSolid(comp);
            s.property("Effects").addProperty("ADBE Fill").property("Color").setValue(GREY);
            return s;
        }, 1);
        setStage("map_effects", -1);

        // Long trails, for PreRender cost and the 600-frame limit.
        var perf = newComp("perf_long", 300, 1920, 1080);
        var perfLayer = movingDot(perf, "dot", RED, 1920, 1080, [100, 100], [1800, 980], 60);
        feedback(perfLayer, { "Alpha Max": 0.98, "Sub-steps": 2 });
        dotComp("limit_600", 700, { "Alpha Max": 1 });
        dotComp("limit_800", 700, { "Alpha Max": 1, "Trail Rebuild Limit (Frames)": 800 });
        dotComp("limit_100", 300, { "Alpha Max": 1, "Trail Rebuild Limit (Frames)": 100 });

        // Alpha Min: on its own, against the same blur without it, and split into sub-steps.
        dotComp("alpha_min", 96, { "Alpha Max": 1, "Alpha Min": 0.2, "Blur Radius": 3 });
        dotComp("alpha_min_ref", 96, { "Alpha Max": 1, "Blur Radius": 3 });
        dotComp("alpha_min_fade", 96, { "Alpha Max": 0.9, "Alpha Min": 0.1 });
        dotComp("alpha_min_sub", 96, { "Alpha Max": 0.9, "Alpha Min": 0.1, "Sub-steps": 3 });

        // A layer with nothing in it, alone and with a map; and one that empties part way through,
        // which should leave its trail to fade. Its mask moves off the layer at frame 30.
        function emptyLayer(name, settings) {
            var c = newComp(name, 48);
            var shape = c.layers.addShape();
            return { comp: c, fx: feedback(shape, settings) };
        }
        emptyLayer("empty_layer", { "Alpha Max": 0.9 });
        var emptyMapped = emptyLayer("empty_mapped", { "Alpha Max": 0.9, "Max Horizontal": 20 });
        var emptyMap = emptyMapped.comp.layers.addSolid(RED, "map", W, H, 1, emptyMapped.comp.duration);
        emptyMap.enabled = false;
        useMap(emptyMapped.fx, emptyMap);
        var emptying = dotComp("empties_midway", 96, { "Alpha Max": 0.9 });
        var emptyingShape = emptying.layer.Masks.property(1).property("maskShape");
        emptyingShape.removeKey(2);
        emptyingShape.setValueAtTime(29 / FPS, dot(200, 180, 30));
        emptyingShape.setValueAtTime(30 / FPS, dot(-5000, -5000, 30));
        emptyingShape.setInterpolationTypeAtKey(2, KeyframeInterpolationType.HOLD);

        // Map Layer pointed at the effect's own layer reads that layer, rather than the trail.
        var own = dotComp("layer_own", 72, { "Alpha Max": 0.9, "Max Horizontal": 100, "Horizontal Channel": 4 });
        useMap(own.fx, own.layer);
        dotComp("layer_none", 72, { "Alpha Max": 0.9, "Max Horizontal": 100, "Horizontal Channel": 4 });
        // Its own layer with Effects & Masks is the layer as the effect leaves it: the trail, as None.
        var ownAll = dotComp("layer_own_effects", 72, { "Alpha Max": 0.9, "Max Horizontal": 100, "Horizontal Channel": 4 });
        useMap(ownAll.fx, ownAll.layer);
        find(ownAll.fx, MAP_LAYER).setLayerInputStage(-1);

        // Shape layers rasterise continuously, unlike the solids above: a moving ellipse, alone,
        // with sub-steps, and with a map. Each is also rendered at half resolution in live_session.
        function shapeComp(name, settings, withMap) {
            var c = newComp(name, 96);
            var layer = c.layers.addShape();
            addEllipse(layer, 60);
            var position = layer.property("Position");
            position.setValueAtTime(0, ACROSS[0]);
            position.setValueAtTime(c.duration, ACROSS[1]);
            var fx = feedback(layer, settings);
            if (withMap) {
                var map = c.layers.addSolid([0.8, 0.2, 0.5], "map", W, H, 1, c.duration);
                map.enabled = false;
                useMap(fx, map);
            }
        }
        shapeComp("shape_trail", { "Alpha Max": 0.9 });
        shapeComp("shape_sub", { "Alpha Max": 0.9, "Sub-steps": 2 });
        // A shape layer as the map instead, on a solid.
        var shapeMap = dotComp("shape_as_map", 96, { "Alpha Max": 0.9, "Max Horizontal": 30, "Horizontal Channel": 4 });
        var mapShape = shapeMap.comp.layers.addShape();
        addEllipse(mapShape, 400);
        mapShape.property("Position").setValue([W / 2, H / 2]);
        mapShape.enabled = false;
        useMap(shapeMap.fx, mapShape);
        shapeComp("shape_mapped", { "Alpha Max": 0.9, "Sub-steps": 2, "Max Horizontal": 6, "Max Vertical": 6, "Horizontal Channel": 1, "Vertical Channel": 2 }, true);
    }

    // Without a GPU, the layer passes through with no trail at all.
    function buildSoftware() {
        app.newProject();
        app.project.bitsPerChannel = 16;
        app.project.gpuAccelType = GpuAccelType.SOFTWARE;
        dotComp("software_trail", 24, { "Alpha Max": 0.9 });
        var plain = newComp("software_plain", 24);
        movingDot(plain, "dot", RED, W, H, ACROSS[0], ACROSS[1], 30);
        app.project.save(new File(OUT + "/feedback_software.aep"));
    }

    app.beginSuppressDialogs();
    buildSoftware();
    var results = [];
    var depths = [16, 8, 32];
    for (var i = 0; i < depths.length; i++) {
        app.newProject();
        app.project.bitsPerChannel = depths[i];
        app.project.gpuAccelType = GpuAccelType.METAL;
        build();
        app.project.save(new File(OUT + "/feedback_" + depths[i] + ".aep"));
        results.push(depths[i] + ":" + app.project.numItems + " items gpu=" + app.project.gpuAccelType);
    }
    app.endSuppressDialogs(false);
    return results.join(" | ");
})();
