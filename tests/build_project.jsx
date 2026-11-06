// Builds the test projects in tests/out/ (which must exist), one per bit depth. Run inside After Effects:
//   osascript -e 'tell application "Adobe After Effects 2026" to DoScriptFile "<path to this file>"'
(function () {
    var W = 640, H = 360, FPS = 24;
    var OUT = new File($.fileName).parent.fsName + "/out";

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

    function dotComp(name, frames, settings, colour) {
        var comp = newComp(name, frames);
        var layer = movingDot(comp, "dot", colour || RED, W, H, ACROSS[0], ACROSS[1], 30);
        return { comp: comp, layer: layer, fx: feedback(layer, settings) };
    }

    function build() {
        dotComp("trail", 96, { "Alpha": 0.85 });
        dotComp("behind", 96, { "Alpha": 0.85, "Composite": 2 });
        dotComp("screen_sub", 96, { "Alpha": 0.9, "Composite": 4, "Sub-steps": 3 });
        dotComp("add_sub", 96, { "Alpha": 0.8, "Composite": 3, "Sub-steps": 1 });
        dotComp("grade_blur", 96, {
            "Alpha": 0.9, "Blur Radius": 4, "Hue Shift": 12, "Saturation": 90, "Gamma": 1.1, "Lightness": 1
        });
        dotComp("self_disp", 96, { "Alpha": 0.9, "Max Horizontal": 12, "Max Vertical": -8 });

        // Keyframes: `keyed` and `keyed_edit` agree up to frame 48 and differ after it.
        var keyed = dotComp("keyed", 96, {});
        var keyedEdit = dotComp("keyed_edit", 96, {});
        var alphaA = find(keyed.fx, "Alpha"), blurA = find(keyed.fx, "Blur Radius");
        var alphaB = find(keyedEdit.fx, "Alpha"), blurB = find(keyedEdit.fx, "Blur Radius");
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
        feedback(left, { "Alpha": 0.9 });
        feedback(right, { "Alpha": 0.9 });

        // The same layer's effect twice over.
        var dup = dotComp("duplicate_fx", 72, { "Alpha": 0.8 });
        feedback(dup.layer, { "Alpha": 0.8, "Hue Shift": 90 });

        // Displacement: each of these should match `neutral_ref`.
        dotComp("neutral_ref", 72, { "Alpha": 0.9 });
        function mapped(name, makeMap, channel) {
            var d = dotComp(name, 72, {
                "Alpha": 0.9, "Source": 2, "Max Horizontal": 100, "Max Vertical": 100,
                "Horizontal Channel": channel, "Vertical Channel": channel
            });
            var map = makeMap(d.comp);
            map.enabled = false;
            find(d.fx, "Layer").setValue(map.index);
        }
        function greySolid(comp) { return comp.layers.addSolid(GREY, "map", W, H, 1, comp.duration); }
        function greySolidRed(comp) { return comp.layers.addSolid(RED, "map", W, H, 1, comp.duration); }
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

        // Does the Layer picker see a layer's masks? If it doesn't, these two come out the same.
        mapped("map_solid_opaque", greySolidRed, 4);
        mapped("map_solid_masked", function (comp) {
            var s = greySolidRed(comp);
            s.Masks.addProperty("Mask").property("maskShape").setValue(dot(5, 5, 2));
            return s;
        }, 4);

        // Long trails, for PreRender cost and the 600-frame limit.
        var perf = newComp("perf_long", 300, 1920, 1080);
        var perfLayer = movingDot(perf, "dot", RED, 1920, 1080, [100, 100], [1800, 980], 60);
        feedback(perfLayer, { "Alpha": 0.98, "Sub-steps": 2 });
        dotComp("limit_700", 700, { "Alpha": 1 });
    }

    app.beginSuppressDialogs();
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
