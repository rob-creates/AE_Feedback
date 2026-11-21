// Shared by build_project.jsx, live_session.jsx and bench_project.jsx.
var FPS = 24;
var MAP_LAYER = "Map Layer";
var BLUR_MAP_LAYER = "Blur Mask Layer";

// Each Feedback control's ID, which AE gives it as its match name, "RPAY Feedback-" and four digits. The
// IDs never change (kDiskIds in src/Feedback.mm), so tests keep working whatever a control is called.
var CONTROL_IDS = {
    "Composite": 1,
    "Opacity": 66,
    "Alpha Max": 2,
    "Alpha Min": 26,
    "Alpha Mask Layer": 55,
    "Blur Radius": 3,
    "Blur Mask Layer": 51,
    "Input Black": 5,
    "Input White": 6,
    "Gamma": 7,
    "Output Black": 8,
    "Output White": 9,
    "Levels Mask Layer": 56,
    "Hue Shift": 12,
    "Saturation": 13,
    "Lightness": 14,
    "HSL Mask Layer": 57,
    "Centre": 32,
    "Position": 47,
    "Uniform Scale": 45,
    "Scale Height": 35,
    "Scale Width": 46,
    "Rotation": 36,
    "Transform Mask Layer": 58,
    "Map Layer": 18,
    "Horizontal Channel": 19,
    "Max Horizontal": 20,
    "Vertical Channel": 21,
    "Max Vertical": 22,
    "Displacement Mask Layer": 59,
    "Edge Type": 39,
    "Edge Strength": 41,
    "Edge Sample Size": 40,
    "Edge Smoothing": 61,
    "Edge Blend": 42,
    "Edge Mask Layer": 60,
    "Sub-steps": 24,
    "Cut-off Precision": 63,
    "Pixel Sampling": 65,
    "Limit Pre-Roll": 49,
    "Pre-Roll Limit (Frames)": 25
};

// A Feedback control, by the name it's listed under here.
function find(fx, name) {
    var id = CONTROL_IDS[name];
    if (id === undefined) throw new Error("No control " + name);
    return fx.property("RPAY Feedback-" + ("000" + id).slice(-4));
}

// Keys `prop` at each frame from 0 with values(frame), held until the next, so it switches rather than
// fading between them.
function heldKeys(prop, frames, values) {
    for (var f = 0; f < frames; f++) prop.setValueAtTime(f / FPS, values(f));
    for (var k = 1; k <= prop.numKeys; k++)
        prop.setInterpolationTypeAtKey(k, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD);
}

function comp(name) {
    for (var i = 1; i <= app.project.numItems; i++) {
        var it = app.project.item(i);
        if (it instanceof CompItem && it.name === name) return it;
    }
    throw new Error("No comp " + name);
}
