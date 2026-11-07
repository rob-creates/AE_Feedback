// Shared by build_project.jsx and live_session.jsx.
var FPS = 24;
var MAP_LAYER = "Map Layer (None = Trail)";

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

function comp(name) {
    for (var i = 1; i <= app.project.numItems; i++) {
        var it = app.project.item(i);
        if (it instanceof CompItem && it.name === name) return it;
    }
    throw new Error("No comp " + name);
}
