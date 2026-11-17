// Where a 2D layer puts its pixels in its comp, at any moment: its own transform and its parents'.
// Used to keep a trail in comp space, and to put Map Layer where it is in the comp.
#pragma once

#include <cstdint>

#include "AE_Effect.h"
#include "AE_GeneralPlug.h"

// A projective map of the plane, row-major, acting on column vectors (x, y, 1).
struct Projective
{
    double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    static Projective translate(double dx, double dy) { return {{1, 0, dx, 0, 1, dy, 0, 0, 1}}; }
    static Projective scale(double sx, double sy) { return {{sx, 0, 0, 0, sy, 0, 0, 0, 1}}; }

    // This after `first`.
    Projective after(const Projective& first) const;
    bool inverse(Projective* out) const;
    // False where the point has no place on the plane.
    bool apply(double x, double y, double* outX, double* outY) const;
    void toFloats(float out[9]) const;
};

class LayerMotion
{
public:
    // Reads what doesn't change from moment to moment. If any of it can't be read, `usable` is false.
    // It's false for a 3D layer too: AE puts that through the camera after its effects, so its pixels
    // have no fixed place in the comp. To work on what a 3D layer looks like, precompose it.
    // The layer the effect is on, or with `other`, that layer in the same comp.
    LayerMotion(PF_InData* in_data, PF_OutData* out_data, AEGP_LayerIDVal other = AEGP_LayerIDVal_NONE);

    bool usable() const { return mUsable; }
    // Shape and text layers, and collapsed precomps, are drawn in comp space already, so their
    // effects see the comp as it is and need nothing moved.
    bool drawnInComp() const { return mDrawnInComp; }
    A_long compWidth() const { return mCompWidth; }
    A_long compHeight() const { return mCompHeight; }

    // Layer pixels to comp pixels, both at full resolution, at a time in the effect's own units. A
    // layer drawn in comp space already has its pixels where they are in the comp.
    bool layerToComp(A_long value, A_u_long scale, Projective* out);

    // Changes whenever where the layer goes at any frame from `from` to `to` changes. Zero if it
    // can't be read.
    uint64_t fingerprint(A_long from, A_long to, A_long step);

private:
    PF_InData* mInData;
    PF_OutData* mOutData;
    AEGP_LayerH mLayer = nullptr;
    bool mUsable = false;
    bool mDrawnInComp = false;
    A_long mCompWidth = 0, mCompHeight = 0;
};
