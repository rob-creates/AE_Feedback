// Parameter structs shared by the plugin and shaders/Feedback.metal.
// All pitches are in pixels per row. Pixels are BGRA float4. AE's worlds hold straight alpha, while
// the feedback image is premultiplied so that blurs and composites are correct; the kernels convert
// where the two meet.
#pragma once

#define FEEDBACK_BLUR_MAX_RADIUS 12

// How one image goes over another: the input over the trail, or edges over the trail. The values are
// the popup's, so saved projects keep their choice. From Multiply on, the top image is blended like a
// layer above with that mode: where either is transparent, the other shows as itself.
enum CompositeMode
{
    COMPOSITE_INPUT_IN_FRONT = 1,
    COMPOSITE_INPUT_BEHIND = 2,
    COMPOSITE_ADD = 3,
    COMPOSITE_SCREEN = 4,
    COMPOSITE_MULTIPLY = 5,
    COMPOSITE_LIGHTEN = 6,
    COMPOSITE_DARKEN = 7,
    COMPOSITE_OVERLAY = 8,
    COMPOSITE_ADD_SUBTRACT = 9, // bottom + 2 * top - 1: brighter than grey adds, darker subtracts
    COMPOSITE_DIFFERENCE = 10
};

// How a kernel reads between pixels. The values are Pixel Sampling's popup's, so saved projects keep
// their choice.
enum SampleFilter
{
    FILTER_BILINEAR = 1,
    FILTER_BICUBIC = 2,
    FILTER_NEAREST = 3 // Pixel: the nearest pixel as it is
};

enum EdgeType
{
    EDGE_COLOUR = 1,
    EDGE_LUMINANCE = 2
};

// What a map reads: nothing, the trail itself (Map Layer set to the effect's own layer at Effects &
// Masks), or a layer from AE.
enum MapSource
{
    MAP_NONE = 0,
    MAP_TRAIL = 1,
    MAP_LAYER = 2
};

enum MapChannel
{
    MAP_CHANNEL_RED = 1,
    MAP_CHANNEL_GREEN = 2,
    MAP_CHANNEL_BLUE = 3,
    MAP_CHANNEL_ALPHA = 4,
    MAP_CHANNEL_LUMA = 5
};

// Where a map is, for a kernel reading it: a layer from AE, or the feedback image itself.
struct MapLookup
{
    int pitch;
    int offsetX; // map pixel = canvas pixel - offset
    int offsetY;
    int width;
    int height;
    int isStraight; // a layer from AE, rather than the feedback image itself
    // With the trail in comp space, the map is in its layer: where each canvas pixel falls in it.
    int warped;
    float fromCanvas[9]; // row-major 3x3, to map pixel coordinates
};

// A group's mask: how much of the group each pixel gets, from its luminance. Without `use`, all of it.
struct Mask
{
    int use;
    MapLookup map;
};

struct DisplaceParams
{
    int srcPitch;
    int dstPitch;
    int width;
    int height;
    int channelX;
    int channelY;
    float amountX; // pixels of shift at full channel value
    float amountY;
    // Where each pixel's shifted position came from before Transform moved it, as canvas pixels:
    // source = inverse * position + offset.
    float inverse[4]; // row-major 2x2
    float offsetX;
    float offsetY;
    MapLookup map;
    Mask transformMask; // each pixel moves this much of the way Transform takes it
    Mask displaceMask; // and shifts this much of its displacement
    int filter; // SampleFilter: how each pixel is read from where it came from
};

struct EdgeParams
{
    int srcPitch;
    int dstPitch;
    int type;
    int blend;
    float distanceX; // canvas pixels between the samples either side
    float distanceY;
    float strength;
    // The image the samples read: the source blurred to their spacing, `smoothFactor` times smaller.
    int smoothPitch;
    int smoothWidth;
    int smoothHeight;
    int smoothFactor;
    Mask mask; // scales the edges
};

// Moving to and from the smaller image that wide blurs run on.
struct BlurResizeParams
{
    int srcPitch;
    int dstPitch;
    int srcWidth;
    int srcHeight;
    int factor;
};

// Adds a weighted share of a smaller image, scaled up, to a full-size one. With `first`, replaces it.
struct AddScaledParams
{
    int srcPitch;
    int dstPitch;
    int srcWidth;
    int srcHeight;
    int factor;
    float weight;
    int first;
};

// Blurs each pixel by its own share of the radius: Blur Mask Layer's luminance, from none at black to
// all of it at white. Blends the two blurs either side of that share, of the unblurred image and blurs
// at 1/8, 1/4, 1/2 and all of the radius.
struct BlurMapParams
{
    int pitch; // of the image and its blurs
    int width;
    int height;
    MapLookup map;
};

struct BlurAxisParams
{
    int srcPitch;
    int dstPitch;
    int width;
    int height;
    int dirX;
    int dirY;
    int radius;
    int repeatEdge; // Repeats the canvas's border beyond it, rather than reading it as transparent.
    float weights[FEEDBACK_BLUR_MAX_RADIUS + 1];
};

struct GradeParams
{
    int srcPitch;
    // The trail can come smaller than the canvas, `srcFactor` times, from a wide blur, and is then
    // sampled between its pixels as it would have been scaled up.
    int srcWidth;
    int srcHeight;
    int srcFactor;
    int dstPitch;
    int applyLevels;
    int applyHSL;
    float inputBlack;
    float inputWhite;
    float inverseGamma;
    float outputBlack;
    float outputWhite;
    float hueShift; // turns
    float saturation; // multiplier
    float lightness; // offset
    float alphaSlope; // new alpha = slope * alpha + offset
    float alphaOffset;
    // How much of each a pixel gets, between leaving it as it was and the whole change.
    Mask levelsMask;
    Mask hslMask;
    Mask alphaMask;
};

struct CompositeParams
{
    int prevPitch;
    int inputPitch;
    int dstPitch;
    int inputOffsetX; // input pixel = canvas pixel - offset
    int inputOffsetY;
    int inputWidth;
    int inputHeight;
    int mode;
    int splitsInput; // all but In Front and Behind with sub-steps: each adds only its share of the input
    float frameAlpha;
    float stepAlpha;
    // With the trail in comp space, the input is drawn through the layer's own move.
    int inputWarped;
    float inputFromCanvas[9]; // row-major 3x3, to input pixel coordinates
    int filter; // SampleFilter: how the input is read, when warped
    float inputOpacity; // Opacity: the input is laid in as if this transparent
};

struct StepParams
{
    GradeParams grade;
    CompositeParams composite;
};

struct ReframeParams
{
    int srcPitch;
    int dstPitch;
    int srcWidth;
    int srcHeight;
    int offsetX; // source pixel = destination pixel + offset
    int offsetY;
};

struct CopyOutParams
{
    int srcPitch;
    int dstPitch;
    int offsetX; // canvas pixel = output pixel + offset
    int offsetY;
    // With the trail in comp space, each output pixel looks up where the layer puts it in comp.
    int warped;
    float canvasFromOutput[9]; // row-major 3x3, to canvas pixel coordinates
    int canvasHeight; // the canvas's width is srcPitch
    int filter; // SampleFilter: how each output pixel is read from the canvas, when warped
};
