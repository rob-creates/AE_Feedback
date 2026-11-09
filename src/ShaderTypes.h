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

enum EdgeType
{
    EDGE_COLOUR = 1,
    EDGE_LUMINANCE = 2
};

enum MapChannel
{
    MAP_CHANNEL_RED = 1,
    MAP_CHANNEL_GREEN = 2,
    MAP_CHANNEL_BLUE = 3,
    MAP_CHANNEL_ALPHA = 4,
    MAP_CHANNEL_LUMA = 5
};

struct DisplaceParams
{
    int srcPitch;
    int mapPitch;
    int dstPitch;
    int width;
    int height;
    int mapOffsetX; // map pixel = canvas pixel - offset
    int mapOffsetY;
    int mapWidth;
    int mapHeight;
    int channelX;
    int channelY;
    int mapIsStraight; // a layer from AE, rather than the feedback image itself
    float amountX; // pixels of shift at full channel value
    float amountY;
    // Where each pixel's shifted position came from before Transform moved it, as canvas pixels:
    // source = inverse * position + offset.
    float inverse[4]; // row-major 2x2
    float offsetX;
    float offsetY;
};

struct EdgeParams
{
    int srcPitch;
    int dstPitch;
    int width;
    int height;
    int type;
    int blend;
    float distanceX; // canvas pixels between the samples either side
    float distanceY;
    float strength;
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

struct BlurAxisParams
{
    int srcPitch;
    int dstPitch;
    int width;
    int height;
    int dirX;
    int dirY;
    int radius;
    float weights[FEEDBACK_BLUR_MAX_RADIUS + 1];
};

struct GradeParams
{
    int srcPitch;
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
    int splitsInput; // Add and Screen with sub-steps: each adds only its share of the input
    float frameAlpha;
    float stepAlpha;
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
};
