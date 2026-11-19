// The Metal side of one feedback iteration: previous image + input -> new image.
#pragma once

#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <memory>

#include "FeedbackCache.h"

// The layers a step reads besides its input: Map Layer for displacement, and each group's mask.
enum MapSlot
{
    SLOT_DISPLACE,
    SLOT_ALPHA_MASK,
    SLOT_BLUR_MASK,
    SLOT_LEVELS_MASK,
    SLOT_HSL_MASK,
    SLOT_TRANSFORM_MASK,
    SLOT_DISPLACE_MASK,
    SLOT_EDGE_MASK,
    SLOT_COUNT
};

struct FrameParams
{
    float alpha; // the fade alone
    float alphaSlope, alphaOffset; // the whole alpha curve: slope * alpha + offset
    float blurRadius; // full-resolution pixels
    int sources[SLOT_COUNT]; // MapSource: what each map and mask reads
    float inputBlack, inputWhite, gamma, outputBlack, outputWhite;
    float hueDegrees, saturation, lightness;
    int channelX, channelY;
    float amountX, amountY; // full-resolution pixels
    float centreX, centreY; // layer pixels at the current resolution, as AE gives a point
    float scaleX, scaleY; // 1 leaves the size alone
    float rotationDegrees; // clockwise
    // One step's move about the centre, at full resolution: x -> linear (x - centre) + centre + move.
    // Worked out from the controls above, and divided between sub-steps as a whole.
    float linear[4]; // row-major 2x2
    float moveX, moveY;
    int edgeType;
    float edgeSize; // full-resolution pixels between the samples either side
    float edgeSmoothing; // 0 samples the image as it is, 1 blurs it by half the spacing first
    float edgeStrength; // 0 is off
    int edgeBlend;
    int compositeMode;
    int splitsInput; // all but In Front and Behind with sub-steps: each adds only its share of the input
    float frameAlpha; // the fades that share is worked out from, per frame and per sub-step
    float stepAlpha;
};

// x' = a x + b y + tx, y' = c x + d y + ty.
struct Affine
{
    float a = 1, b = 0, c = 0, d = 1, tx = 0, ty = 0;
};

// Where Transform moves a point in one step, in layer pixels at the current resolution. A reduced
// resolution that differs across and down rescales the full-resolution move to match.
inline Affine TrailTransform(const FrameParams& f, float resolutionX, float resolutionY)
{
    Affine t;
    t.a = f.linear[0];
    t.b = f.linear[1] * resolutionX / resolutionY;
    t.c = f.linear[2] * resolutionY / resolutionX;
    t.d = f.linear[3];
    // centre + move + A (x - centre)
    t.tx = f.centreX + f.moveX * resolutionX - (t.a * f.centreX + t.b * f.centreY);
    t.ty = f.centreY + f.moveY * resolutionY - (t.c * f.centreX + t.d * f.centreY);
    return t;
}

// The least scale a step takes, which stands for 0: a move that can't be undone can't say where each
// pixel came from.
constexpr float kMinScale = 1e-4f;

// Scale at 0 shrinks the trail to nothing. Taken at kMinScale, the pixel at the move's fixed point would
// still sample itself, and stay, so the moved trail is cleared outright instead.
inline bool CollapsesTrail(const FrameParams& f)
{
    return std::min(f.scaleX, f.scaleY) <= kMinScale;
}

inline bool MovesTrail(const FrameParams& f)
{
    return f.moveX != 0.0f || f.moveY != 0.0f || f.linear[0] != 1.0f || f.linear[1] != 0.0f || f.linear[2] != 0.0f ||
           f.linear[3] != 1.0f;
}

// A radius covers about two sigma, the way most blur tools work. The canvas has to allow for what this
// produces, so PreRender and the renderer share it.
inline float BlurSigma(float radius)
{
    return radius * 0.5f;
}

// A Gaussian is cut off this many sigma out, past which too little of it is left to matter.
const float kGaussianTails = 3.0f;

// Edge samples `spacing` apart read the image blurred by this much. At full smoothing, detail finer
// than their spacing can't alias.
inline float EdgeSmoothingSigma(float spacing, float smoothing)
{
    return spacing * 0.5f * smoothing;
}

// How far beyond itself an edge pass can reach: one sample out, plus the tails of what it reads at
// full smoothing.
inline float EdgeReach(float spacing)
{
    return spacing + kGaussianTails * EdgeSmoothingSigma(spacing, 1.0f);
}

// A buffer plus where it sits on the canvas. A null buffer means nothing is there. A warped image is
// found through a projective map from canvas pixels instead of an offset.
struct PlacedImage
{
    id<MTLBuffer> buffer;
    int pitch;
    int offsetX, offsetY;
    int width, height;
    bool warped;
    float fromCanvas[9]; // row-major 3x3
};

// Steps queued up to run together, as waiting for the GPU after each one wastes time. A step's buffers
// come back to this rather than the shared pool until the batch has run: another instance of the effect
// can run while a batch is open, and must not write over what the batch has still to read.
struct StepBatch
{
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> commands;
    std::vector<id<MTLBuffer>> freeBuffers;
    bool failed = false; // a buffer couldn't be had, so what's queued is incomplete
};

class FeedbackRenderer
{
public:
    static std::unique_ptr<FeedbackRenderer> create(id<MTLDevice> device, id<MTLLibrary> library);

    FeedbackCache& cache() { return *mCache; }

    // Queues one step, and returns the image it will produce. A nil prev starts the loop from transparent
    // black. Nothing here can be read outside the batch until it has been flushed.
    id<MTLBuffer> step(
        StepBatch& batch,
        const CanvasKey& canvas,
        id<MTLBuffer> prev,
        const PlacedImage& input,
        const PlacedImage (&maps)[SLOT_COUNT],
        const FrameParams& params);

    // Runs what's queued and waits for it. Returns false if the GPU work failed.
    bool flush(StepBatch& batch);

    // Each returns a new image from the scratch pool, or nil if the GPU work failed.
    // reframe moves an image onto another canvas; compositeInput lays the input over an image.
    id<MTLBuffer> reframe(
        id<MTLCommandQueue> queue,
        const CanvasKey& from,
        const CanvasKey& to,
        id<MTLBuffer> image);

    id<MTLBuffer> compositeInput(
        id<MTLCommandQueue> queue,
        const CanvasKey& canvas,
        id<MTLBuffer> image,
        const PlacedImage& input,
        int compositeMode);

    // With canvasFromOutput, each output pixel is looked up through it rather than by offset.
    bool copyOut(
        id<MTLCommandQueue> queue,
        const CanvasKey& canvas,
        id<MTLBuffer> image,
        const PlacedImage& output,
        const float* canvasFromOutput = nullptr);

private:
    FeedbackRenderer() = default;

    // An image `factor` times smaller than the canvas. `owned` if it's this step's to hand back early.
    struct SmallBlur
    {
        id<MTLBuffer> buffer;
        int width, height, factor;
        bool owned = false;
    };

    // The image to be blurred, halved again and again as far as any blur needs, shared by every blur
    // of it in a step. [j] is 2^j times smaller, and [0] the image itself.
    struct Pyramid
    {
        std::vector<SmallBlur> levels;
    };

    SmallBlur pyramidLevel(id<MTLComputeCommandEncoder> encoder, Pyramid& pyramid, int level, StepBatch& batch,
                           std::vector<id<MTLBuffer>>& scratch);

    // Wide blurs are worked out on a smaller image, which this leaves small.
    // `maxDirectSigma` is the widest blur run at a level before moving to the next smaller one.
    // `repeatEdge` repeats the border beyond the image, rather than reading it as transparent.
    SmallBlur encodeSmallBlur(
        id<MTLComputeCommandEncoder> encoder,
        Pyramid& pyramid,
        float sigmaX,
        float sigmaY,
        float maxDirectSigma,
        bool repeatEdge,
        StepBatch& batch,
        std::vector<id<MTLBuffer>>& scratch);

    // The blur: Gaussians from far narrower to far wider than `width` mixed into a long-tailed blur
    // whose widths add, as a Cauchy distribution's do, so a copy spreads in step with its age rather
    // than its square root. Each is worked out at its own size and they're
    // summed from the smallest up.
    id<MTLBuffer> encodeSpread(
        id<MTLComputeCommandEncoder> encoder,
        Pyramid& pyramid,
        const CanvasKey& canvas,
        float widthX,
        float widthY,
        StepBatch& batch,
        std::vector<id<MTLBuffer>>& scratch);

    // Hands a buffer back for the rest of the step to reuse, once everything reading it is encoded.
    // Dispatches in one encoder run in order, so nothing later can overwrite it too soon.
    void release(id<MTLBuffer> buffer, StepBatch& batch, std::vector<id<MTLBuffer>>& scratch);

    id<MTLBuffer> acquireForBatch(StepBatch& batch, size_t bytes);

    id<MTLComputePipelineState> mDisplace, mEdge, mBlurDown, mBlurAxis, mAddScaled, mBlurByMap, mGradeAndComposite, mComposite, mReframe, mCopyOut;
    id<MTLBuffer> mEmpty; // bound wherever a kernel's optional buffer is absent
    std::unique_ptr<FeedbackCache> mCache;
};
