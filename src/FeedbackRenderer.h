// The Metal side of one feedback iteration: previous image + input -> new image.
#pragma once

#import <Metal/Metal.h>
#include <cmath>
#include <memory>

#include "FeedbackCache.h"

struct FrameParams
{
    float alpha; // the fade alone
    float alphaSlope, alphaOffset; // the whole alpha curve: slope * alpha + offset
    float blurRadius; // full-resolution pixels
    float inputBlack, inputWhite, gamma, outputBlack, outputWhite;
    float hueDegrees, saturation, lightness;
    int displaceFromLayer;
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
        const PlacedImage& map,
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

    id<MTLBuffer> encodeBlur(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> source,
        const CanvasKey& canvas,
        float sigmaX,
        float sigmaY,
        StepBatch& batch,
        std::vector<id<MTLBuffer>>& scratch);

    id<MTLBuffer> acquireForBatch(StepBatch& batch, size_t bytes);

    id<MTLComputePipelineState> mDisplace, mEdge, mBlurDown, mBlurAxis, mBlurUp, mGradeAndComposite, mComposite, mReframe, mCopyOut;
    id<MTLBuffer> mEmpty; // bound wherever a kernel's optional buffer is absent
    std::unique_ptr<FeedbackCache> mCache;
};
