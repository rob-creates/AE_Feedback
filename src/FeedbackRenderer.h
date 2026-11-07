// The Metal side of one feedback iteration: previous image + input -> new image.
#pragma once

#import <Metal/Metal.h>
#include <memory>

#include "FeedbackCache.h"

struct FrameParams
{
    float alpha; // the fade alone, which the input's strength in Add is worked out from
    float alphaSlope, alphaOffset; // the whole alpha curve: slope * alpha + offset
    float blurRadius; // full-resolution pixels
    float inputBlack, inputWhite, gamma, outputBlack, outputWhite;
    float hueDegrees, saturation, lightness;
    int displaceFromLayer;
    int channelX, channelY;
    float amountX, amountY; // full-resolution pixels
    int compositeMode;
    float inputGain; // how much of the input one iteration adds
    int screenGain; // in Screen, derive the input's strength from the two alphas below
    float frameAlpha;
    float stepAlpha;
};

// A radius covers about two sigma, the way most blur tools work. The canvas has to allow for what this
// produces, so PreRender and the renderer share it.
inline float BlurSigma(float radius)
{
    return radius * 0.5f;
}

// A buffer plus where it sits on the canvas. A null buffer means nothing is there.
struct PlacedImage
{
    id<MTLBuffer> buffer;
    int pitch;
    int offsetX, offsetY;
    int width, height;
};

class FeedbackRenderer
{
public:
    static std::unique_ptr<FeedbackRenderer> create(id<MTLDevice> device, id<MTLLibrary> library);

    FeedbackCache& cache() { return *mCache; }

    // Returns a new image from the scratch pool, or nil if the GPU work failed.
    // A nil prev starts the loop from transparent black.
    id<MTLBuffer> step(
        id<MTLCommandQueue> queue,
        const CanvasKey& canvas,
        id<MTLBuffer> prev,
        const PlacedImage& input,
        const PlacedImage& map,
        const FrameParams& params);

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

    bool copyOut(id<MTLCommandQueue> queue, const CanvasKey& canvas, id<MTLBuffer> image, const PlacedImage& output);

private:
    FeedbackRenderer() = default;

    id<MTLBuffer> encodeBlur(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> source,
        const CanvasKey& canvas,
        float sigmaX,
        float sigmaY,
        std::vector<id<MTLBuffer>>& scratch);

    id<MTLComputePipelineState> mDisplace, mBlurDown, mBlurAxis, mBlurUp, mGrade, mComposite, mReframe, mCopyOut;
    id<MTLBuffer> mEmpty; // bound wherever a kernel's optional buffer is absent
    std::unique_ptr<FeedbackCache> mCache;
};
