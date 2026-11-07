#include "FeedbackRenderer.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>

#include "ShaderTypes.h"

namespace
{
const float kMinSigma = 0.05f;
const float kMaxDirectSigma = FEEDBACK_BLUR_MAX_RADIUS / 3.0f;

// The buffers bind in order from index 0, followed by the parameters.
template <typename Params>
void Encode(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers,
    const Params& params,
    int width,
    int height)
{
    NSUInteger index = 0;
    for (id<MTLBuffer> buffer : buffers)
        [encoder setBuffer:buffer offset:0 atIndex:index++];
    [encoder setBytes:&params length:sizeof(params) atIndex:index];
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
}

bool Succeeded(id<MTLCommandBuffer> commands)
{
    [commands commit];
    [commands waitUntilCompleted];
    return commands.status == MTLCommandBufferStatusCompleted;
}

// One kernel on its own, waited for.
template <typename Params>
bool RunOnce(
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers,
    const Params& params,
    int width,
    int height)
{
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    Encode(encoder, pipeline, buffers, params, width, height);
    [encoder endEncoding];
    return Succeeded(commands);
}

void FillWeights(BlurAxisParams& p, float sigma)
{
    // Sampling a Gaussian this narrow would lose almost all of it, and sub-steps rely on many small
    // blurs adding up. Three taps with the exact variance do that.
    if (sigma < 0.5f)
    {
        p.radius = 1;
        p.weights[0] = 1.0f - sigma * sigma;
        p.weights[1] = 0.5f * sigma * sigma;
        return;
    }

    p.radius = std::min(int(std::ceil(sigma * 3.0f)), FEEDBACK_BLUR_MAX_RADIUS);
    float total = 0.0f;
    for (int i = 0; i <= p.radius; i++)
    {
        p.weights[i] = std::exp(-0.5f * float(i * i) / (sigma * sigma));
        total += i == 0 ? p.weights[i] : 2.0f * p.weights[i];
    }
    for (int i = 0; i <= p.radius; i++)
        p.weights[i] /= total;
}

// A missing input has zero size, so the kernel never reads it.
CompositeParams MakeCompositeParams(const CanvasKey& canvas, const PlacedImage& input, int mode)
{
    CompositeParams p{};
    p.prevPitch = canvas.width;
    p.dstPitch = canvas.width;
    p.mode = mode;
    p.inputGain = 1.0f;
    p.inputPitch = input.pitch;
    p.inputOffsetX = input.offsetX;
    p.inputOffsetY = input.offsetY;
    p.inputWidth = input.width;
    p.inputHeight = input.height;
    return p;
}
} // namespace

std::unique_ptr<FeedbackRenderer> FeedbackRenderer::create(id<MTLDevice> device, id<MTLLibrary> library)
{
    std::unique_ptr<FeedbackRenderer> r(new FeedbackRenderer());

    auto pipeline = [&](NSString* name) -> id<MTLComputePipelineState>
    {
        NSError* error = nil;
        id<MTLFunction> fn = [library newFunctionWithName:name];
        return fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
    };
    r->mDisplace = pipeline(@"displace");
    r->mBlurDown = pipeline(@"blurDown");
    r->mBlurAxis = pipeline(@"blurAxis");
    r->mBlurUp = pipeline(@"blurUp");
    r->mGrade = pipeline(@"grade");
    r->mComposite = pipeline(@"composite");
    r->mReframe = pipeline(@"reframe");
    r->mCopyOut = pipeline(@"copyOut");
    if (!r->mDisplace || !r->mBlurDown || !r->mBlurAxis || !r->mBlurUp || !r->mGrade || !r->mComposite ||
        !r->mReframe || !r->mCopyOut)
        return nullptr;

    r->mEmpty = [device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
    r->mCache = std::make_unique<FeedbackCache>(device);
    return r;
}

id<MTLBuffer> FeedbackRenderer::encodeBlur(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> source,
    const CanvasKey& canvas,
    float sigmaX,
    float sigmaY,
    std::vector<id<MTLBuffer>>& scratch)
{
    auto acquire = [&](size_t bytes)
    {
        id<MTLBuffer> buffer = mCache->acquireScratch(bytes);
        scratch.push_back(buffer);
        return buffer;
    };

    // Very wide blurs run on a smaller image, so their cost doesn't grow with the radius.
    int factor = 1;
    while (std::max(sigmaX, sigmaY) / factor > kMaxDirectSigma)
        factor *= 2;

    // The box filter already contributes some blur, so the Gaussian only supplies the rest.
    auto lowSigma = [&](float s)
    {
        float boxVariance = (float(factor * factor) - 1.0f) / 12.0f;
        return std::sqrt(std::max(s * s - boxVariance, 0.0f)) / float(factor);
    };

    const int w = (canvas.width + factor - 1) / factor;
    const int h = (canvas.height + factor - 1) / factor;
    const size_t lowBytes = size_t(w) * size_t(h) * 16;

    id<MTLBuffer> current = source;

    if (factor > 1)
    {
        BlurResizeParams p{canvas.width, w, canvas.width, canvas.height, factor};
        id<MTLBuffer> low = acquire(lowBytes);
        Encode(encoder, mBlurDown, {current, low}, p, w, h);
        current = low;
    }

    for (int axis = 0; axis < 2; axis++)
    {
        float sigma = lowSigma(axis == 0 ? sigmaX : sigmaY);
        if (sigma < kMinSigma)
            continue;
        BlurAxisParams p{};
        p.srcPitch = w;
        p.dstPitch = w;
        p.width = w;
        p.height = h;
        p.dirX = axis == 0 ? 1 : 0;
        p.dirY = axis == 0 ? 0 : 1;
        FillWeights(p, sigma);
        id<MTLBuffer> next = acquire(lowBytes);
        Encode(encoder, mBlurAxis, {current, next}, p, w, h);
        current = next;
    }

    if (factor > 1)
    {
        BlurResizeParams p{w, canvas.width, w, h, factor};
        id<MTLBuffer> full = acquire(canvas.bytes());
        Encode(encoder, mBlurUp, {current, full}, p, canvas.width, canvas.height);
        current = full;
    }
    return current;
}

id<MTLBuffer> FeedbackRenderer::step(
    id<MTLCommandQueue> queue,
    const CanvasKey& canvas,
    id<MTLBuffer> prev,
    const PlacedImage& input,
    const PlacedImage& map,
    const FrameParams& f)
{
    const int width = canvas.width;
    const int height = canvas.height;
    const float scaleX = ScaleValue(canvas.downsampleX);
    const float scaleY = ScaleValue(canvas.downsampleY);

    std::vector<id<MTLBuffer>> scratch;
    auto acquire = [&]
    {
        id<MTLBuffer> buffer = mCache->acquireScratch(canvas.bytes());
        scratch.push_back(buffer);
        return buffer;
    };
    id<MTLBuffer> result = mCache->acquireScratch(canvas.bytes());

    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLBuffer> current = prev;

    if (!current)
    {
        current = acquire();
        id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
        [blit fillBuffer:current range:NSMakeRange(0, canvas.bytes()) value:0];
        [blit endEncoding];
    }

    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];

    // Displacement
    const float amountX = f.amountX * scaleX;
    const float amountY = f.amountY * scaleY;
    const bool mapMissing = f.displaceFromLayer && !map.buffer;
    if ((amountX != 0.0f || amountY != 0.0f) && !mapMissing)
    {
        DisplaceParams p{};
        p.srcPitch = width;
        p.dstPitch = width;
        p.width = width;
        p.height = height;
        p.channelX = f.channelX;
        p.channelY = f.channelY;
        p.amountX = amountX;
        p.amountY = amountY;
        id<MTLBuffer> mapBuffer = current;
        if (f.displaceFromLayer)
        {
            mapBuffer = map.buffer;
            p.mapIsStraight = 1;
            p.mapPitch = map.pitch;
            p.mapOffsetX = map.offsetX;
            p.mapOffsetY = map.offsetY;
            p.mapWidth = map.width;
            p.mapHeight = map.height;
        }
        else
        {
            p.mapPitch = width;
            p.mapWidth = width;
            p.mapHeight = height;
        }
        id<MTLBuffer> next = acquire();
        Encode(encoder, mDisplace, {current, mapBuffer, next}, p, width, height);
        current = next;
    }

    // Blur
    const float sigmaX = BlurSigma(f.blurRadius) * scaleX;
    const float sigmaY = BlurSigma(f.blurRadius) * scaleY;
    if (std::max(sigmaX, sigmaY) >= kMinSigma)
        current = encodeBlur(encoder, current, canvas, sigmaX, sigmaY, scratch);

    // Levels, HSL, Alpha. Neutral settings skip the maths, so they're exactly a no-op.
    GradeParams grade{};
    grade.srcPitch = width;
    grade.dstPitch = width;
    grade.applyLevels = !(f.inputBlack == 0.0f && f.inputWhite == 1.0f && f.gamma == 1.0f &&
                          f.outputBlack == 0.0f && f.outputWhite == 1.0f);
    grade.applyHSL = !(f.hueDegrees == 0.0f && f.saturation == 1.0f && f.lightness == 0.0f);
    grade.inputBlack = f.inputBlack;
    grade.inputWhite = f.inputWhite;
    grade.inverseGamma = 1.0f / std::max(f.gamma, 0.01f);
    grade.outputBlack = f.outputBlack;
    grade.outputWhite = f.outputWhite;
    grade.hueShift = f.hueDegrees / 360.0f;
    grade.saturation = f.saturation;
    grade.lightness = f.lightness;
    grade.alphaSlope = f.alphaSlope;
    grade.alphaOffset = f.alphaOffset;
    {
        id<MTLBuffer> next = acquire();
        Encode(encoder, mGrade, {current, next}, grade, width, height);
        current = next;
    }

    // Composite with this frame's input.
    {
        CompositeParams p = MakeCompositeParams(canvas, input, f.compositeMode);
        p.inputGain = f.inputGain;
        p.screenGain = f.screenGain;
        p.frameAlpha = f.frameAlpha;
        p.stepAlpha = f.stepAlpha;
        Encode(encoder, mComposite, {current, input.buffer ? input.buffer : mEmpty, result}, p, width, height);
    }

    [encoder endEncoding];
    bool ok = Succeeded(commands);

    for (id<MTLBuffer> buffer : scratch)
        mCache->releaseScratch(buffer);

    if (!ok)
    {
        mCache->releaseScratch(result);
        return nil;
    }
    return result;
}

id<MTLBuffer> FeedbackRenderer::reframe(
    id<MTLCommandQueue> queue,
    const CanvasKey& from,
    const CanvasKey& to,
    id<MTLBuffer> image)
{
    id<MTLBuffer> result = mCache->acquireScratch(to.bytes());
    ReframeParams p{from.width, to.width, from.width, from.height, to.left - from.left, to.top - from.top};
    if (!RunOnce(queue, mReframe, {image, result}, p, to.width, to.height))
    {
        mCache->releaseScratch(result);
        return nil;
    }
    return result;
}

id<MTLBuffer> FeedbackRenderer::compositeInput(
    id<MTLCommandQueue> queue,
    const CanvasKey& canvas,
    id<MTLBuffer> image,
    const PlacedImage& input,
    int compositeMode)
{
    id<MTLBuffer> result = mCache->acquireScratch(canvas.bytes());
    CompositeParams p = MakeCompositeParams(canvas, input, compositeMode);
    if (!RunOnce(queue, mComposite, {image, input.buffer ? input.buffer : mEmpty, result}, p, canvas.width, canvas.height))
    {
        mCache->releaseScratch(result);
        return nil;
    }
    return result;
}

bool FeedbackRenderer::copyOut(
    id<MTLCommandQueue> queue, const CanvasKey& canvas, id<MTLBuffer> image, const PlacedImage& output)
{
    const int width = std::min(output.width, canvas.width - output.offsetX);
    const int height = std::min(output.height, canvas.height - output.offsetY);
    if (width <= 0 || height <= 0)
        return true;

    CopyOutParams p{canvas.width, output.pitch, output.offsetX, output.offsetY};
    return RunOnce(queue, mCopyOut, {image, output.buffer}, p, width, height);
}
