#include "FeedbackRenderer.h"

#include <algorithm>
#include <cmath>

#include "ShaderTypes.h"

namespace
{
const float kMinSigma = 0.05f;
const float kMaxDirectSigma = FEEDBACK_BLUR_MAX_RADIUS / 3.0f;

void Dispatch(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, int width, int height)
{
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
}

void FillWeights(BlurAxisParams& p, float sigma)
{
    // Sampling a Gaussian this narrow would lose almost all of it, and sub-steps rely on many small
    // blurs adding up. Three taps with the exact variance do that.
    if (sigma > 0.0f && sigma < 0.5f)
    {
        p.radius = 1;
        p.weights[0] = 1.0f - sigma * sigma;
        p.weights[1] = 0.5f * sigma * sigma;
        return;
    }

    p.radius = std::clamp(int(std::ceil(sigma * 3.0f)), 0, FEEDBACK_BLUR_MAX_RADIUS);
    float total = 0.0f;
    for (int i = 0; i <= FEEDBACK_BLUR_MAX_RADIUS; i++)
    {
        float w = (sigma > 0.0f && i <= p.radius) ? std::exp(-0.5f * float(i * i) / (sigma * sigma)) : 0.0f;
        p.weights[i] = w;
        total += i == 0 ? w : 2.0f * w;
    }
    if (total <= 0.0f)
    {
        p.weights[0] = 1.0f;
        return;
    }
    for (int i = 0; i <= FEEDBACK_BLUR_MAX_RADIUS; i++)
        p.weights[i] /= total;
}

CompositeParams MakeCompositeParams(const CanvasKey& canvas, const PlacedImage& input, int mode)
{
    CompositeParams p{};
    p.prevPitch = canvas.width;
    p.dstPitch = canvas.width;
    p.width = canvas.width;
    p.height = canvas.height;
    p.mode = mode;
    p.inputGain = 1.0f;
    if (input.buffer)
    {
        p.inputPitch = input.pitch;
        p.inputOffsetX = input.offsetX;
        p.inputOffsetY = input.offsetY;
        p.inputWidth = input.width;
        p.inputHeight = input.height;
    }
    return p;
}

bool Succeeded(id<MTLCommandBuffer> commands)
{
    [commands commit];
    [commands waitUntilCompleted];
    return commands.status == MTLCommandBufferStatusCompleted;
}
} // namespace

std::unique_ptr<FeedbackRenderer> FeedbackRenderer::create(id<MTLDevice> device, id<MTLLibrary> library)
{
    std::unique_ptr<FeedbackRenderer> r(new FeedbackRenderer());
    r->mDevice = device;

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
    int currentPitch = canvas.width;

    if (factor > 1)
    {
        BlurDownParams p{canvas.width, w, canvas.width, canvas.height, w, h, factor};
        id<MTLBuffer> low = acquire(lowBytes);
        [encoder setBuffer:current offset:0 atIndex:0];
        [encoder setBuffer:low offset:0 atIndex:1];
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        Dispatch(encoder, mBlurDown, w, h);
        current = low;
        currentPitch = w;
    }

    for (int axis = 0; axis < 2; axis++)
    {
        float sigma = lowSigma(axis == 0 ? sigmaX : sigmaY);
        if (sigma < kMinSigma)
            continue;
        BlurAxisParams p{};
        p.srcPitch = currentPitch;
        p.dstPitch = w;
        p.width = w;
        p.height = h;
        p.dirX = axis == 0 ? 1 : 0;
        p.dirY = axis == 0 ? 0 : 1;
        FillWeights(p, sigma);
        id<MTLBuffer> next = acquire(lowBytes);
        [encoder setBuffer:current offset:0 atIndex:0];
        [encoder setBuffer:next offset:0 atIndex:1];
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        Dispatch(encoder, mBlurAxis, w, h);
        current = next;
        currentPitch = w;
    }

    if (factor > 1)
    {
        BlurUpParams p{w, canvas.width, w, h, canvas.width, canvas.height, factor};
        id<MTLBuffer> full = acquire(canvas.bytes());
        [encoder setBuffer:current offset:0 atIndex:0];
        [encoder setBuffer:full offset:0 atIndex:1];
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        Dispatch(encoder, mBlurUp, canvas.width, canvas.height);
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
    const float scaleX = float(canvas.downsampleXNum) / float(std::max<A_u_long>(canvas.downsampleXDen, 1));
    const float scaleY = float(canvas.downsampleYNum) / float(std::max<A_u_long>(canvas.downsampleYDen, 1));

    std::vector<id<MTLBuffer>> scratch;
    id<MTLBuffer> result = mCache->acquireScratch(canvas.bytes());

    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLBuffer> current = prev;

    if (!current)
    {
        current = mCache->acquireScratch(canvas.bytes());
        scratch.push_back(current);
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
        id<MTLBuffer> next = mCache->acquireScratch(canvas.bytes());
        scratch.push_back(next);
        [encoder setBuffer:current offset:0 atIndex:0];
        [encoder setBuffer:mapBuffer offset:0 atIndex:1];
        [encoder setBuffer:next offset:0 atIndex:2];
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        Dispatch(encoder, mDisplace, width, height);
        current = next;
    }

    // Blur. Radius maps to sigma the way most blur tools do: a radius covers about two sigma.
    const float sigmaX = f.blurRadius * 0.5f * scaleX;
    const float sigmaY = f.blurRadius * 0.5f * scaleY;
    if (std::max(sigmaX, sigmaY) >= kMinSigma)
        current = encodeBlur(encoder, current, canvas, sigmaX, sigmaY, scratch);

    // Levels, HSL, Alpha. Neutral settings skip the maths, so they're exactly a no-op.
    GradeParams grade{};
    grade.srcPitch = width;
    grade.dstPitch = width;
    grade.width = width;
    grade.height = height;
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
        id<MTLBuffer> next = mCache->acquireScratch(canvas.bytes());
        scratch.push_back(next);
        [encoder setBuffer:current offset:0 atIndex:0];
        [encoder setBuffer:next offset:0 atIndex:1];
        [encoder setBytes:&grade length:sizeof(grade) atIndex:2];
        Dispatch(encoder, mGrade, width, height);
        current = next;
    }

    // Composite with this frame's input.
    {
        CompositeParams p = MakeCompositeParams(canvas, input, f.compositeMode);
        p.inputGain = f.inputGain;
        p.screenGain = f.screenGain;
        p.frameAlpha = f.frameAlpha;
        p.stepAlpha = f.stepAlpha;
        [encoder setBuffer:current offset:0 atIndex:0];
        [encoder setBuffer:(input.buffer ? input.buffer : mEmpty) offset:0 atIndex:1];
        [encoder setBuffer:result offset:0 atIndex:2];
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        Dispatch(encoder, mComposite, width, height);
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
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setBuffer:image offset:0 atIndex:0];
    [encoder setBuffer:result offset:0 atIndex:1];
    [encoder setBytes:&p length:sizeof(p) atIndex:2];
    Dispatch(encoder, mReframe, to.width, to.height);
    [encoder endEncoding];
    if (!Succeeded(commands))
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
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setBuffer:image offset:0 atIndex:0];
    [encoder setBuffer:(input.buffer ? input.buffer : mEmpty) offset:0 atIndex:1];
    [encoder setBuffer:result offset:0 atIndex:2];
    [encoder setBytes:&p length:sizeof(p) atIndex:3];
    Dispatch(encoder, mComposite, canvas.width, canvas.height);
    [encoder endEncoding];
    if (!Succeeded(commands))
    {
        mCache->releaseScratch(result);
        return nil;
    }
    return result;
}

bool FeedbackRenderer::copyOut(
    id<MTLCommandQueue> queue,
    const CanvasKey& canvas,
    id<MTLBuffer> image,
    id<MTLBuffer> output,
    int outputPitch,
    int outputWidth,
    int outputHeight,
    int offsetX,
    int offsetY)
{
    const int width = std::min(outputWidth, canvas.width - offsetX);
    const int height = std::min(outputHeight, canvas.height - offsetY);
    if (width <= 0 || height <= 0)
        return true;

    CopyOutParams p{canvas.width, outputPitch, offsetX, offsetY};
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setBuffer:image offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&p length:sizeof(p) atIndex:2];
    Dispatch(encoder, mCopyOut, width, height);
    [encoder endEncoding];
    return Succeeded(commands);
}
