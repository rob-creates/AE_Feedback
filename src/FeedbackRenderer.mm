#include "FeedbackRenderer.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <functional>
#include <map>
#include <vector>

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

// Spread's mix: a Cauchy distribution of width 1 is Gaussians whose sigmas s are spread so that
// P(s <= x) = erfc(1 / (sqrt(2) x)). Here that's split into octaves, each one Gaussian at its middle,
// from far narrower than 1 to `widest` (in multiples of `width`'s pixels), and weighed to add to 1.
struct SpreadPart
{
    float sigma, weight; // sigma in multiples of the width
};

std::vector<SpreadPart> SpreadParts(float width, float widest)
{
    auto below = [](double x) { return x > 0.0 ? std::erfc(1.0 / (std::sqrt(2.0) * x)) : 0.0; };
    const double last = std::max(double(widest) / std::max(double(width), 1e-6), 1.0);
    std::vector<SpreadPart> parts;
    double total = 0.0;
    for (double lo = 0.0, hi = 0.25; lo < last; lo = hi, hi *= 2.0)
    {
        const double top = std::min(hi, last);
        const double weight = below(top) - below(lo);
        if (weight <= 1e-5)
            continue;
        parts.push_back({float(lo > 0.0 ? std::sqrt(lo * top) : 0.125), float(weight)});
        total += weight;
    }
    for (SpreadPart& part : parts)
        part.weight = float(part.weight / total);
    return parts;
}

// Describes a map for a kernel: a placed layer, or else the trail itself, and returns its buffer.
id<MTLBuffer> FillMap(MapLookup& m, const PlacedImage& image, int source, id<MTLBuffer> trail, int width, int height)
{
    if (source != MAP_LAYER)
    {
        m.pitch = width;
        m.width = width;
        m.height = height;
        return trail;
    }
    m.isStraight = 1;
    m.pitch = image.pitch;
    m.offsetX = image.offsetX;
    m.offsetY = image.offsetY;
    m.width = image.width;
    m.height = image.height;
    m.warped = image.warped;
    if (image.warped)
        std::copy(image.fromCanvas, image.fromCanvas + 9, m.fromCanvas);
    return image.buffer;
}

// Where Transform moves a canvas pixel from, as `inverse * position + offset`.
void InvertTransform(const FrameParams& f, const CanvasKey& canvas, DisplaceParams* p)
{
    const Affine layer = TrailTransform(f, ScaleValue(canvas.downsampleX), ScaleValue(canvas.downsampleY));
    // The same move with the canvas's corner as the origin.
    Affine t = layer;
    t.tx = layer.a * canvas.left + layer.b * canvas.top + layer.tx - canvas.left;
    t.ty = layer.c * canvas.left + layer.d * canvas.top + layer.ty - canvas.top;

    const float det = t.a * t.d - t.b * t.c;
    p->inverse[0] = t.d / det;
    p->inverse[1] = -t.b / det;
    p->inverse[2] = -t.c / det;
    p->inverse[3] = t.a / det;
    p->offsetX = -(p->inverse[0] * t.tx + p->inverse[1] * t.ty);
    p->offsetY = -(p->inverse[2] * t.tx + p->inverse[3] * t.ty);
}

// A missing input has zero size, so the kernel never reads it.
CompositeParams MakeCompositeParams(const CanvasKey& canvas, const PlacedImage& input, int mode)
{
    CompositeParams p{};
    p.prevPitch = canvas.width;
    p.dstPitch = canvas.width;
    p.mode = mode;
    p.inputPitch = input.pitch;
    p.inputOffsetX = input.offsetX;
    p.inputOffsetY = input.offsetY;
    p.inputWidth = input.width;
    p.inputHeight = input.height;
    p.inputWarped = input.warped;
    if (input.warped)
        std::copy(input.fromCanvas, input.fromCanvas + 9, p.inputFromCanvas);
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
    r->mEdge = pipeline(@"edge");
    r->mBlurDown = pipeline(@"blurDown");
    r->mBlurAxis = pipeline(@"blurAxis");
    r->mAddScaled = pipeline(@"addScaled");
    r->mBlurByMap = pipeline(@"blurByMap");
    r->mGradeAndComposite = pipeline(@"gradeAndComposite");
    r->mComposite = pipeline(@"composite");
    r->mReframe = pipeline(@"reframe");
    r->mCopyOut = pipeline(@"copyOut");
    if (!r->mDisplace || !r->mEdge || !r->mBlurDown || !r->mBlurAxis || !r->mAddScaled || !r->mBlurByMap || !r->mGradeAndComposite || !r->mComposite ||
        !r->mReframe || !r->mCopyOut)
        return nullptr;

    r->mEmpty = [device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
    r->mCache = std::make_unique<FeedbackCache>(device);
    return r;
}

id<MTLBuffer> FeedbackRenderer::acquireForBatch(StepBatch& batch, size_t bytes)
{
    for (auto it = batch.freeBuffers.begin(); it != batch.freeBuffers.end(); ++it)
    {
        if ((*it).length >= bytes && (*it).length <= bytes * 2)
        {
            id<MTLBuffer> buffer = *it;
            batch.freeBuffers.erase(it);
            return buffer;
        }
    }
    return mCache->acquireScratch(bytes);
}

bool FeedbackRenderer::flush(StepBatch& batch)
{
    bool ok = true;
    if (batch.commands)
    {
        ok = Succeeded(batch.commands);
        batch.commands = nil;
    }
    for (id<MTLBuffer> buffer : batch.freeBuffers)
        mCache->releaseScratch(buffer);
    batch.freeBuffers.clear();
    return ok;
}

void FeedbackRenderer::release(id<MTLBuffer> buffer, StepBatch& batch, std::vector<id<MTLBuffer>>& scratch)
{
    auto found = std::find(scratch.begin(), scratch.end(), buffer);
    if (found == scratch.end())
        return;
    scratch.erase(found);
    batch.freeBuffers.push_back(buffer);
}

FeedbackRenderer::SmallBlur FeedbackRenderer::pyramidLevel(
    id<MTLComputeCommandEncoder> encoder, Pyramid& pyramid, int level, StepBatch& batch, std::vector<id<MTLBuffer>>& scratch)
{
    while (int(pyramid.levels.size()) <= level)
    {
        const SmallBlur above = pyramid.levels.back();
        SmallBlur next{nil, (above.width + 1) / 2, (above.height + 1) / 2, above.factor * 2};
        next.buffer = acquireForBatch(batch, size_t(next.width) * size_t(next.height) * 16);
        scratch.push_back(next.buffer);
        BlurResizeParams p{above.width, next.width, above.width, above.height, 2};
        Encode(encoder, mBlurDown, {above.buffer, next.buffer}, p, next.width, next.height);
        pyramid.levels.push_back(next);
    }
    return pyramid.levels[size_t(level)];
}

FeedbackRenderer::SmallBlur FeedbackRenderer::encodeSmallBlur(
    id<MTLComputeCommandEncoder> encoder,
    Pyramid& pyramid,
    float sigmaX,
    float sigmaY,
    StepBatch& batch,
    std::vector<id<MTLBuffer>>& scratch)
{
    // Very wide blurs run on a smaller image, so their cost doesn't grow with the radius.
    int level = 0;
    while (std::max(sigmaX, sigmaY) / float(1 << level) > kMaxDirectSigma)
        level++;
    SmallBlur result = pyramidLevel(encoder, pyramid, level, batch, scratch);
    const int factor = result.factor;

    // Halving averages 2x2 blocks, which add up to a box of `factor`. That already contributes some
    // blur, so the Gaussian only supplies the rest.
    auto lowSigma = [&](float s)
    {
        float boxVariance = (float(factor * factor) - 1.0f) / 12.0f;
        return std::sqrt(std::max(s * s - boxVariance, 0.0f)) / float(factor);
    };

    for (int axis = 0; axis < 2; axis++)
    {
        float sigma = lowSigma(axis == 0 ? sigmaX : sigmaY);
        if (sigma < kMinSigma)
            continue;
        BlurAxisParams p{};
        p.srcPitch = result.width;
        p.dstPitch = result.width;
        p.width = result.width;
        p.height = result.height;
        p.dirX = axis == 0 ? 1 : 0;
        p.dirY = axis == 0 ? 0 : 1;
        FillWeights(p, sigma);
        id<MTLBuffer> next = acquireForBatch(batch, size_t(result.width) * size_t(result.height) * 16);
        scratch.push_back(next);
        Encode(encoder, mBlurAxis, {result.buffer, next}, p, result.width, result.height);
        if (result.owned)
            release(result.buffer, batch, scratch);
        result.buffer = next;
        result.owned = true;
    }
    return result;
}

id<MTLBuffer> FeedbackRenderer::encodeSpread(
    id<MTLComputeCommandEncoder> encoder,
    Pyramid& pyramid,
    const CanvasKey& canvas,
    float widthX,
    float widthY,
    StepBatch& batch,
    std::vector<id<MTLBuffer>>& scratch)
{
    const float width = std::max(widthX, widthY);
    // Nothing is worth spreading further than across the canvas.
    const float widest = std::max(float(std::max(canvas.width, canvas.height)) * 0.5f, 4.0f * width);

    // Each part blurred at its own size, gathered by size from the smallest.
    std::map<int, std::vector<std::pair<SmallBlur, float>>, std::greater<int>> bySize;
    for (const SpreadPart& part : SpreadParts(width, widest))
    {
        const float sigmaX = part.sigma * widthX, sigmaY = part.sigma * widthY;
        const SmallBlur small = std::max(sigmaX, sigmaY) >= kMinSigma
                                    ? encodeSmallBlur(encoder, pyramid, sigmaX, sigmaY, batch, scratch)
                                    : pyramidLevel(encoder, pyramid, 0, batch, scratch);
        bySize[small.factor].push_back({small, part.weight});
    }
    if (bySize.empty())
        return pyramid.levels[0].buffer;

    // At each size, the sum so far scaled up, plus the parts of that size; then the next size up.
    SmallBlur sum{};
    bool any = false;
    auto addInto = [&](const SmallBlur& into, const SmallBlur& from, float weight, bool first)
    {
        AddScaledParams p{from.width, into.width, from.width, from.height, from.factor / into.factor, weight, first ? 1 : 0};
        Encode(encoder, mAddScaled, {from.buffer, into.buffer}, p, into.width, into.height);
        if (from.owned)
            release(from.buffer, batch, scratch);
    };
    auto sized = [&](int factor)
    {
        SmallBlur s{nil, (canvas.width + factor - 1) / factor, (canvas.height + factor - 1) / factor, factor, true};
        s.buffer = acquireForBatch(batch, size_t(s.width) * size_t(s.height) * 16);
        scratch.push_back(s.buffer);
        return s;
    };
    for (const auto& [factor, parts] : bySize)
    {
        SmallBlur next = sized(factor);
        bool first = true;
        if (any)
        {
            addInto(next, sum, 1.0f, true);
            first = false;
        }
        for (const auto& [part, weight] : parts)
        {
            addInto(next, part, weight, first);
            first = false;
        }
        sum = next;
        any = true;
    }
    if (sum.factor > 1)
    {
        SmallBlur full = sized(1);
        addInto(full, sum, 1.0f, true);
        sum = full;
    }
    return sum.buffer;
}

id<MTLBuffer> FeedbackRenderer::step(
    StepBatch& batch,
    const CanvasKey& canvas,
    id<MTLBuffer> prev,
    const PlacedImage& input,
    const PlacedImage& map,
    const PlacedImage& blurMap,
    const FrameParams& f)
{
    const int width = canvas.width;
    const int height = canvas.height;
    const float scaleX = ScaleValue(canvas.downsampleX);
    const float scaleY = ScaleValue(canvas.downsampleY);

    std::vector<id<MTLBuffer>> scratch;
    auto acquire = [&]
    {
        id<MTLBuffer> buffer = acquireForBatch(batch, canvas.bytes());
        scratch.push_back(buffer);
        return buffer;
    };
    id<MTLBuffer> result = acquireForBatch(batch, canvas.bytes());

    if (!batch.commands)
        batch.commands = [batch.queue commandBuffer];
    id<MTLCommandBuffer> commands = batch.commands;
    id<MTLBuffer> current = prev;

    if (!current)
    {
        current = acquire();
        id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
        [blit fillBuffer:current range:NSMakeRange(0, canvas.bytes()) value:0];
        [blit endEncoding];
    }

    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];

    // Transform and displacement, in one resampling. Neutral settings skip it, so they're exactly a no-op.
    const float amountX = f.amountX * scaleX;
    const float amountY = f.amountY * scaleY;
    // None displaces nothing, and nor does a layer with nothing there.
    const bool mapMissing = f.displaceSource == MAP_NONE || (f.displaceSource == MAP_LAYER && !map.buffer);
    const bool displaces = (amountX != 0.0f || amountY != 0.0f) && !mapMissing;
    if (displaces || MovesTrail(f))
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
        p.inverse[0] = p.inverse[3] = 1.0f;
        if (MovesTrail(f))
            InvertTransform(f, canvas, &p);
        // No map: zero size, so nothing shifts.
        id<MTLBuffer> mapBuffer = displaces ? FillMap(p.map, map, f.displaceSource, current, width, height) : mEmpty;
        id<MTLBuffer> next = acquire();
        Encode(encoder, mDisplace, {current, mapBuffer, next}, p, width, height);
        current = next;
    }

    // Edges, found on the moved trail.
    if (f.edgeStrength > 0.0f)
    {
        EdgeParams p{};
        p.srcPitch = width;
        p.dstPitch = width;
        p.width = width;
        p.height = height;
        p.type = f.edgeType;
        p.blend = f.edgeBlend;
        p.distanceX = std::max(f.edgeSize * scaleX, 0.05f);
        p.distanceY = std::max(f.edgeSize * scaleY, 0.05f);
        p.strength = f.edgeStrength;
        id<MTLBuffer> next = acquire();
        Encode(encoder, mEdge, {current, next}, p, width, height);
        current = next;
    }

    // Blur
    const float sigmaX = BlurSigma(f.blurRadius) * scaleX;
    const float sigmaY = BlurSigma(f.blurRadius) * scaleY;
    // Every blur of this image shares one pyramid of it.
    Pyramid pyramid{{SmallBlur{current, width, height, 1}}};
    auto blurBy = [&](float share)
    {
        if (std::max(sigmaX, sigmaY) * share < kMinSigma)
            return current;
        return encodeSpread(encoder, pyramid, canvas, sigmaX * share, sigmaY * share, batch, scratch);
    };
    // A layer with nothing there blurs nothing, as black would.
    const bool blurMapMissing = f.blurSource == MAP_LAYER && !blurMap.buffer;
    if (std::max(sigmaX, sigmaY) >= kMinSigma && !blurMapMissing)
    {
        if (f.blurSource == MAP_NONE)
            current = blurBy(1.0f);
        else
        {
            BlurMapParams p{};
            p.pitch = width;
            p.width = width;
            p.height = height;
            id<MTLBuffer> mapBuffer = FillMap(p.map, blurMap, f.blurSource, current, width, height);
            id<MTLBuffer> next = acquire();
            Encode(encoder, mBlurByMap,
                   {current, blurBy(0.125f), blurBy(0.25f), blurBy(0.5f), blurBy(1.0f),
                    mapBuffer, next},
                   p, width, height);
            current = next;
        }
    }

    // Levels, HSL and Alpha, then composite with this frame's input. Neutral settings skip the maths,
    // so they're exactly a no-op.
    StepParams last{};
    GradeParams& grade = last.grade;
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
    last.composite = MakeCompositeParams(canvas, input, f.compositeMode);
    last.composite.splitsInput = f.splitsInput;
    last.composite.frameAlpha = f.frameAlpha;
    last.composite.stepAlpha = f.stepAlpha;
    Encode(encoder, mGradeAndComposite, {current, input.buffer ? input.buffer : mEmpty, result}, last, width, height);

    [encoder endEncoding];

    batch.freeBuffers.insert(batch.freeBuffers.end(), scratch.begin(), scratch.end());
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
    id<MTLCommandQueue> queue,
    const CanvasKey& canvas,
    id<MTLBuffer> image,
    const PlacedImage& output,
    const float* canvasFromOutput)
{
    CopyOutParams p{};
    p.srcPitch = canvas.width;
    p.dstPitch = output.pitch;
    p.offsetX = output.offsetX;
    p.offsetY = output.offsetY;
    p.canvasHeight = canvas.height;
    int width = std::min(output.width, canvas.width - output.offsetX);
    int height = std::min(output.height, canvas.height - output.offsetY);
    if (canvasFromOutput)
    {
        p.warped = 1;
        std::copy(canvasFromOutput, canvasFromOutput + 9, p.canvasFromOutput);
        width = output.width;
        height = output.height;
    }
    if (width <= 0 || height <= 0)
        return true;
    return RunOnce(queue, mCopyOut, {image, output.buffer}, p, width, height);
}
