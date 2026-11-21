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
const float kMaxDirectSigma = FEEDBACK_BLUR_MAX_RADIUS / kGaussianTails;
// Edges resample what they read anyway, so it only needs to be as fine as its blur.
const float kEdgeMaxDirectSigma = 1.0f;

// The buffers bind in order from index 0, followed by the parameters. A missing buffer, which couldn't
// be allocated, leaves the kernel out rather than have it read or write nothing, and returns false.
template <typename Params>
bool Encode(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers,
    const Params& params,
    int width,
    int height)
{
    for (id<MTLBuffer> buffer : buffers)
        if (!buffer)
            return false;
    NSUInteger index = 0;
    for (id<MTLBuffer> buffer : buffers)
        [encoder setBuffer:buffer offset:0 atIndex:index++];
    [encoder setBytes:&params length:sizeof(params) atIndex:index];
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    return true;
}

bool Succeeded(id<MTLCommandBuffer> commands)
{
    [commands commit];
    [commands waitUntilCompleted];
    return commands.status == MTLCommandBufferStatusCompleted;
}

// One kernel, queued in the batch.
template <typename Params>
bool EncodeInBatch(
    StepBatch& batch,
    id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers,
    const Params& params,
    int width,
    int height)
{
    if (!batch.commands)
        batch.commands = [batch.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [batch.commands computeCommandEncoder];
    const bool encoded = Encode(encoder, pipeline, buffers, params, width, height);
    [encoder endEncoding];
    return encoded;
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

    p.radius = std::min(int(std::ceil(sigma * kGaussianTails)), FEEDBACK_BLUR_MAX_RADIUS);
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

float SpreadReachIn(float width, float widest, double invisible)
{
    if (width <= 0.0f)
        return 0.0f;
    const std::vector<SpreadPart> parts = SpreadParts(width, widest);
    // What lies further than r from the centre, of each Gaussian in the plane, weighed.
    const auto beyond = [&](double r)
    {
        double t = 0.0;
        for (const SpreadPart& part : parts)
        {
            const double s = double(part.sigma) * double(width);
            t += double(part.weight) * std::exp(-r * r / (2.0 * s * s));
        }
        return t;
    };
    double lo = 0.0, hi = std::max(double(width), 1.0);
    while (beyond(hi) > invisible && hi < 1e7)
        hi *= 2.0;
    for (int i = 0; i < 40; i++)
    {
        const double mid = 0.5 * (lo + hi);
        (beyond(mid) > invisible ? lo : hi) = mid;
    }
    return float(hi);
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

// Describes a group's mask for a kernel, and returns the buffer to bind for it. None masks nothing, so
// the group applies in full; a layer with nothing there reads as black, so it doesn't apply at all.
id<MTLBuffer> FillMask(Mask& m, const PlacedImage& image, int source, id<MTLBuffer> trail, int width, int height,
                       id<MTLBuffer> empty)
{
    m.use = source != MAP_NONE;
    if (!m.use)
        return empty;
    id<MTLBuffer> buffer = FillMap(m.map, image, source, trail, width, height);
    return buffer ? buffer : empty;
}

// Where Transform moves a canvas pixel from, as `inverse * position + offset`.
void InvertTransform(const FrameParams& f, const CanvasKey& canvas, DisplaceParams* p)
{
    // Every pixel comes from far beyond the canvas, so nothing moved survives. A Transform mask's
    // partial move, part way to there, is beyond it too, as it would be at any scale near 0.
    if (CollapsesTrail(f))
    {
        std::fill(std::begin(p->inverse), std::end(p->inverse), 0.0f);
        p->offsetX = p->offsetY = -1e7f;
        return;
    }
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

// A missing input has zero size, so the kernel never reads it. `filter` and `opacity` are FrameParams'.
CompositeParams MakeCompositeParams(const CanvasKey& canvas, const PlacedImage& input, int mode, int filter, float opacity)
{
    CompositeParams p{};
    p.prevPitch = canvas.width;
    p.dstPitch = canvas.width;
    p.mode = mode;
    p.filter = filter;
    p.inputOpacity = opacity;
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

float SpreadReach(float width, float widest, double invisible)
{
    return SpreadReachIn(width, widest, invisible);
}


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
    id<MTLBuffer> buffer = mCache->acquireScratch(bytes);
    if (!buffer)
        batch.failed = true;
    return buffer;
}

id<MTLBuffer> FeedbackRenderer::copyPixels(StepBatch& batch, id<MTLBuffer> source, int pitch, int height)
{
    const size_t bytes = size_t(pitch) * size_t(height) * 16;
    if (!source || bytes == 0 || source.length < bytes)
        return nil;
    id<MTLBuffer> copy = [source.device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    if (!copy)
        return nil;
    if (!batch.commands)
        batch.commands = [batch.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [batch.commands blitCommandEncoder];
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:copy destinationOffset:0 size:bytes];
    [blit endEncoding];
    return copy;
}

bool FeedbackRenderer::flush(StepBatch& batch)
{
    bool ok = !batch.failed;
    batch.failed = false;
    if (batch.commands)
    {
        ok = Succeeded(batch.commands) && ok;
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
    float maxDirectSigma,
    bool repeatEdge,
    StepBatch& batch,
    std::vector<id<MTLBuffer>>& scratch)
{
    // Very wide blurs run on a smaller image, so their cost doesn't grow with the radius.
    int level = 0;
    while (std::max(sigmaX, sigmaY) / float(1 << level) > maxDirectSigma)
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
        p.repeatEdge = repeatEdge ? 1 : 0;
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

FeedbackRenderer::SmallBlur FeedbackRenderer::encodeSpread(
    id<MTLComputeCommandEncoder> encoder,
    Pyramid& pyramid,
    const CanvasKey& canvas,
    float widestGiven,
    float widthX,
    float widthY,
    bool fullSize,
    StepBatch& batch,
    std::vector<id<MTLBuffer>>& scratch)
{
    const float width = std::max(widthX, widthY);
    // Nothing is worth spreading further than across the canvas, or the comp the canvas lies in.
    const float across = widestGiven > 0.0f ? widestGiven : float(std::max(canvas.width, canvas.height)) * 0.5f;
    const float widest = std::max(across, 4.0f * width);

    // Each part blurred at its own size, gathered by size from the smallest.
    std::map<int, std::vector<std::pair<SmallBlur, float>>, std::greater<int>> bySize;
    for (const SpreadPart& part : SpreadParts(width, widest))
    {
        const float sigmaX = part.sigma * widthX, sigmaY = part.sigma * widthY;
        const SmallBlur small = std::max(sigmaX, sigmaY) >= kMinSigma
                                    ? encodeSmallBlur(encoder, pyramid, sigmaX, sigmaY, kMaxDirectSigma, false, batch, scratch)
                                    : pyramidLevel(encoder, pyramid, 0, batch, scratch);
        bySize[small.factor].push_back({small, part.weight});
    }
    if (bySize.empty())
        return pyramid.levels[0];

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
    if (sum.factor > 1 && fullSize)
    {
        SmallBlur full = sized(1);
        addInto(full, sum, 1.0f, true);
        sum = full;
    }
    return sum;
}

id<MTLBuffer> FeedbackRenderer::step(
    StepBatch& batch,
    const CanvasKey& canvas,
    id<MTLBuffer> prev,
    const PlacedImage& input,
    const PlacedImage (&maps)[SLOT_COUNT],
    const FrameParams& f)
{
    const PlacedImage& map = maps[SLOT_DISPLACE];
    const PlacedImage& blurMap = maps[SLOT_BLUR_MASK];
    const int displaceSource = f.sources[SLOT_DISPLACE], blurSource = f.sources[SLOT_BLUR_MASK];
    auto mask = [&](Mask& m, int slot, id<MTLBuffer> trail)
    { return FillMask(m, maps[slot], f.sources[slot], trail, canvas.width, canvas.height, mEmpty); };
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
        current = acquire();
    if (current != prev && current)
    {
        id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
        [blit fillBuffer:current range:NSMakeRange(0, canvas.bytes()) value:0];
        [blit endEncoding];
    }

    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];

    // Transform and displacement, in one resampling. Neutral settings skip it, so they're exactly a no-op.
    const float amountX = f.amountX * scaleX;
    const float amountY = f.amountY * scaleY;
    // None displaces nothing, and nor does a layer with nothing there.
    const bool mapMissing = displaceSource == MAP_NONE || (displaceSource == MAP_LAYER && !map.buffer);
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
        p.filter = f.filter;
        if (MovesTrail(f))
            InvertTransform(f, canvas, &p);
        // No map: zero size, so nothing shifts.
        id<MTLBuffer> mapBuffer = displaces ? FillMap(p.map, map, displaceSource, current, width, height) : mEmpty;
        id<MTLBuffer> transformMask = mask(p.transformMask, SLOT_TRANSFORM_MASK, current);
        id<MTLBuffer> displaceMask = mask(p.displaceMask, SLOT_DISPLACE_MASK, current);
        id<MTLBuffer> next = acquire();
        Encode(encoder, mDisplace, {current, mapBuffer, transformMask, displaceMask, next}, p, width, height);
        current = next;
    }

    // Edges, found on the moved trail.
    if (f.edgeStrength > 0.0f)
    {
        EdgeParams p{};
        p.srcPitch = width;
        p.dstPitch = width;
        p.type = f.edgeType;
        p.blend = f.edgeBlend;
        p.distanceX = std::max(f.edgeSize * scaleX, 0.05f);
        p.distanceY = std::max(f.edgeSize * scaleY, 0.05f);
        p.strength = f.edgeStrength;
        // Single pixels far apart alias whatever's finer than their spacing, and feedback builds that
        // up into a grid. Blurred to their spacing first, each sample stands for the area around it.
        Pyramid edgePyramid{{SmallBlur{current, width, height, 1}}};
        const SmallBlur smooth =
            encodeSmallBlur(encoder, edgePyramid, EdgeSmoothingSigma(p.distanceX, f.edgeSmoothing),
                            EdgeSmoothingSigma(p.distanceY, f.edgeSmoothing), kEdgeMaxDirectSigma, true, batch, scratch);
        p.smoothPitch = smooth.width;
        p.smoothWidth = smooth.width;
        p.smoothHeight = smooth.height;
        p.smoothFactor = smooth.factor;
        id<MTLBuffer> edgeMask = mask(p.mask, SLOT_EDGE_MASK, current);
        id<MTLBuffer> next = acquire();
        Encode(encoder, mEdge, {current, smooth.buffer, edgeMask, next}, p, width, height);
        if (smooth.owned)
            release(smooth.buffer, batch, scratch);
        // The halved images are free for the blur's own pyramid.
        for (size_t i = 1; i < edgePyramid.levels.size(); i++)
            release(edgePyramid.levels[i].buffer, batch, scratch);
        current = next;
    }

    // Blur
    const float sigmaX = BlurSigma(f.blurRadius) * scaleX;
    const float sigmaY = BlurSigma(f.blurRadius) * scaleY;
    // Every blur of this image shares one pyramid of it.
    Pyramid pyramid{{SmallBlur{current, width, height, 1}}};
    // Sub-steps and reduced resolution divide the blur into widths far under a pixel, which only add up
    // over many steps, so any blur at all goes ahead. The spread leaves out its parts too narrow to matter.
    auto blurBy = [&](float share, bool fullSize)
    {
        if (std::max(sigmaX, sigmaY) * share <= 0.0f)
            return SmallBlur{current, width, height, 1};
        return encodeSpread(encoder, pyramid, canvas, f.spreadWidest, sigmaX * share, sigmaY * share, fullSize, batch,
                            scratch);
    };
    // The trail as the grade reads it, which can be smaller than the canvas.
    SmallBlur graded{current, width, height, 1};
    // A layer with nothing there blurs nothing, as black would.
    const bool blurMapMissing = blurSource == MAP_LAYER && !blurMap.buffer;
    if (std::max(sigmaX, sigmaY) > 0.0f && !blurMapMissing)
    {
        if (blurSource == MAP_NONE)
        {
            // Only the grade reads the blurred trail, unless a mask is the trail itself, so it can stay at
            // the blur's own size, and save scaling it up first.
            const auto readsTrail = [&](int slot) { return f.sources[slot] == MAP_TRAIL; };
            graded = blurBy(1.0f, readsTrail(SLOT_LEVELS_MASK) || readsTrail(SLOT_HSL_MASK) || readsTrail(SLOT_ALPHA_MASK));
            current = graded.buffer;
        }
        else
        {
            BlurMapParams p{};
            p.pitch = width;
            p.width = width;
            p.height = height;
            id<MTLBuffer> mapBuffer = FillMap(p.map, blurMap, blurSource, current, width, height);
            id<MTLBuffer> next = acquire();
            Encode(encoder, mBlurByMap,
                   {current, blurBy(0.125f, true).buffer, blurBy(0.25f, true).buffer, blurBy(0.5f, true).buffer,
                    blurBy(1.0f, true).buffer, mapBuffer, next},
                   p, width, height);
            current = next;
            graded = SmallBlur{current, width, height, 1};
        }
    }

    // Levels, HSL and Alpha, then composite with this frame's input. Neutral settings skip the maths,
    // so they're exactly a no-op.
    StepParams last{};
    GradeParams& grade = last.grade;
    grade.srcPitch = graded.width;
    grade.srcWidth = graded.width;
    grade.srcHeight = graded.height;
    grade.srcFactor = graded.factor;
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
    last.composite = MakeCompositeParams(canvas, input, f.compositeMode, f.filter, f.inputOpacity);
    last.composite.splitsInput = f.splitsInput;
    last.composite.frameAlpha = f.frameAlpha;
    last.composite.stepAlpha = f.stepAlpha;
    id<MTLBuffer> levelsMask = mask(grade.levelsMask, SLOT_LEVELS_MASK, current);
    id<MTLBuffer> hslMask = mask(grade.hslMask, SLOT_HSL_MASK, current);
    id<MTLBuffer> alphaMask = mask(grade.alphaMask, SLOT_ALPHA_MASK, current);
    Encode(encoder, mGradeAndComposite,
           {current, input.buffer ? input.buffer : mEmpty, levelsMask, hslMask, alphaMask, result}, last, width, height);

    [encoder endEncoding];

    batch.freeBuffers.insert(batch.freeBuffers.end(), scratch.begin(), scratch.end());
    return result;
}

id<MTLBuffer> FeedbackRenderer::reframe(
    StepBatch& batch,
    const CanvasKey& from,
    const CanvasKey& to,
    id<MTLBuffer> image)
{
    id<MTLBuffer> result = acquireForBatch(batch, to.bytes());
    ReframeParams p{from.width, to.width, from.width, from.height, to.left - from.left, to.top - from.top};
    return EncodeInBatch(batch, mReframe, {image, result}, p, to.width, to.height) ? result : nil;
}

id<MTLBuffer> FeedbackRenderer::compositeInput(
    StepBatch& batch,
    const CanvasKey& canvas,
    id<MTLBuffer> image,
    const PlacedImage& input,
    int compositeMode,
    int filter,
    float opacity)
{
    id<MTLBuffer> result = acquireForBatch(batch, canvas.bytes());
    CompositeParams p = MakeCompositeParams(canvas, input, compositeMode, filter, opacity);
    return EncodeInBatch(batch, mComposite, {image, input.buffer ? input.buffer : mEmpty, result}, p, canvas.width,
                         canvas.height)
               ? result
               : nil;
}

bool FeedbackRenderer::copyOut(
    StepBatch& batch,
    const CanvasKey& canvas,
    id<MTLBuffer> image,
    const PlacedImage& output,
    const float* canvasFromOutput,
    int filter)
{
    CopyOutParams p{};
    p.filter = filter;
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
    return EncodeInBatch(batch, mCopyOut, {image, output.buffer}, p, width, height);
}
