// Feedback: an Echo-style effect that feeds a processed copy of its own output into each frame.
//
// AE may ask for frames in any order and caches them by hash, so each frame has to be a function
// of its inputs rather than of whatever rendered last. Frame N is therefore computed from a cached
// image of frame N-1, but only if AE confirms all input and parameter history up to N-1 is
// unchanged. Otherwise the loop rebuilds from blank, starting far enough back that anything older
// has faded to nothing. The same history is mixed into AE's frame GUID, so editing an earlier frame
// invalidates everything after it.
//
// The canvas follows the trail: each frame's covers everywhere its visible history can reach, and
// cached images are moved onto it. So a still layer costs nothing extra, and a small moving layer
// only pays for the area its trail actually occupies.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_GeneralPlug.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_EffectSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHelper.h"
#include "Param_Utils.h"

#include "FeedbackRenderer.h"
#include "ShaderTypes.h"

namespace
{
enum
{
    PARAM_INPUT = 0,
    PARAM_COMPOSITE,
    PARAM_ALPHA_START,
    PARAM_ALPHA_MAX,
    PARAM_ALPHA_MIN,
    PARAM_ALPHA_END,
    PARAM_BLUR_START,
    PARAM_BLUR,
    PARAM_BLUR_END,
    PARAM_LEVELS_START,
    PARAM_INPUT_BLACK,
    PARAM_INPUT_WHITE,
    PARAM_GAMMA,
    PARAM_OUTPUT_BLACK,
    PARAM_OUTPUT_WHITE,
    PARAM_LEVELS_END,
    PARAM_HSL_START,
    PARAM_HUE,
    PARAM_SATURATION,
    PARAM_LIGHTNESS,
    PARAM_HSL_END,
    PARAM_DISPLACE_START,
    PARAM_DISPLACE_LAYER,
    PARAM_CHANNEL_X,
    PARAM_AMOUNT_X,
    PARAM_CHANNEL_Y,
    PARAM_AMOUNT_Y,
    PARAM_DISPLACE_END,
    PARAM_SUBSTEPS,
    PARAM_REBUILD_LIMIT,
    PARAM_COUNT
};

// Saved projects find each control by this ID, so a control keeps its ID wherever it moves. In the
// order above. ID 17 belonged to a Source popup, and mustn't be reused.
const int kDiskIds[PARAM_COUNT] = {0, 1, 27, 2, 26, 28, 29, 3, 30, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25};

int DiskId(int param)
{
    return kDiskIds[param];
}

const int kMaxSubsteps = 15;

// Jumping to a frame replays no more than this many frames, so older trail is missing until playback
// passes through it. Playing forward carries the whole trail along in one image, so this doesn't
// shorten it. By default, Alpha Max 0.99 has all but faded out by the limit.
const int kDefaultRebuildLimit = 600;
const int kLongestRebuildLimit = 100000;

// Limits on the canvas, at full resolution for the spread and as actual pixels for the side. These
// only bite on extreme settings, such as large displacement, and trim what lies furthest out.
const float kMaxSpread = 1000.0f;
const A_long kMaxCanvasSide = 8192;
const int kCanvasSnapBits = 6;

// A rebuild starts where earlier frames have faded below this, about half an 8-bit level.
const double kInvisibleContribution = 0.002;

// A rebuild keeps its last few frames, so scrubbing or caching backwards can resume from them
// instead of rebuilding again.
const int kRebuildFramesKept = 8;

#ifdef FEEDBACK_LOG
void Log(const char* fmt, ...)
{
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    static FILE* f = fopen((std::string(getenv("HOME")) + "/Library/Logs/AEFeedback.log").c_str(), "a");
    if (!f)
        return;
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fputc('\n', f);
    fflush(f);
}
#else
void Log(const char*, ...) {}
#endif

// A moment in time, precise enough to fall between frames.
struct SampleTime
{
    A_long value;
    A_long step;
    A_u_long scale;
};

struct SubstepCheckout
{
    SampleTime time;
    A_long inputId;
    PF_LRect inputRect; // empty if there's no input to fetch
    A_long mapId;
    PF_LRect mapRect; // empty if there's no map to fetch
};

struct FrameCheckout
{
    A_long time;
    std::vector<SubstepCheckout> substeps; // oldest first; the last one is at the frame's own time
};

struct PreRenderData
{
    bool displaceFromLayer = false;
    CanvasKey canvas;
    PF_LRect outputRect;
    int samplesPerFrame; // the frame's own moment plus any sub-steps
    std::vector<FrameCheckout> frames; // [0] is the current frame, [k] is k frames earlier

    // The cached image to resume from, held here so it can't be evicted before the render. Only
    // frames newer than this one were checked out, so the render has no other way to rebuild it.
    id<MTLBuffer> resumeImage;
    CanvasKey resumeCanvas;
    int resumeK; // how many frames before the current one the image is, or -1 for no image
};

// What PreRender needs to know about a frame. Asking AE is slow when multiplied by a long trail and
// by the many PreRender calls AE makes per render, while the answers only change with the history.
struct SampleFacts
{
    SampleTime time;
    PF_LRect inputRect;
    PF_LRect mapRect;
};

struct FrameFacts
{
    float alpha, blurRadius, amountX, amountY;
    bool fromLayer;
    std::vector<SampleFacts> samples; // oldest first; the last one is at the frame's own time
};

// Remembered frame facts, trusted only while AE reports the same history as when they were read.
// Checking out a layer can run PreRender for another instance of this effect further up the stack,
// so a run stays alive and in place for whoever is using it while the list changes around it. AE
// lets one thread in at a time, so this needs a lock before enabling MFR.
class FrameFactsMemo
{
public:
    struct Key
    {
        A_long step;
        A_u_long scale;
        PF_RationalScale downsampleX, downsampleY;
        int samplesPerFrame;
        int variant; // see FindFactsAtReducedResolution

        bool operator==(const Key& o) const
        {
            return step == o.step && scale == o.scale && SameScale(downsampleX, o.downsampleX) &&
                   SameScale(downsampleY, o.downsampleY) && samplesPerFrame == o.samplesPerFrame && variant == o.variant;
        }
    };

    struct Run
    {
        Key key;
        A_long endExclusive = 0;
        PF_State history{};
        std::unordered_map<A_long, FrameFacts> frames;
    };

    std::vector<std::shared_ptr<Run>> runs; // most recently used first
};

const size_t kMaxFactRuns = 8;

struct MetalData
{
    std::unique_ptr<FeedbackRenderer> renderer;
    FrameFactsMemo memo;
};

void DisposePreRenderData(void* p)
{
    delete static_cast<PreRenderData*>(p);
}

NSString* MetallibPath()
{
    Dl_info info{};
    dladdr((const void*)&DisposePreRenderData, &info);
    NSString* exe = [NSString stringWithUTF8String:info.dli_fname];
    NSString* contents = [[exe stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];
    return [[contents stringByAppendingPathComponent:@"Resources"] stringByAppendingPathComponent:@"Feedback.metallib"];
}

inline bool IsEmpty(const PF_LRect& r)
{
    return r.left >= r.right || r.top >= r.bottom;
}

inline PF_LRect Intersect(const PF_LRect& a, const PF_LRect& b)
{
    PF_LRect r;
    r.left = std::max(a.left, b.left);
    r.top = std::max(a.top, b.top);
    r.right = std::min(a.right, b.right);
    r.bottom = std::min(a.bottom, b.bottom);
    return r;
}

// `substepsBefore` of `substeps` earlier than `frameTime`, expressed exactly. AE times are integer
// fractions, so the scale grows when the frame step doesn't divide evenly. Returns false if that
// would overflow AE's 32-bit times, which takes over an hour of timeline at the worst settings.
bool MakeSampleTime(PF_InData* in_data, A_long frameTime, int substepsBefore, int substeps, SampleTime* out)
{
    const A_long step = in_data->time_step;
    const long long multiple = substeps / std::gcd<long long, long long>(step, substeps);
    const long long subStep = step * multiple / substeps;
    const long long value = frameTime * multiple - substepsBefore * subStep;
    const long long scale = (long long)in_data->time_scale * multiple;
    if (value > INT_MAX || value < INT_MIN || scale > INT_MAX)
        return false;
    *out = SampleTime{A_long(value), A_long(subStep), A_u_long(scale)};
    return true;
}

// True if the history up to, but not including, `endExclusive` could be read.
// The state is zeroed either way.
bool ReadHistory(PF_InData* in_data, const PF_ParamUtilsSuite3& utils, A_long endExclusive, PF_State* state)
{
    AEFX_CLR_STRUCT(*state);
    if (endExclusive <= 0)
        return false;
    A_Time start = {0, static_cast<A_u_long>(in_data->time_scale)};
    A_Time duration = {endExclusive, static_cast<A_u_long>(in_data->time_scale)};
    return utils.PF_GetCurrentState(in_data->effect_ref, PF_ParamIndex_CHECK_ALL, &start, &duration, state) ==
           PF_Err_NONE;
}

struct ParamReader
{
    PF_InData* in_data;
    SampleTime time;
    PF_Err err = PF_Err_NONE;

    template <typename Get>
    auto read(int index, Get get)
    {
        PF_ParamDef def;
        AEFX_CLR_STRUCT(def);
        if (!err)
            err = PF_CHECKOUT_PARAM(in_data, index, time.value, time.step, time.scale, &def);
        auto value = get(def);
        PF_CHECKIN_PARAM(in_data, &def);
        return value;
    }

    float slider(int index)
    {
        return read(index, [](const PF_ParamDef& d) { return float(d.u.fs_d.value); });
    }

    int popup(int index)
    {
        return read(index, [](const PF_ParamDef& d) { return int(d.u.pd.value); });
    }

    float angleDegrees(int index)
    {
        return read(index, [](const PF_ParamDef& d) { return float(d.u.ad.value) / 65536.0f; });
    }

    int integer(int index)
    {
        return read(index, [](const PF_ParamDef& d) { return int(d.u.sd.value); });
    }
};

inline SampleTime FrameSample(PF_InData* in_data, A_long time)
{
    return SampleTime{time, in_data->time_step, in_data->time_scale};
}

// new alpha = Max * clamp((alpha * Max - Min) / (Max - Min), 0, 1), as a slope and offset. Opaque pixels
// land on Max, and anything that would land below Min after fading goes to nothing. With Min at 0,
// this is a plain multiply by Max.
void SetAlphaCurve(float maximum, float minimum, FrameParams* f)
{
    const float span = maximum - minimum;
    f->alphaSlope = span > 1e-5f ? maximum * maximum / span : 0.0f;
    f->alphaOffset = span > 1e-5f ? -maximum * minimum / span : (minimum > 0.0f ? -1.0f : 0.0f);
}

// The offset that, with the slope's nth root, gives the same result after n steps.
float RootOffset(float offset, float slope, float subSlope, float share)
{
    return std::abs(slope - 1.0f) < 1e-5f ? offset * share : offset * (subSlope - 1.0f) / (slope - 1.0f);
}

FrameParams ReadFrameParams(PF_InData* in_data, const SampleTime& time, PF_Err* err)
{
    ParamReader r{in_data, time};
    FrameParams f{};
    f.compositeMode = r.popup(PARAM_COMPOSITE);
    f.inputGain = 1.0f;
    f.frameAlpha = f.stepAlpha = 1.0f;
    f.alpha = r.slider(PARAM_ALPHA_MAX);
    SetAlphaCurve(f.alpha, r.slider(PARAM_ALPHA_MIN), &f);
    f.blurRadius = r.slider(PARAM_BLUR);
    f.inputBlack = r.slider(PARAM_INPUT_BLACK);
    f.inputWhite = r.slider(PARAM_INPUT_WHITE);
    f.gamma = r.slider(PARAM_GAMMA);
    f.outputBlack = r.slider(PARAM_OUTPUT_BLACK);
    f.outputWhite = r.slider(PARAM_OUTPUT_WHITE);
    f.hueDegrees = r.angleDegrees(PARAM_HUE);
    f.saturation = r.slider(PARAM_SATURATION) / 100.0f;
    f.lightness = r.slider(PARAM_LIGHTNESS) / 100.0f;
    f.channelX = r.popup(PARAM_CHANNEL_X);
    f.amountX = r.slider(PARAM_AMOUNT_X);
    f.channelY = r.popup(PARAM_CHANNEL_Y);
    f.amountY = r.slider(PARAM_AMOUNT_Y);
    *err = r.err;
    return f;
}

// Each control describes what happens per frame. With n sub-steps, each one applies an nth of that,
// so changing the sub-step count smooths the trail without changing its overall look.
FrameParams DivideIntoSubsteps(FrameParams f, int n)
{
    if (n <= 1)
        return f;
    const float share = 1.0f / float(n);

    const float frameAlpha = f.alpha;
    f.alpha = std::pow(f.alpha, share);
    if (f.alphaSlope > 0.0f)
    {
        const float subSlope = std::pow(f.alphaSlope, share);
        f.alphaOffset = RootOffset(f.alphaOffset, f.alphaSlope, subSlope, share);
        f.alphaSlope = subSlope;
    }
    // Blurs add by variance, so n small ones of radius r / sqrt(n) equal one of radius r.
    f.blurRadius *= std::sqrt(share);
    f.amountX *= share;
    f.amountY *= share;
    f.hueDegrees *= share;
    f.saturation = std::pow(f.saturation, share);
    f.lightness *= share;

    // Levels is a linear map plus a gamma. Exact roots of each stand in for the combination.
    const float range = std::max(f.inputWhite - f.inputBlack, 1e-5f);
    const float slope = (f.outputWhite - f.outputBlack) / range;
    const float offset = f.outputBlack - slope * f.inputBlack;
    const float subSlope = slope > 0.0f ? std::pow(slope, share) : 1.0f + (slope - 1.0f) * share;
    const float subOffset = RootOffset(offset, slope, subSlope, share);
    f.gamma = std::pow(f.gamma, share);
    f.inputBlack = 0.0f;
    f.inputWhite = 1.0f;
    f.outputBlack = subOffset;
    f.outputWhite = subSlope + subOffset;

    // In Add and Screen, the input builds up with every step rather than replacing what's there.
    // Each step adds only enough for still content to settle where one step per frame would, and
    // the current frame is laid over the finished trail at full strength when it's displayed.
    if (f.compositeMode == COMPOSITE_ADD)
    {
        f.inputGain = frameAlpha < 0.999f ? frameAlpha * (1.0f - f.alpha) / (1.0f - frameAlpha) : share;
    }
    else if (f.compositeMode == COMPOSITE_SCREEN)
    {
        // Just below 1, where the shader's formula is still well behaved.
        f.screenGain = 1;
        f.frameAlpha = std::min(frameAlpha, 0.999f);
        f.stepAlpha = std::pow(f.frameAlpha, share);
    }
    return f;
}

PF_Err GlobalSetup(PF_OutData* out_data)
{
    // Must match AE_Effect_Version in FeedbackPiPL.r (AE compares the encoded value, build included).
    out_data->my_version = PF_VERSION(1, 0, 0, PF_Stage_DEVELOP, 0);
    // Must match AE_Effect_Global_OutFlags in FeedbackPiPL.r.
    out_data->out_flags = PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_DEEP_COLOR_AWARE;
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |
                           PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT | PF_OutFlag2_I_MIX_GUID_DEPENDENCIES |
                           PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    return PF_Err_NONE;
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    auto slider = [&](const char* name, double min, double max, double sMin, double sMax, double dflt, int prec, int id)
    {
        AEFX_CLR_STRUCT(def);
        PF_ADD_FLOAT_SLIDERX(name, min, max, sMin, sMax, dflt, prec, PF_ValueDisplayFlag_NONE, 0, DiskId(id));
        return err;
    };
    auto topic = [&](const char* name, int id)
    {
        AEFX_CLR_STRUCT(def);
        PF_ADD_TOPIC(name, DiskId(id));
        return err;
    };
    auto endTopic = [&](int id)
    {
        AEFX_CLR_STRUCT(def);
        PF_END_TOPIC(DiskId(id));
        return err;
    };
    auto popup = [&](const char* name, int choices, int dflt, const char* items, int id)
    {
        AEFX_CLR_STRUCT(def);
        PF_ADD_POPUP(name, choices, dflt, items, DiskId(id));
        return err;
    };
    // These describe the whole loop rather than a moment in it, so they can't be keyframed.
    auto fixedInteger = [&](const char* name, int min, int max, int sliderMax, int dflt, int id)
    {
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        PF_ADD_SLIDER(name, min, max, min, sliderMax, dflt, DiskId(id));
        return err;
    };

    popup("Composite", 4, COMPOSITE_INPUT_IN_FRONT, "Input In Front|Input Behind|Add|Screen", PARAM_COMPOSITE);

    topic("Alpha", PARAM_ALPHA_START);
    slider("Alpha Max", 0, 1, 0, 1, 0.85, 3, PARAM_ALPHA_MAX);
    slider("Alpha Min", 0, 1, 0, 1, 0, 3, PARAM_ALPHA_MIN);
    endTopic(PARAM_ALPHA_END);

    topic("Blur", PARAM_BLUR_START);
    slider("Blur Radius", 0, 500, 0, 50, 0, 2, PARAM_BLUR);
    endTopic(PARAM_BLUR_END);

    topic("Levels", PARAM_LEVELS_START);
    slider("Input Black", -1, 2, 0, 1, 0, 3, PARAM_INPUT_BLACK);
    slider("Input White", -1, 2, 0, 1, 1, 3, PARAM_INPUT_WHITE);
    slider("Gamma", 0.1, 10, 0.5, 2, 1, 3, PARAM_GAMMA);
    slider("Output Black", -1, 2, 0, 1, 0, 3, PARAM_OUTPUT_BLACK);
    slider("Output White", -1, 2, 0, 1, 1, 3, PARAM_OUTPUT_WHITE);
    endTopic(PARAM_LEVELS_END);

    topic("HSL", PARAM_HSL_START);
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Hue Shift", 0, DiskId(PARAM_HUE));
    slider("Saturation", 0, 400, 0, 200, 100, 1, PARAM_SATURATION);
    slider("Lightness", -100, 100, -10, 10, 0, 2, PARAM_LIGHTNESS);
    endTopic(PARAM_HSL_END);

    topic("Displacement", PARAM_DISPLACE_START);
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER("Map Layer (None = Trail)", PF_LayerDefault_NONE, DiskId(PARAM_DISPLACE_LAYER));
    popup("Horizontal Channel", 5, MAP_CHANNEL_LUMA, "Red|Green|Blue|Alpha|Luma", PARAM_CHANNEL_X);
    slider("Max Horizontal", -4000, 4000, -100, 100, 0, 2, PARAM_AMOUNT_X);
    popup("Vertical Channel", 5, MAP_CHANNEL_LUMA, "Red|Green|Blue|Alpha|Luma", PARAM_CHANNEL_Y);
    slider("Max Vertical", -4000, 4000, -100, 100, 0, 2, PARAM_AMOUNT_Y);
    endTopic(PARAM_DISPLACE_END);

    fixedInteger("Sub-steps", 0, kMaxSubsteps, kMaxSubsteps, 0, PARAM_SUBSTEPS);
    fixedInteger("Trail Rebuild Limit (Frames)", 1, kLongestRebuildLimit, 3000, kDefaultRebuildLimit, PARAM_REBUILD_LIMIT);

    out_data->num_params = PARAM_COUNT;
    return err;
}

PF_Err GPUDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra)
{
    if (extra->input->what_gpu != PF_GPU_Framework_METAL)
        return PF_Err_NONE;

    PF_GPUDeviceInfo info;
    AEFX_CLR_STRUCT(info);
    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    PF_Err err = gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info);
    if (err)
        return err;

    @autoreleasepool
    {
        id<MTLDevice> device = (__bridge id<MTLDevice>)info.devicePV;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:MetallibPath()] error:&error];
        if (!library)
        {
            Log("metallib load failed: %s", error.localizedDescription.UTF8String);
            return PF_Err_INTERNAL_STRUCT_DAMAGED;
        }
        auto renderer = FeedbackRenderer::create(device, library);
        if (!renderer)
        {
            Log("pipeline creation failed");
            return PF_Err_INTERNAL_STRUCT_DAMAGED;
        }
        auto* data = new MetalData();
        data->renderer = std::move(renderer);
        extra->output->gpu_data = data;
    }
    out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    return PF_Err_NONE;
}

PF_Err GPUDeviceSetdown(PF_GPUDeviceSetdownExtra* extra)
{
    @autoreleasepool
    {
        delete static_cast<MetalData*>(extra->input->gpu_data);
    }
    return PF_Err_NONE;
}

inline PF_LRect Union(const PF_LRect& a, const PF_LRect& b)
{
    PF_LRect r;
    r.left = std::min(a.left, b.left);
    r.top = std::min(a.top, b.top);
    r.right = std::max(a.right, b.right);
    r.bottom = std::max(a.bottom, b.bottom);
    return r;
}

// Keeps `side` pixels of [lo, hi], staying as close to `focus` as possible.
inline void TrimAxis(A_long* lo, A_long* hi, A_long focus, A_long side)
{
    if (*hi - *lo <= side)
        return;
    *lo = std::clamp(focus - side / 2, *lo, *hi - side);
    *hi = *lo + side;
}

PF_Err CheckoutLayer(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    PF_RenderRequest request,
    const PF_LRect& rect,
    int param,
    const SampleTime& time,
    A_long id,
    PF_CheckoutResult* result)
{
    AEFX_CLR_STRUCT(*result);
    request.rect = rect;
    return extra->cb->checkout_layer(
        in_data->effect_ref, param, id, &request, time.value, time.step, time.scale, result);
}

// Everything a layer can show at a moment, however little AE wants from this effect. A shape
// layer's bounds follow its contents, so this differs from moment to moment. An empty request
// returns them without rendering anything.
PF_Err LayerBounds(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    int param,
    const SampleTime& time,
    A_long* nextId,
    PF_LRect* bounds)
{
    PF_CheckoutResult result;
    PF_Err err = CheckoutLayer(in_data, extra, request, PF_LRect{0, 0, 0, 0}, param, time, (*nextId)++, &result);
    *bounds = result.max_result_rect;
    return err;
}

// Requests pixels that the render can fetch. Nothing renders unless it actually does.
PF_Err CheckoutWholeLayer(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    int param,
    const SampleTime& time,
    A_long id,
    PF_LRect* rect)
{
    // Far bigger than any layer, so this is the whole layer whatever its bounds are in this call.
    const A_long anyLayer = 1000000;
    PF_CheckoutResult result;
    PF_Err err = CheckoutLayer(
        in_data, extra, request, PF_LRect{-anyLayer, -anyLayer, anyLayer, anyLayer}, param, time, id, &result);
    *rect = result.result_rect;
    return err;
}

// Finds the image for `time`, if one exists whose history is still current.
id<MTLBuffer> FindCached(
    PF_InData* in_data,
    const PF_ParamUtilsSuite3& utils,
    FeedbackCache& cache,
    const CanvasKey& canvas,
    A_long time,
    A_long step,
    CanvasKey* imageCanvas)
{
    const auto candidates = cache.candidates(canvas, time);
    PF_State history;
    if (candidates.empty() || !ReadHistory(in_data, utils, time + step, &history))
        return nil;
    for (const auto& candidate : candidates)
    {
        A_Boolean same = 0;
        utils.PF_AreStatesIdentical(in_data->effect_ref, &candidate.history, &history, &same);
        if (same)
        {
            cache.touch(candidate.buffer);
            *imageCanvas = candidate.key;
            return candidate.buffer;
        }
    }
    return nil;
}

// Finds the remembered facts for this history, making room for a new set if there are none.
std::shared_ptr<FrameFactsMemo::Run> FindFacts(
    PF_InData* in_data, const PF_ParamUtilsSuite3& utils, FrameFactsMemo& memo, const FrameFactsMemo::Key& key)
{
    for (size_t i = 0; i < memo.runs.size(); i++)
    {
        const std::shared_ptr<FrameFactsMemo::Run> run = memo.runs[i];
        if (!(run->key == key))
            continue;
        PF_State history;
        A_Boolean same = 0;
        if (ReadHistory(in_data, utils, run->endExclusive, &history))
            utils.PF_AreStatesIdentical(in_data->effect_ref, &run->history, &history, &same);
        if (same)
        {
            std::rotate(memo.runs.begin(), memo.runs.begin() + i, memo.runs.begin() + i + 1);
            return run;
        }
    }
    auto run = std::make_shared<FrameFactsMemo::Run>();
    run->key = key;
    memo.runs.insert(memo.runs.begin(), run);
    if (memo.runs.size() > kMaxFactRuns)
        memo.runs.pop_back();
    return run;
}

// At reduced resolution, some PreRender calls report a shape layer's bounds at full scale and others
// scaled, though nothing else about the calls differs. The bounds of the two can't share a memo, so
// each gets a run of its own, and asking AE about one frame already in a run shows if it's this kind.
std::shared_ptr<FrameFactsMemo::Run> FindFactsAtReducedResolution(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    const PF_ParamUtilsSuite3& utils,
    FrameFactsMemo& memo,
    FrameFactsMemo::Key key,
    A_long* nextId,
    PF_Err* err)
{
    std::shared_ptr<FrameFactsMemo::Run> run;
    for (key.variant = 0; key.variant < 2; key.variant++)
    {
        run = FindFacts(in_data, utils, memo, key);
        const SampleFacts* known = nullptr;
        for (const auto& frame : run->frames)
        {
            for (const SampleFacts& sample : frame.second.samples)
                if (!IsEmpty(sample.inputRect))
                    known = &sample;
            if (known)
                break;
        }
        if (!known)
            return run;

        PF_LRect now{};
        *err = LayerBounds(in_data, extra, request, PARAM_INPUT, known->time, nextId, &now);
        if (*err)
            return run;
        const PF_LRect& was = known->inputRect;
        if (now.left == was.left && now.top == was.top && now.right == was.right && now.bottom == was.bottom)
            return run;
    }
    // Neither matches, so the layer has changed in a way the history doesn't show. Start again.
    run->frames.clear();
    run->endExclusive = 0;
    return run;
}

// Asks AE about one frame in the history.
PF_Err ReadFrameFacts(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    A_long time,
    int samplesPerFrame,
    bool fromLayer,
    A_long* nextId,
    FrameFacts* facts)
{
    ParamReader r{in_data, FrameSample(in_data, time)};
    facts->alpha = r.slider(PARAM_ALPHA_MAX);
    facts->blurRadius = r.slider(PARAM_BLUR);
    facts->amountX = r.slider(PARAM_AMOUNT_X);
    facts->amountY = r.slider(PARAM_AMOUNT_Y);
    facts->fromLayer = fromLayer;
    PF_Err err = r.err;

    for (int before = samplesPerFrame - 1; !err && before >= 0; before--)
    {
        SampleFacts sample{};
        if (!MakeSampleTime(in_data, time, before, samplesPerFrame, &sample.time))
            sample.time = FrameSample(in_data, time - before * in_data->time_step / samplesPerFrame);
        ERR(LayerBounds(in_data, extra, request, PARAM_INPUT, sample.time, nextId, &sample.inputRect));
        if (!err && facts->fromLayer)
            ERR(LayerBounds(in_data, extra, request, PARAM_DISPLACE_LAYER, sample.time, nextId, &sample.mapRect));
        facts->samples.push_back(sample);
    }
    return err;
}

// Whether the Map Layer control has a layer picked. Checking out the layer can't tell None apart from
// a layer that's empty at the moment, which should shift nothing. If this can't be read, it reports a
// layer, because a missing map displaces nothing, which is the safer way to be wrong.
bool MapLayerIsPicked(PF_InData* in_data, PF_OutData* out_data)
{
    A_long picked = 1;
    try
    {
        AEFX_SuiteScoper<AEGP_UtilitySuite6> util(in_data, kAEGPUtilitySuite, kAEGPUtilitySuiteVersion6, out_data);
        AEFX_SuiteScoper<AEGP_PFInterfaceSuite1> face(in_data, kAEGPPFInterfaceSuite, kAEGPPFInterfaceSuiteVersion1, out_data);
        AEFX_SuiteScoper<AEGP_EffectSuite5> effects(in_data, kAEGPEffectSuite, kAEGPEffectSuiteVersion5, out_data);
        AEFX_SuiteScoper<AEGP_StreamSuite7> streams(in_data, kAEGPStreamSuite, kAEGPStreamSuiteVersion7, out_data);

        static AEGP_PluginID pluginId = 0;
        if (!pluginId && util->AEGP_RegisterWithAEGP(nullptr, "Feedback", &pluginId))
            return true;

        AEGP_EffectRefH effect = nullptr;
        if (face->AEGP_GetNewEffectForEffect(pluginId, in_data->effect_ref, &effect))
            return true;
        AEGP_StreamRefH stream = nullptr;
        if (!streams->AEGP_GetNewEffectStreamByIndex(pluginId, effect, PARAM_DISPLACE_LAYER, &stream))
        {
            AEGP_StreamValue2 value{};
            A_Time time = {in_data->current_time, static_cast<A_u_long>(in_data->time_scale)};
            if (!streams->AEGP_GetNewStreamValue(pluginId, stream, AEGP_LTimeMode_CompTime, &time, TRUE, &value))
            {
                picked = value.val.layer_id;
                streams->AEGP_DisposeStreamValue(&value);
            }
            streams->AEGP_DisposeStream(stream);
        }
        effects->AEGP_DisposeEffect(effect);
    }
    catch (...)
    {
        return true;
    }
    return picked != AEGP_LayerIDVal_NONE;
}

PF_Err PreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    auto preRenderStart = std::chrono::steady_clock::now();

    const A_long now = in_data->current_time;
    const A_long step = in_data->time_step;

    AEFX_SuiteScoper<PF_ParamUtilsSuite3> utilsSuite(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);
    const PF_ParamUtilsSuite3& utils = *utilsSuite.get();

    // Every earlier frame affects this one, so AE's cached copy is only valid if they're unchanged.
    {
        PF_State history;
        // AE requires this call on every PreRender. With no earlier frames, the zeroed state goes in.
        ReadHistory(in_data, utils, now, &history);
        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(history), &history));

        // AE's disk cache outlives the plugin, so frames saved by a different build mustn't be reused.
        static const char kBuild[] = __DATE__ " " __TIME__;
        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(kBuild), kBuild));
    }

    PF_RenderRequest request = extra->input->output_request;
    request.channel_mask = PF_ChannelMask_ARGB;

    int samplesPerFrame = 1;
    int rebuildLimit = kDefaultRebuildLimit;
    {
        ParamReader r{in_data, FrameSample(in_data, now)};
        samplesPerFrame = r.integer(PARAM_SUBSTEPS) + 1;
        rebuildLimit = r.integer(PARAM_REBUILD_LIMIT);
        ERR(r.err);
    }

    const float scaleX = ScaleValue(in_data->downsample_x);
    const float scaleY = ScaleValue(in_data->downsample_y);
    const A_long framesBefore = step > 0 ? std::max<A_long>(now / step, 0) : 0;

    auto data = std::make_unique<PreRenderData>();
    data->samplesPerFrame = samplesPerFrame;
    data->displaceFromLayer = MapLayerIsPicked(in_data, out_data);
    data->resumeK = -1;

    auto* metal = const_cast<MetalData*>(static_cast<const MetalData*>(extra->input->gpu_data));

    // Without a GPU there's nothing to remember facts in, or to resume from.
    FrameFactsMemo localMemo;
    FrameFactsMemo& memo = metal ? metal->memo : localMemo;
    FrameFactsMemo::Key factsKey{
        step,
        static_cast<A_u_long>(in_data->time_scale),
        in_data->downsample_x,
        in_data->downsample_y,
        samplesPerFrame,
        0};
    A_long nextId = 0;
    std::shared_ptr<FrameFactsMemo::Run> run;
    if (!err && (in_data->downsample_x.den > 1 || in_data->downsample_y.den > 1))
        run = FindFactsAtReducedResolution(in_data, extra, request, utils, memo, factsKey, &nextId, &err);
    else if (!err)
        run = FindFacts(in_data, utils, memo, factsKey);

    // Walk back until earlier frames have faded to nothing or the layer begins. The canvas is
    // whatever this reaches: each input's bounds, plus how far later frames can have spread it. It
    // goes past the rebuild limit, because playing forward can still be carrying what's older.
    int framesRead = 0;
    std::vector<const FrameFacts*> window;
    bool hasArea = false;
    PF_LRect area{};
    A_long focusX = 0, focusY = 0;
    double remaining = 1.0;
    float varianceX = 0.0f, varianceY = 0.0f; // blurs add by variance
    float shiftX = 0.0f, shiftY = 0.0f;
    bool oldest = false;

    for (int k = 0; !err && !oldest; k++)
    {
        const A_long time = now - k * step;
        auto found = run->frames.find(time);
        if (found == run->frames.end())
        {
            FrameFacts read{};
            ERR(ReadFrameFacts(in_data, extra, request, time, samplesPerFrame, data->displaceFromLayer, &nextId, &read));
            if (err)
                break;
            found = run->frames.emplace(time, std::move(read)).first;
            framesRead++;
        }
        const FrameFacts& facts = found->second;
        if (k <= rebuildLimit)
            window.push_back(&facts);

        remaining *= facts.alpha;
        oldest = remaining < kInvisibleContribution || k >= framesBefore;

        // The oldest frame in the window starts from blank, so only its final moment is used.
        const size_t firstSample = oldest ? facts.samples.size() - 1 : 0;
        bool hasFrameArea = false, hasMap = false;
        PF_LRect frameArea{};
        for (size_t i = firstSample; i < facts.samples.size(); i++)
        {
            const SampleFacts& sample = facts.samples[i];
            if (!IsEmpty(sample.inputRect))
            {
                frameArea = hasFrameArea ? Union(frameArea, sample.inputRect) : sample.inputRect;
                hasFrameArea = true;
            }
            hasMap = hasMap || (facts.fromLayer && !IsEmpty(sample.mapRect));
        }

        const float sigmaX = BlurSigma(facts.blurRadius) * scaleX;
        const float sigmaY = BlurSigma(facts.blurRadius) * scaleY;
        varianceX += sigmaX * sigmaX;
        varianceY += sigmaY * sigmaY;
        if (!facts.fromLayer || hasMap)
        {
            shiftX += std::abs(facts.amountX) * scaleX;
            shiftY += std::abs(facts.amountY) * scaleY;
        }

        if (hasFrameArea)
        {
            // Three sigma of blur, plus a pixel for displacement's filtering.
            const float limitX = kMaxSpread * scaleX, limitY = kMaxSpread * scaleY;
            const A_long padX = A_long(std::ceil(std::min(3.0f * std::sqrt(varianceX) + shiftX, limitX))) + 1;
            const A_long padY = A_long(std::ceil(std::min(3.0f * std::sqrt(varianceY) + shiftY, limitY))) + 1;
            frameArea.left -= padX;
            frameArea.top -= padY;
            frameArea.right += padX;
            frameArea.bottom += padY;
            if (!hasArea)
            {
                focusX = (frameArea.left + frameArea.right) / 2;
                focusY = (frameArea.top + frameArea.bottom) / 2;
            }
            area = hasArea ? Union(area, frameArea) : frameArea;
            hasArea = true;
        }
    }

    // The history now vouches for everything read so far. If it can't be read, keep nothing.
    if (!err && (framesRead > 0 || now + step > run->endExclusive))
    {
        run->endExclusive = std::max(run->endExclusive, now + step);
        if (!ReadHistory(in_data, utils, run->endExclusive, &run->history))
            run->endExclusive = 0; // never matches, so the next call starts again
    }

    if (!hasArea)
        area = extra->input->output_request.rect;

    // Rounding out means a slowly moving trail reuses the same canvas for many frames.
    area.left = (area.left >> kCanvasSnapBits) << kCanvasSnapBits;
    area.top = (area.top >> kCanvasSnapBits) << kCanvasSnapBits;
    area.right = ((area.right + (1 << kCanvasSnapBits) - 1) >> kCanvasSnapBits) << kCanvasSnapBits;
    area.bottom = ((area.bottom + (1 << kCanvasSnapBits) - 1) >> kCanvasSnapBits) << kCanvasSnapBits;
    TrimAxis(&area.left, &area.right, focusX, kMaxCanvasSide);
    TrimAxis(&area.top, &area.bottom, focusY, kMaxCanvasSide);

    extra->output->max_result_rect = area;
    extra->output->result_rect = Intersect(extra->input->output_request.rect, area);

    data->outputRect = extra->output->result_rect;
    data->canvas.left = area.left;
    data->canvas.top = area.top;
    data->canvas.width = area.right - area.left;
    data->canvas.height = area.bottom - area.top;
    data->canvas.downsampleX = in_data->downsample_x;
    data->canvas.downsampleY = in_data->downsample_y;

    // Only frames newer than the newest usable image need their pixels.
    int checkoutFrames = int(window.size());
    if (metal && !err)
    {
        FeedbackCache& cache = metal->renderer->cache();
        for (int k = 0; k < int(window.size()); k++)
        {
            id<MTLBuffer> image = FindCached(in_data, utils, cache, data->canvas, now - k * step, step, &data->resumeCanvas);
            if (image)
            {
                data->resumeImage = image;
                data->resumeK = k;
                checkoutFrames = std::max(k, 1); // AE won't give out the output without the current input
                break;
            }
        }
    }

    for (int k = 0; !err && k < checkoutFrames; k++)
    {
        const FrameFacts& facts = *window[k];
        FrameCheckout frame{};
        frame.time = now - k * step;

        // Beyond what's computed, the current input is all that's asked for. A rebuild's oldest frame
        // starts from blank, so only its final moment is used.
        const bool finalMomentOnly = data->resumeK >= 0 ? k >= data->resumeK : k == int(window.size()) - 1;
        const size_t firstSample = finalMomentOnly ? facts.samples.size() - 1 : 0;

        for (size_t i = firstSample; !err && i < facts.samples.size(); i++)
        {
            const SampleFacts& sample = facts.samples[i];
            SubstepCheckout sub{};
            sub.time = sample.time;
            // AE only hands out the output after an input has been checked out, so the current
            // frame's always is, even when the layer is empty.
            const bool isCurrent = k == 0 && i + 1 == facts.samples.size();
            if (!IsEmpty(sample.inputRect) || isCurrent)
            {
                sub.inputId = nextId++;
                ERR(CheckoutWholeLayer(
                    in_data, extra, request, PARAM_INPUT, sample.time, sub.inputId, &sub.inputRect));
            }
            if (!err && facts.fromLayer && !IsEmpty(sample.mapRect))
            {
                sub.mapId = nextId++;
                ERR(CheckoutWholeLayer(
                    in_data, extra, request, PARAM_DISPLACE_LAYER, sample.time, sub.mapId, &sub.mapRect));
            }
            frame.substeps.push_back(sub);
        }
        data->frames.push_back(std::move(frame));
    }

    Log("prerender frame=%d window=%d samples=%d read=%d resume=%d checkouts=%d canvas=%dx%d ms=%.2f",
        step > 0 ? now / step : 0,
        int(window.size()) - 1,
        samplesPerFrame,
        framesRead,
        data->resumeK,
        nextId,
        data->canvas.width,
        data->canvas.height,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - preRenderStart).count());

    if (!err)
    {
        extra->output->pre_render_data = data.release();
        extra->output->delete_pre_render_data_func = DisposePreRenderData;
    }
    return err;
}

PF_Err RenderGPU(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    auto* data = static_cast<PreRenderData*>(extra->input->pre_render_data);
    if (!data || data->frames.empty())
        return PF_Err_NONE;

    auto* metal = const_cast<MetalData*>(static_cast<const MetalData*>(extra->input->gpu_data));
    if (!metal)
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    FeedbackRenderer& renderer = *metal->renderer;
    FeedbackCache& cache = renderer.cache();

    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    AEFX_SuiteScoper<PF_ParamUtilsSuite3> utilsSuite(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);
    const PF_ParamUtilsSuite3& utils = *utilsSuite.get();
    AEFX_SuiteScoper<PF_WorldSuite2> worlds(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

    PF_GPUDeviceInfo info;
    AEFX_CLR_STRUCT(info);
    ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info));
    if (err)
        return err;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)info.command_queuePV;

    const CanvasKey& canvas = data->canvas;
    const A_long step = in_data->time_step;
    const int frameCount = int(data->frames.size());

    auto startTime = std::chrono::steady_clock::now();
    int framesComputed = 0;
    int fetched = 0;
    double fetchMs = 0.0, gpuMs = 0.0;
    auto msSince = [](std::chrono::steady_clock::time_point t)
    { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); };

    // Reads a checked-out world as a placed image. A missing world means nothing there, not failure.
    auto place = [&](PF_EffectWorld* world, const PF_LRect& rect, PlacedImage* image) -> PF_Err
    {
        *image = PlacedImage{};
        if (!world)
            return PF_Err_NONE;
        PF_PixelFormat format = PF_PixelFormat_INVALID;
        PF_Err e = worlds->PF_GetPixelFormat(world, &format);
        if (!e && format != PF_PixelFormat_GPU_BGRA128)
            e = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        void* memory = nullptr;
        if (!e)
            e = gpu->GetGPUWorldData(in_data->effect_ref, world, &memory);
        if (!e)
        {
            image->buffer = (__bridge id<MTLBuffer>)memory;
            image->pitch = world->rowbytes / 16;
            image->offsetX = rect.left - canvas.left;
            image->offsetY = rect.top - canvas.top;
            image->width = world->width;
            image->height = world->height;
        }
        return e;
    };

    @autoreleasepool
    {
        // AE only hands out the output after an input has been checked out, even an empty one.
        const SubstepCheckout& currentSub = data->frames[0].substeps.back();
        PF_EffectWorld* currentWorld = nullptr;
        PF_EffectWorld* output = nullptr;
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, currentSub.inputId, &currentWorld));
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
        PF_EffectWorld* currentInput = !IsEmpty(currentSub.inputRect) ? currentWorld : nullptr;

        // Newest usable starting point within the window.
        id<MTLBuffer> image = nil;
        CanvasKey imageCanvas = canvas;
        int resumeK = frameCount; // the first frame to compute is resumeK - 1
        for (int k = 0; !err && k < frameCount; k++)
        {
            image = FindCached(in_data, utils, cache, canvas, data->frames[k].time, step, &imageCanvas);
            if (image)
            {
                resumeK = k;
                break;
            }
        }
        if (!image && data->resumeImage)
        {
            image = data->resumeImage;
            imageCanvas = data->resumeCanvas;
            resumeK = data->resumeK;
        }
        framesComputed = resumeK;

        // Set while `image` is a scratch buffer of ours rather than one the cache owns.
        bool imageIsScratch = false;

        if (image && !(imageCanvas == canvas))
        {
            image = renderer.reframe(queue, imageCanvas, canvas, image);
            imageIsScratch = image != nil;
            if (!image)
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }

        for (int k = resumeK - 1; !err && k >= 0; k--)
        {
            const FrameCheckout& frame = data->frames[k];
            const int subCount = int(frame.substeps.size());

            for (int s = 0; !err && s < subCount; s++)
            {
                // Command buffers are autoreleased and hold on to what they used, so a long rebuild
                // mustn't keep them all until it ends.
                @autoreleasepool
                {
                    const SubstepCheckout& sub = frame.substeps[s];
                    const bool isCurrentSample = k == 0 && s == subCount - 1;

                    // Until there's an image or an input, every step would only produce blank again.
                    if (!image && !isCurrentSample && IsEmpty(sub.inputRect))
                        continue;

                    if (!isCurrentSample)
                        ERR(PF_ABORT(in_data));

                    FrameParams params{};
                    if (!err)
                    {
                        params = DivideIntoSubsteps(ReadFrameParams(in_data, sub.time, &err), data->samplesPerFrame);
                        params.displaceFromLayer = data->displaceFromLayer;
                    }

                    auto fetchStart = std::chrono::steady_clock::now();
                    PF_EffectWorld* inputWorld = isCurrentSample ? currentInput : nullptr;
                    if (!err && !isCurrentSample && !IsEmpty(sub.inputRect))
                    {
                        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, sub.inputId, &inputWorld));
                        fetched++;
                    }
                    PF_EffectWorld* mapWorld = nullptr;
                    if (!err && !IsEmpty(sub.mapRect))
                        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, sub.mapId, &mapWorld));
                    fetchMs += msSince(fetchStart);

                    PlacedImage input{}, map{};
                    if (!err)
                        ERR(place(inputWorld, sub.inputRect, &input));
                    if (!err)
                        ERR(place(mapWorld, sub.mapRect, &map));

                    id<MTLBuffer> next = nil;
                    if (!err)
                    {
                        auto gpuStart = std::chrono::steady_clock::now();
                        next = renderer.step(queue, canvas, image, input, map, params);
                        gpuMs += msSince(gpuStart);
                        if (!next)
                            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                    }

                    if (!isCurrentSample && inputWorld)
                        extra->cb->checkin_layer_pixels(in_data->effect_ref, sub.inputId);
                    if (mapWorld)
                        extra->cb->checkin_layer_pixels(in_data->effect_ref, sub.mapId);

                    if (next)
                    {
                        if (imageIsScratch)
                            cache.releaseScratch(image);
                        image = next;
                        imageIsScratch = true;
                    }
                }
            }

            if (err)
                break;

            // Keep only the frames most likely to be asked for next; the rest of a long rebuild
            // would otherwise push useful images out of the cache.
            PF_State history;
            if (image && k < kRebuildFramesKept && ReadHistory(in_data, utils, frame.time + step, &history))
            {
                cache.publish(canvas, frame.time, history, image);
                imageIsScratch = false;
            }
        }

        if (!err && image && output)
        {
            // With sub-steps, Add and Screen store only the trail, so the current frame goes on top
            // at full strength here. It stays sharp, and isn't fed back in twice.
            id<MTLBuffer> shown = image;
            bool shownIsScratch = false;
            if (data->samplesPerFrame > 1 && currentInput)
            {
                ParamReader r{in_data, FrameSample(in_data, in_data->current_time)};
                const int mode = r.popup(PARAM_COMPOSITE);
                ERR(r.err);
                if (!err && (mode == COMPOSITE_ADD || mode == COMPOSITE_SCREEN))
                {
                    PlacedImage input{};
                    ERR(place(currentInput, currentSub.inputRect, &input));
                    if (!err)
                    {
                        shown = renderer.compositeInput(queue, canvas, image, input, mode);
                        shownIsScratch = shown != nil;
                        if (!shown)
                            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                    }
                }
            }

            PlacedImage placedOutput{};
            ERR(place(output, data->outputRect, &placedOutput));
            if (!err && !renderer.copyOut(queue, canvas, shown, placedOutput))
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            if (shownIsScratch)
                renderer.cache().releaseScratch(shown);
        }

        if (imageIsScratch)
            cache.releaseScratch(image);
        extra->cb->checkin_layer_pixels(in_data->effect_ref, currentSub.inputId);
    }

    Log("frame=%d computed=%d samples=%d fetched=%d fetchMs=%.1f gpuMs=%.1f %dx%d out=%dx%d "
        "took=%.1fms err=%d",
        step > 0 ? in_data->current_time / step : 0,
        framesComputed,
        data->samplesPerFrame,
        fetched,
        fetchMs,
        gpuMs,
        canvas.width,
        canvas.height,
        data->outputRect.right - data->outputRect.left,
        data->outputRect.bottom - data->outputRect.top,
        msSince(startTime),
        err);
    return err;
}

// Without a GPU the effect can't run, so the layer passes through unchanged.
PF_Err RenderCPU(PF_InData* in_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    auto* data = static_cast<PreRenderData*>(extra->input->pre_render_data);
    if (!data || data->frames.empty())
        return PF_Err_NONE;
    const SubstepCheckout& current = data->frames[0].substeps.back();
    if (IsEmpty(current.inputRect))
        return PF_Err_NONE;

    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, current.inputId, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (!err && input && output)
    {
        // The output can be larger than the layer now, so the copy may only cover part of it.
        const PF_LRect overlap = Intersect(current.inputRect, data->outputRect);
        if (!IsEmpty(overlap))
        {
            PF_Rect source, dest;
            source.left = overlap.left - current.inputRect.left;
            source.top = overlap.top - current.inputRect.top;
            source.right = overlap.right - current.inputRect.left;
            source.bottom = overlap.bottom - current.inputRect.top;
            dest.left = overlap.left - data->outputRect.left;
            dest.top = overlap.top - data->outputRect.top;
            dest.right = overlap.right - data->outputRect.left;
            dest.bottom = overlap.bottom - data->outputRect.top;
            ERR(PF_COPY(input, output, &source, &dest));
        }
    }
    Log("CPU render time=%d: passing through, err=%d", in_data->current_time, err);
    extra->cb->checkin_layer_pixels(in_data->effect_ref, current.inputId);
    return err;
}
} // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB2 inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;
    result = PF_REGISTER_EFFECT_EXT2(
        inPtr, inPluginDataCallBackPtr, "Feedback", "RPAY Feedback", "Stylize", AE_RESERVED_INFO, "EffectMain",
        "https://robpayne.co.uk");
    return result;
}

extern "C" DllExport PF_Err EffectMain(
    PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra)
{
    PF_Err err = PF_Err_NONE;
    try
    {
        switch (cmd)
        {
        case PF_Cmd_ABOUT:
            PF_SPRINTF(out_data->return_msg,
                       "Feedback 1.0\rFeeds a processed copy of the output into each frame.\r\r"
                       "Trail Rebuild Limit: how many frames are recalculated when you jump to a frame. "
                       "Playing forward keeps the full trail. Raise it if a trail looks shorter after jumping.");
            break;
        case PF_Cmd_GLOBAL_SETUP:
            err = GlobalSetup(out_data);
            break;
        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_data, out_data);
            break;
        case PF_Cmd_GPU_DEVICE_SETUP:
            err = GPUDeviceSetup(in_data, out_data, static_cast<PF_GPUDeviceSetupExtra*>(extra));
            break;
        case PF_Cmd_GPU_DEVICE_SETDOWN:
            err = GPUDeviceSetdown(static_cast<PF_GPUDeviceSetdownExtra*>(extra));
            break;
        case PF_Cmd_SMART_PRE_RENDER:
            err = PreRender(in_data, out_data, static_cast<PF_PreRenderExtra*>(extra));
            break;
        case PF_Cmd_SMART_RENDER:
            err = RenderCPU(in_data, static_cast<PF_SmartRenderExtra*>(extra));
            break;
        case PF_Cmd_SMART_RENDER_GPU:
            err = RenderGPU(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
            break;
        default:
            break;
        }
    }
    catch (PF_Err& thrown)
    {
        err = thrown;
    }
    catch (...)
    {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}
