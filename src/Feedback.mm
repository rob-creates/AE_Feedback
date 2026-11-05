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
#include <vector>

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
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
    PARAM_ALPHA,
    PARAM_BLUR,
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
    PARAM_DISPLACE_SOURCE,
    PARAM_DISPLACE_LAYER,
    PARAM_CHANNEL_X,
    PARAM_AMOUNT_X,
    PARAM_CHANNEL_Y,
    PARAM_AMOUNT_Y,
    PARAM_DISPLACE_END,
    PARAM_SUBSTEPS,
    PARAM_COUNT
};

enum
{
    SOURCE_SELF = 1,
    SOURCE_LAYER = 2
};

const int kMaxSubsteps = 15;

// A rebuild never starts further back than this. Alpha 0.99 fades out within it; anything higher
// loses what's older.
const int kMaxFramesBack = 600;

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

struct MetalData
{
    std::unique_ptr<FeedbackRenderer> renderer;
};

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
    bool hasInput;
    PF_LRect inputRect;
    A_long mapId;
    bool hasMap;
    PF_LRect mapRect;
};

struct FrameCheckout
{
    A_long time;
    std::vector<SubstepCheckout> substeps; // oldest first; the last one is at the frame's own time
};

struct PreRenderData
{
    CanvasKey canvas;
    PF_LRect outputRect;
    int samplesPerFrame; // the frame's own moment plus any sub-steps
    std::vector<FrameCheckout> frames; // [0] is the current frame, [k] is k frames earlier
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
bool ReadHistory(PF_InData* in_data, const PF_ParamUtilsSuite3& utils, A_long endExclusive, PF_State* state)
{
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

    PF_ParamDef checkout(int index)
    {
        PF_ParamDef def;
        AEFX_CLR_STRUCT(def);
        if (!err)
            err = PF_CHECKOUT_PARAM(in_data, index, time.value, time.step, time.scale, &def);
        return def;
    }

    float slider(int index)
    {
        PF_ParamDef def = checkout(index);
        float value = float(def.u.fs_d.value);
        PF_CHECKIN_PARAM(in_data, &def);
        return value;
    }

    int popup(int index)
    {
        PF_ParamDef def = checkout(index);
        int value = def.u.pd.value;
        PF_CHECKIN_PARAM(in_data, &def);
        return value;
    }

    float angleDegrees(int index)
    {
        PF_ParamDef def = checkout(index);
        float value = float(def.u.ad.value) / 65536.0f;
        PF_CHECKIN_PARAM(in_data, &def);
        return value;
    }

    int integer(int index)
    {
        PF_ParamDef def = checkout(index);
        int value = def.u.sd.value;
        PF_CHECKIN_PARAM(in_data, &def);
        return value;
    }
};

inline SampleTime FrameSample(PF_InData* in_data, A_long time)
{
    return SampleTime{time, in_data->time_step, in_data->time_scale};
}

FrameParams ReadFrameParams(PF_InData* in_data, const SampleTime& time, PF_Err* err)
{
    ParamReader r{in_data, time};
    FrameParams f{};
    f.compositeMode = r.popup(PARAM_COMPOSITE);
    f.inputGain = 1.0f;
    f.frameAlpha = f.stepAlpha = 1.0f;
    f.alpha = r.slider(PARAM_ALPHA);
    f.blurRadius = r.slider(PARAM_BLUR);
    f.inputBlack = r.slider(PARAM_INPUT_BLACK);
    f.inputWhite = r.slider(PARAM_INPUT_WHITE);
    f.gamma = r.slider(PARAM_GAMMA);
    f.outputBlack = r.slider(PARAM_OUTPUT_BLACK);
    f.outputWhite = r.slider(PARAM_OUTPUT_WHITE);
    f.hueDegrees = r.angleDegrees(PARAM_HUE);
    f.saturation = r.slider(PARAM_SATURATION) / 100.0f;
    f.lightness = r.slider(PARAM_LIGHTNESS) / 100.0f;
    f.displaceFromLayer = r.popup(PARAM_DISPLACE_SOURCE) == SOURCE_LAYER;
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
    const float subOffset =
        std::abs(slope - 1.0f) < 1e-5f ? offset * share : offset * (subSlope - 1.0f) / (slope - 1.0f);
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
        PF_ADD_FLOAT_SLIDERX(name, min, max, sMin, sMax, dflt, prec, PF_ValueDisplayFlag_NONE, 0, id);
        return err;
    };
    auto topic = [&](const char* name, int id)
    {
        AEFX_CLR_STRUCT(def);
        PF_ADD_TOPIC(name, id);
        return err;
    };
    auto endTopic = [&](int id)
    {
        AEFX_CLR_STRUCT(def);
        PF_END_TOPIC(id);
        return err;
    };
    auto popup = [&](const char* name, int choices, int dflt, const char* items, int id)
    {
        AEFX_CLR_STRUCT(def);
        PF_ADD_POPUP(name, choices, dflt, items, id);
        return err;
    };
    // These describe the whole loop rather than a moment in it, so they can't be keyframed.
    auto fixedInteger = [&](const char* name, int min, int max, int sliderMax, int dflt, int id)
    {
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        PF_ADD_SLIDER(name, min, max, min, sliderMax, dflt, id);
        return err;
    };

    popup("Composite", 4, COMPOSITE_INPUT_IN_FRONT, "Input In Front|Input Behind|Add|Screen", PARAM_COMPOSITE);
    slider("Alpha", 0, 1, 0, 1, 0.85, 3, PARAM_ALPHA);
    slider("Blur Radius", 0, 500, 0, 50, 0, 2, PARAM_BLUR);

    topic("Levels", PARAM_LEVELS_START);
    slider("Input Black", -1, 2, 0, 1, 0, 3, PARAM_INPUT_BLACK);
    slider("Input White", -1, 2, 0, 1, 1, 3, PARAM_INPUT_WHITE);
    slider("Gamma", 0.1, 10, 0.5, 2, 1, 3, PARAM_GAMMA);
    slider("Output Black", -1, 2, 0, 1, 0, 3, PARAM_OUTPUT_BLACK);
    slider("Output White", -1, 2, 0, 1, 1, 3, PARAM_OUTPUT_WHITE);
    endTopic(PARAM_LEVELS_END);

    topic("HSL", PARAM_HSL_START);
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Hue Shift", 0, PARAM_HUE);
    slider("Saturation", 0, 400, 0, 200, 100, 1, PARAM_SATURATION);
    slider("Lightness", -100, 100, -10, 10, 0, 2, PARAM_LIGHTNESS);
    endTopic(PARAM_HSL_END);

    topic("Displacement", PARAM_DISPLACE_START);
    popup("Source", 2, SOURCE_SELF, "Self|Layer", PARAM_DISPLACE_SOURCE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER("Layer", PF_LayerDefault_NONE, PARAM_DISPLACE_LAYER);
    popup("Horizontal Channel", 5, MAP_CHANNEL_LUMA, "Red|Green|Blue|Alpha|Luma", PARAM_CHANNEL_X);
    slider("Max Horizontal", -4000, 4000, -100, 100, 0, 2, PARAM_AMOUNT_X);
    popup("Vertical Channel", 5, MAP_CHANNEL_LUMA, "Red|Green|Blue|Alpha|Luma", PARAM_CHANNEL_Y);
    slider("Max Vertical", -4000, 4000, -100, 100, 0, 2, PARAM_AMOUNT_Y);
    endTopic(PARAM_DISPLACE_END);

    fixedInteger("Sub-steps", 0, kMaxSubsteps, kMaxSubsteps, 0, PARAM_SUBSTEPS);

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

// Everything a layer can show at a moment, however little AE wants from this effect. A shape
// layer's bounds follow its contents, so this differs from moment to moment.
PF_Err CheckoutWholeLayer(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    PF_RenderRequest request,
    int param,
    A_long id,
    A_long* nextUnusedId,
    const SampleTime& time,
    PF_LRect* rect)
{
    PF_CheckoutResult result;
    AEFX_CLR_STRUCT(result);
    // An empty request returns the layer's full size without rendering anything.
    request.rect = PF_LRect{0, 0, 0, 0};
    PF_Err err = extra->cb->checkout_layer(
        in_data->effect_ref, param, (*nextUnusedId)++, &request, time.value, time.step, time.scale, &result);
    if (err)
        return err;
    request.rect = result.max_result_rect;
    AEFX_CLR_STRUCT(result);
    err = extra->cb->checkout_layer(
        in_data->effect_ref, param, id, &request, time.value, time.step, time.scale, &result);
    *rect = result.result_rect;
    return err;
}

PF_Err PreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    auto preRenderStart = std::chrono::steady_clock::now();

    const A_long now = in_data->current_time;
    const A_long step = in_data->time_step;

    // Every earlier frame affects this one, so AE's cached copy is only valid if they're unchanged.
    {
        AEFX_SuiteScoper<PF_ParamUtilsSuite3> utils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);
        PF_State history;
        AEFX_CLR_STRUCT(history);
        // AE requires this call on every PreRender. With no earlier frames, the zeroed state goes in.
        ReadHistory(in_data, *utils.operator->(), now, &history);
        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(history), &history));
    }

    PF_RenderRequest request = extra->input->output_request;
    request.channel_mask = PF_ChannelMask_ARGB;

    int samplesPerFrame = 1;
    {
        ParamReader r{in_data, FrameSample(in_data, now)};
        samplesPerFrame = r.integer(PARAM_SUBSTEPS) + 1;
        ERR(r.err);
    }

    const float scaleX = float(in_data->downsample_x.num) / float(std::max<A_long>(in_data->downsample_x.den, 1));
    const float scaleY = float(in_data->downsample_y.num) / float(std::max<A_long>(in_data->downsample_y.den, 1));
    const A_long frameLimit = step > 0 ? std::min<A_long>(kMaxFramesBack, std::max<A_long>(now / step, 0)) : 0;

    auto data = std::make_unique<PreRenderData>();
    data->samplesPerFrame = samplesPerFrame;

    // Walk back until earlier frames have faded to nothing or the layer begins. The canvas is
    // whatever this reaches: each input's bounds, plus how far later frames can have spread it.
    // Checking out a frame costs almost nothing; it's only rendered if a rebuild fetches it.
    A_long nextId = 0;
    bool hasArea = false;
    PF_LRect area{};
    A_long focusX = 0, focusY = 0;
    double remaining = 1.0;
    float varianceX = 0.0f, varianceY = 0.0f; // blurs add by variance
    float shiftX = 0.0f, shiftY = 0.0f;
    bool oldest = false;

    for (int k = 0; !err && !oldest; k++)
    {
        FrameCheckout frame{};
        frame.time = now - k * step;

        ParamReader r{in_data, FrameSample(in_data, frame.time)};
        remaining *= r.slider(PARAM_ALPHA);
        const float blurRadius = r.slider(PARAM_BLUR);
        const float amountX = r.slider(PARAM_AMOUNT_X);
        const float amountY = r.slider(PARAM_AMOUNT_Y);
        const bool fromLayer = r.popup(PARAM_DISPLACE_SOURCE) == SOURCE_LAYER;
        ERR(r.err);
        oldest = remaining < kInvisibleContribution || k >= frameLimit;

        // The oldest frame in the window starts from blank, so only its final moment is used.
        const int count = oldest ? 1 : samplesPerFrame;
        bool hasFrameArea = false, hasMap = false;
        PF_LRect frameArea{};

        for (int before = count - 1; !err && before >= 0; before--)
        {
            SubstepCheckout sub{};
            if (!MakeSampleTime(in_data, frame.time, before, samplesPerFrame, &sub.time))
                sub.time = FrameSample(in_data, frame.time - before * step / samplesPerFrame);

            sub.inputId = nextId++;
            ERR(CheckoutWholeLayer(in_data, extra, request, PARAM_INPUT, sub.inputId, &nextId, sub.time, &sub.inputRect));
            sub.hasInput = !IsEmpty(sub.inputRect);
            if (sub.hasInput)
            {
                frameArea = hasFrameArea ? Union(frameArea, sub.inputRect) : sub.inputRect;
                hasFrameArea = true;
            }

            if (!err && fromLayer)
            {
                sub.mapId = nextId++;
                ERR(CheckoutWholeLayer(
                    in_data, extra, request, PARAM_DISPLACE_LAYER, sub.mapId, &nextId, sub.time, &sub.mapRect));
                sub.hasMap = !IsEmpty(sub.mapRect);
                hasMap = hasMap || sub.hasMap;
            }
            frame.substeps.push_back(sub);
        }
        data->frames.push_back(std::move(frame));

        const float sigmaX = blurRadius * 0.5f * scaleX;
        const float sigmaY = blurRadius * 0.5f * scaleY;
        varianceX += sigmaX * sigmaX;
        varianceY += sigmaY * sigmaY;
        if (!fromLayer || hasMap)
        {
            shiftX += std::abs(amountX) * scaleX;
            shiftY += std::abs(amountY) * scaleY;
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
    data->canvas.downsampleXNum = in_data->downsample_x.num;
    data->canvas.downsampleXDen = in_data->downsample_x.den;
    data->canvas.downsampleYNum = in_data->downsample_y.num;
    data->canvas.downsampleYDen = in_data->downsample_y.den;

    Log("prerender frame=%d window=%d samples=%d checkouts=%d canvas=%dx%d ms=%.2f",
        step > 0 ? now / step : 0,
        int(data->frames.size()) - 1,
        samplesPerFrame,
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

enum CacheResult
{
    CACHE_HIT = 0,
    CACHE_NO_ENTRY,
    CACHE_NO_HISTORY,
    CACHE_STALE
};

// Finds the image for `time`, if one exists whose history is still current.
id<MTLBuffer> FindCached(
    PF_InData* in_data,
    const PF_ParamUtilsSuite3& utils,
    FeedbackCache& cache,
    const CanvasKey& canvas,
    A_long time,
    A_long step,
    CanvasKey* imageCanvas,
    CacheResult* result)
{
    *result = CACHE_NO_ENTRY;
    if (!cache.hasFrame(canvas, time))
        return nil;
    *result = CACHE_NO_HISTORY;
    PF_State history;
    AEFX_CLR_STRUCT(history);
    if (!ReadHistory(in_data, utils, time + step, &history))
        return nil;
    *result = CACHE_STALE;
    for (const auto& candidate : cache.candidates(canvas, time))
    {
        A_Boolean same = 0;
        utils.PF_AreStatesIdentical(in_data->effect_ref, &candidate.history, &history, &same);
        if (same)
        {
            cache.touch(candidate.buffer);
            *imageCanvas = candidate.key;
            *result = CACHE_HIT;
            return candidate.buffer;
        }
    }
    return nil;
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
    AEFX_SuiteScoper<PF_ParamUtilsSuite3> utils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);
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
    int previousFrameResult = -1;
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
        // AE only hands out the output after an input has been checked out.
        const SubstepCheckout& currentSub = data->frames[0].substeps.back();
        PF_EffectWorld* currentInput = nullptr;
        PF_EffectWorld* output = nullptr;
        if (currentSub.hasInput)
            ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, currentSub.inputId, &currentInput));
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output));

        // Newest usable starting point within the window.
        id<MTLBuffer> image = nil;
        CanvasKey imageCanvas = canvas;
        int resumeK = frameCount; // the first frame to compute is resumeK - 1
        for (int k = 0; !err && k < frameCount; k++)
        {
            CacheResult result = CACHE_NO_ENTRY;
            image = FindCached(in_data, *utils.operator->(), cache, canvas, data->frames[k].time, step, &imageCanvas, &result);
            if (k == 1)
                previousFrameResult = result;
            if (image)
            {
                resumeK = k;
                break;
            }
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
                const SubstepCheckout& sub = frame.substeps[s];
                const bool isCurrentSample = k == 0 && s == subCount - 1;
                if (!isCurrentSample)
                    ERR(PF_ABORT(in_data));

                FrameParams params{};
                if (!err)
                    params = DivideIntoSubsteps(ReadFrameParams(in_data, sub.time, &err), data->samplesPerFrame);

                auto fetchStart = std::chrono::steady_clock::now();
                PF_EffectWorld* inputWorld = isCurrentSample ? currentInput : nullptr;
                if (!err && !isCurrentSample && sub.hasInput)
                {
                    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, sub.inputId, &inputWorld));
                    fetched++;
                }
                PF_EffectWorld* mapWorld = nullptr;
                if (!err && sub.hasMap)
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

            if (err)
                break;

            // Keep only the frames most likely to be asked for next; the rest of a long rebuild
            // would otherwise push useful images out of the cache.
            PF_State history;
            AEFX_CLR_STRUCT(history);
            if (k < kRebuildFramesKept &&
                ReadHistory(in_data, *utils.operator->(), frame.time + step, &history))
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

            void* outputMemory = nullptr;
            ERR(gpu->GetGPUWorldData(in_data->effect_ref, output, &outputMemory));
            if (!err)
            {
                bool ok = renderer.copyOut(
                    queue, canvas, shown, (__bridge id<MTLBuffer>)outputMemory, output->rowbytes / 16, output->width,
                    output->height, data->outputRect.left - canvas.left, data->outputRect.top - canvas.top);
                if (!ok)
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
            if (shownIsScratch)
                renderer.cache().releaseScratch(shown);
        }

        if (imageIsScratch)
            cache.releaseScratch(image);
        if (currentInput)
            extra->cb->checkin_layer_pixels(in_data->effect_ref, currentSub.inputId);
    }

    Log("frame=%d window=%d computed=%d samples=%d prev=%d fetched=%d fetchMs=%.1f gpuMs=%.1f %dx%d out=%dx%d "
        "took=%.1fms err=%d",
        step > 0 ? in_data->current_time / step : 0,
        frameCount - 1,
        framesComputed,
        data->samplesPerFrame,
        previousFrameResult,
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
    if (!current.hasInput)
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
            PF_SPRINTF(out_data->return_msg, "Feedback 1.0\rFeeds a processed copy of the output into each frame.");
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
