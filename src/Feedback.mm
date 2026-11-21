// Feedback: an Echo-style effect that feeds a processed copy of its own output into each frame.
//
// AE may ask for frames in any order and caches them by hash, so each frame has to be a function
// of its inputs rather than of whatever rendered last. Frame N is therefore computed from a cached
// image of frame N-1, but only if AE confirms all input and parameter history up to N-1 is
// unchanged. Otherwise the loop rebuilds from blank, starting at the layer's in-point or far enough
// back that anything older has faded to nothing. The same history is mixed into AE's frame GUID, so editing an earlier frame
// invalidates everything after it.
//
// The canvas follows the trail: each frame's covers everywhere its visible history can reach, and
// cached images are moved onto it. So a still layer costs nothing extra, and a small moving layer
// only pays for the area its trail actually occupies.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <complex>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
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

#include "BuildStamp.h"
#include "FeedbackRenderer.h"
#include "Hash.h"
#include "LayerMotion.h"
#include "ShaderTypes.h"

namespace
{
enum
{
    PARAM_INPUT = 0,
    PARAM_COMPOSITE,
    PARAM_OPACITY,
    PARAM_ALPHA_START,
    PARAM_ALPHA_MAX,
    PARAM_ALPHA_MIN,
    PARAM_ALPHA_MASK,
    PARAM_ALPHA_END,
    PARAM_BLUR_START,
    PARAM_BLUR,
    PARAM_BLUR_LAYER,
    PARAM_BLUR_END,
    PARAM_LEVELS_START,
    PARAM_INPUT_BLACK,
    PARAM_INPUT_WHITE,
    PARAM_GAMMA,
    PARAM_OUTPUT_BLACK,
    PARAM_OUTPUT_WHITE,
    PARAM_LEVELS_MASK,
    PARAM_LEVELS_END,
    PARAM_HSL_START,
    PARAM_HUE,
    PARAM_SATURATION,
    PARAM_LIGHTNESS,
    PARAM_HSL_MASK,
    PARAM_HSL_END,
    PARAM_TRANSFORM_START,
    PARAM_CENTRE,
    PARAM_POSITION,
    PARAM_UNIFORM_SCALE,
    PARAM_SCALE_HEIGHT,
    PARAM_SCALE_WIDTH,
    PARAM_ROTATION,
    PARAM_TRANSFORM_MASK,
    PARAM_TRANSFORM_END,
    PARAM_DISPLACE_START,
    PARAM_DISPLACE_LAYER,
    PARAM_CHANNEL_X,
    PARAM_AMOUNT_X,
    PARAM_CHANNEL_Y,
    PARAM_AMOUNT_Y,
    PARAM_DISPLACE_MASK,
    PARAM_DISPLACE_END,
    PARAM_EDGE_START,
    PARAM_EDGE_TYPE,
    PARAM_EDGE_STRENGTH,
    PARAM_EDGE_SIZE,
    PARAM_EDGE_SMOOTHING,
    PARAM_EDGE_BLEND,
    PARAM_EDGE_MASK,
    PARAM_EDGE_END,
    PARAM_SAMPLING_GROUP_START,
    PARAM_SUBSTEPS,
    PARAM_PRECISION,
    PARAM_SAMPLING,
    PARAM_LIMIT_PREROLL,
    PARAM_PREROLL_LIMIT,
    PARAM_SAMPLING_GROUP_END,
    PARAM_COUNT
};

// Saved projects find each control by this ID, so a control keeps its ID wherever it moves. In the
// order above. ID 17 belonged to a Source popup, 33 and 34 to Position X and Y sliders, and 44 to a
// Position where the centre went rather than how far it moved, 48 to a comp-space checkbox, 50 to a
// Blur Type popup, 54 to Update Every (Frames), 62 to an Update checkbox, which Opacity replaced, and 64
// to a Static Maps checkbox, and they mustn't be reused.
const int kDiskIds[PARAM_COUNT] = {0, 1, 66, 27, 2, 26, 55, 28, 29, 3, 51, 30, 4, 5, 6, 7, 8, 9, 56, 10,
                                    11, 12, 13, 14, 57, 15, 31, 32, 47, 45, 35, 46, 36, 58, 37, 16, 18, 19, 20, 21,
                                    22, 59, 23, 38, 39, 41, 40, 61, 42, 60, 43, 52, 24, 63, 65, 49, 25, 53};

int DiskId(int param)
{
    return kDiskIds[param];
}

const int kMaxSubsteps = 15;

// With Limit Pre-Roll on, jumping to a frame replays no more than this many frames, so older trail is
// missing until playback passes through it. Playing forward carries the whole trail along in one
// image, so this doesn't shorten it. By default, Alpha Max 0.98 has all but faded out by the limit.
const int kDefaultPreRollLimit = 300;
const int kLongestPreRollLimit = 100000;

// Limits on the canvas, at full resolution for the spread and as actual pixels for the side. These
// only bite on extreme settings, such as large displacement, and trim what lies furthest out.
const float kMaxSpread = 1000.0f;
// How far the blur reaches in a frame, in multiples of its width.
const float kSpreadReach = 6.0f;
const A_long kMaxCanvasSide = 8192;
// A comp-space trail keeps this much beyond the comp's edges, in full-resolution pixels, so a blur
// near an edge still sees what's just outside.
const double kCompMargin = 64.0;
const int kCanvasSnapBits = 6;

// Precision: a rebuild starts where earlier frames have faded below half a level at that depth, so
// they can't change what's seen. 32-bit float has no levels; there it's float's own precision.
enum
{
    PRECISION_8_BIT = 1,
    PRECISION_16_BIT,
    PRECISION_32_BIT_FLOAT,
};

// Pixel Sampling's choices are the SampleFilter values: how the trail is moved, and the input placed,
// between pixels. Bilinear mixes the four nearest, which softens a little with every move that isn't by
// whole pixels. Bicubic softens far less, without fringes. Pixel moves by whole pixels and carries the
// fraction left over on to the next move, so the trail stays exactly as sharp as it went in.

double InvisibleContribution(int precision)
{
    switch (precision)
    {
    case PRECISION_16_BIT:
        return 0.5 / 32768.0;
    case PRECISION_32_BIT_FLOAT:
        return 1.0 / 16777216.0;
    default:
        return 0.002; // as before the control, a shade over half an 8-bit level
    }
}

// A rebuild keeps its last few frames, so scrubbing or caching backwards can resume from them
// instead of rebuilding again.
const int kRebuildFramesKept = 8;

// How many steps of a rebuild run together. More holds on to more of AE's input images at once.
const int kStepsPerBatch = 8;

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

// What PreRender needs to know about a frame. Asking AE is slow when multiplied by a long trail and
// by the many PreRender calls AE makes per render, while the answers only change with the history.
// The layer control for each MapSlot.
const int kMapParams[SLOT_COUNT] = {PARAM_DISPLACE_LAYER, PARAM_ALPHA_MASK,     PARAM_BLUR_LAYER,    PARAM_LEVELS_MASK,
                                    PARAM_HSL_MASK,       PARAM_TRANSFORM_MASK, PARAM_DISPLACE_MASK, PARAM_EDGE_MASK};
// What each map reads, as MapSource values.
using MapSources = std::array<int, SLOT_COUNT>;

struct SampleFacts
{
    SampleTime time;
    // Every control at this moment, as each step reads them. Opacity, inputOpacity, is how much of the
    // layer goes into the trail.
    FrameParams params;
    PF_LRect inputRect; // empty when none does
    PF_LRect mapRects[SLOT_COUNT]; // each map layer's bounds, if it's read
};

struct FrameFacts
{
    float alpha, blurRadius, amountX, amountY;
    float edgeSize; // 0 when edges are off
    bool moves; // Transform moves the trail
    std::vector<SampleFacts> samples; // oldest first; the last one is at the frame's own time
    // What the effect's expressions gave over the frame's moments when these were read
    // (ExpressionPrints::frame), which AE's history doesn't show. Zero without expressions.
    uint64_t expressions = 0;
};

// Remembered frame facts, trusted only while AE reports the same history as when they were read and
// the same bounds for a frame already in the run. History alone isn't enough: at reduced resolution,
// some PreRender calls report a shape layer's bounds at full scale and others scaled, though nothing
// else about the calls differs. Each kind ends up with a run of its own.
// Checking out a layer can run PreRender for another instance of this effect further up the stack,
// so a run stays alive and in place for whoever is using it while the list changes around it.
// AE renders several frames at once, so everything here is read and written under the lock, which
// is never held while calling AE: AE can call back in, from this thread or another.
class FrameFactsMemo
{
public:
    struct Key
    {
        A_long step;
        A_u_long scale;
        PF_RationalScale downsampleX, downsampleY;
        int samplesPerStep;
        uint64_t owner; // as CanvasKey::owner

        bool operator==(const Key& o) const
        {
            return step == o.step && scale == o.scale && SameScale(downsampleX, o.downsampleX) &&
                   SameScale(downsampleY, o.downsampleY) && samplesPerStep == o.samplesPerStep && owner == o.owner;
        }
    };

    struct Run
    {
        Key key;
        // Every frame in `frames` before endExclusive was read with this history. Facts go in only
        // once a history covering them has been read, and `version` counts each change to it.
        A_long endExclusive = 0;
        PF_State history{};
        unsigned version = 0;
        std::unordered_map<A_long, FrameFacts> frames;
        bool hasProbe = false; // one frame's bounds, set once the run knows of a layer with any
        SampleFacts probe{};
    };

    std::vector<std::shared_ptr<Run>> runs; // most recently used first
    std::mutex lock;
};

const size_t kMaxFactRuns = 8;

// A layer's pixels at a moment either come from another render that fetched them (`shared`), or are
// checked out from AE and, if `keep`, copied and kept for other renders under `key` and `state`.
struct FetchTag
{
    FetchKey key{};
    PF_State state{};
    FetchedPixels shared{};
    bool keep = false;
    bool tagged = false; // key and state are known, so it can be looked for and kept
};

struct SubstepCheckout
{
    SampleTime time;
    FrameParams params; // every control at this moment, as SampleFacts has them
    A_long inputId;
    PF_LRect inputRect; // empty if there's no input to use
    FetchTag inputFetch;
    Projective layerToComp; // with the trail in comp space: where the layer is at this moment
    struct Map
    {
        A_long id;
        PF_LRect rect; // empty if there's no map to fetch
        Projective toComp; // and where its layer is
        FetchTag fetch;
    } maps[SLOT_COUNT];
};

struct FrameCheckout
{
    A_long time;
    std::vector<SubstepCheckout> substeps; // oldest first; the last one is at the frame's own time
    uint64_t motion = 0; // when fingerprinted: the fingerprint an image of this frame is kept with
};

// When the trail moves on: one frame of its layer's own comp at a time, whatever rate it's rendered at.
// Nested in a comp at another rate, or rendered at one, AE asks for moments between those frames, a
// rounded step apart, and a trail stepped by that would fade per frame asked for, and never land on a
// frame kept from before. `last` is the newest of the comp's frames at or before the moment asked for,
// `now`, and `share` how far past it `now` lies, in frames: the trail takes that much of a step more.
struct TrailGrid
{
    A_long now = 0;
    A_long last = 0;
    A_long step = 0;
    float share = 0.0f;

    bool between() const { return share > 0.0f; }
};

struct PreRenderData
{
    MapSources sources{}; // all MAP_NONE
    CanvasKey canvas;
    PF_LRect outputRect;
    TrailGrid grid;
    int samplesPerStep; // the frame's own moment plus any sub-steps
    float opacity = 1.0f; // how much of the layer goes into the trail at the moment shown
    // With sub-steps, or between frames, all but In Front and Behind store only the trail, so the current
    // frame goes on top when it's shown, with this composite mode, at its Opacity. Zero when it doesn't.
    int overlayMode = 0;

    // FrameParams::spreadWidest: in comp space, half the longer side of the comp and its margin, as the
    // canvas was before it followed the trail. In the layer, from the canvas, as before.
    float spreadWidest = 0.0f;

    int filter = FILTER_BILINEAR; // Pixel Sampling

    // AE only hands out the output after an input has been checked out, so this always is, even
    // when the layer is empty or the frame needs no computing.
    A_long currentInputId;
    PF_LRect currentInputRect;
    FetchTag currentInputFetch; // AE always gives it, but other renders' rebuilds can use it

    std::vector<FrameCheckout> frames; // the frames to compute: [k] is k frames before grid.last
    // Between frames, the part of a step from grid.last to the moment asked for, with the current input.
    std::optional<SubstepCheckout> between;

    // The cached image from just before those frames, held here so it can't be evicted before the
    // render. Nil means they start from blank.
    TrailImage resume;

    // With the trail in comp space (canvas.compSpace), the canvas is the comp, and the output is the
    // layer's view of it.
    Projective layerToCompNow;

    // On an adjustment layer, images are kept with the project's timestamp from before their inputs
    // were read, and checked against the comp.
    bool watchesComp = false;
    // Images are kept with a fingerprint of what AE's history doesn't show: where the layer went, for a
    // comp-space trail, and what expressions gave. FrameCheckout::motion.
    bool fingerprinted = false;
    AEGP_TimeStamp made{};
};

// The pixels of a layer, at the current resolution, to and from the comp's. AE's moves are at full
// resolution, and a pixel's position is its centre.
Projective AtResolution(const Projective& fullResolution, PF_InData* in_data)
{
    const double sx = ScaleValue(in_data->downsample_x), sy = ScaleValue(in_data->downsample_y);
    return Projective::scale(sx, sy).after(fullResolution).after(Projective::scale(1.0 / sx, 1.0 / sy));
}

// Pixel indices in one image to pixel indices in another, through a move between their spaces.
Projective Between(const Projective& move, A_long fromLeft, A_long fromTop, A_long toLeft, A_long toTop)
{
    return Projective::translate(-0.5 - toLeft, -0.5 - toTop)
        .after(move)
        .after(Projective::translate(0.5 + fromLeft, 0.5 + fromTop));
}

// Has `image`, placed at `rect` in a layer that `layerToComp` puts in the comp, looked up from canvas
// pixels through that move instead of by offset. False if the move can't be undone.
bool WarpOntoCanvas(PlacedImage* image, const Projective& layerToComp, const PF_LRect& rect, const CanvasKey& canvas,
                    PF_InData* in_data)
{
    Projective toLayer;
    if (!AtResolution(layerToComp, in_data).inverse(&toLayer))
        return false;
    image->warped = true;
    Between(toLayer, canvas.left, canvas.top, rect.left, rect.top).toFloats(image->fromCanvas);
    return true;
}

bool MakeSampleTime(PF_InData* in_data, A_long frameTime, A_long step, int substepsBefore, int substeps, SampleTime* out);

// Fingerprints of where the layer went, for comp-space trails: for a frame, the frames that can still
// be seen in it, from `window` frames back. Each frame's reading and each sum is kept, and a frame next
// to one already summed only adds and takes away the ends, so asking for a run of frames is cheap.
// Whether `a` comes before `b` in the comp, for a layer whose time moves `step` a frame.
inline bool Before(A_long a, A_long b, A_long step)
{
    return step > 0 ? a < b : a > b;
}

class MotionPrints
{
public:
    // Where each map layer went counts too. Empty ones are left out. With sub-steps, a frame's reading
    // takes in each of its moments, as the layer is put where it is at each.
    // `expressions`, if given, reads what the effect's expressions give at a moment, which counts too.
    MotionPrints(PF_InData* in_data, LayerMotion& motion, std::optional<LayerMotion> (&maps)[SLOT_COUNT],
                 A_long firstTime, A_long step, int window, int samplesPerStep,
                 std::function<uint64_t(const SampleTime&)> expressions = nullptr)
        : mInData(in_data), mMotion(motion), mExpressions(std::move(expressions)), mFirst(firstTime), mStep(step),
          mWindow(window), mSamplesPerStep(samplesPerStep)
    {
        for (auto& map : maps)
            if (map)
                mMaps.push_back(&*map);
    }

    uint64_t at(A_long time)
    {
        if (auto found = mSums.find(time); found != mSums.end())
            return found->second;
        uint64_t sum = 0;
        auto later = mSums.find(time + mStep), earlier = mSums.find(time - mStep);
        if (aligned(time) && later != mSums.end() && aligned(time + mStep))
        {
            sum = later->second - frame(time + mStep);
            if (Before(oldest(time), oldest(time + mStep), mStep))
                sum += frame(oldest(time));
        }
        else if (aligned(time) && earlier != mSums.end() && aligned(time - mStep))
        {
            sum = earlier->second + frame(time);
            if (Before(oldest(time - mStep), oldest(time), mStep))
                sum -= frame(oldest(time - mStep));
        }
        else
        {
            for (A_long t = oldest(time); !Before(time, t, mStep); t += mStep)
                sum += frame(t);
        }
        mSums.emplace(time, sum);
        return sum;
    }

private:
    A_long oldest(A_long time) const
    {
        const A_long back = time - mWindow * mStep;
        return Before(back, mFirst, mStep) ? mFirst : back;
    }
    // Whether the frames summed for `time` fall on its own steps, so a neighbour's differ only at the ends.
    bool aligned(A_long time) const { return mStep != 0 && (time - oldest(time)) % mStep == 0; }

    uint64_t frame(A_long t)
    {
        auto found = mFrames.find(t);
        if (found == mFrames.end())
        {
            uint64_t print = kHashStart;
            for (int before = mSamplesPerStep - 1; before >= 0; before--)
            {
                SampleTime at{t, mStep, A_u_long(mInData->time_scale)};
                if (before > 0 && !MakeSampleTime(mInData, t, mStep, before, mSamplesPerStep, &at))
                    continue;
                Mix(print, mMotion.fingerprint(at.value, at.scale));
                for (LayerMotion* map : mMaps)
                    Mix(print, map->fingerprint(at.value, at.scale));
                if (mExpressions)
                    Mix(print, mExpressions(at));
            }
            // Scrambled together with its time: the frames are summed, and a reading merely offset by its
            // time would let two histories with the same readings at different frames sum alike.
            Mix(print, t);
            found = mFrames.emplace(t, Scramble(print)).first;
        }
        return found->second;
    }

    PF_InData* mInData;
    LayerMotion& mMotion;
    std::vector<LayerMotion*> mMaps;
    std::function<uint64_t(const SampleTime&)> mExpressions;
    A_long mFirst, mStep;
    int mWindow, mSamplesPerStep;
    std::unordered_map<A_long, uint64_t> mFrames, mSums;
};

// The values expressions gave, read by ExpressionPrints and kept for each instance of the effect until
// anything in the project changes, as the project's timestamp tells. A moment is a time and its scale.
struct ExpressionReadings
{
    using Values = std::unordered_map<uint64_t, uint64_t>; // by ExpressionPrints::Moment
    struct Instance
    {
        AEGP_TimeStamp made{};
        std::vector<int> params; // the controls an expression drives
        // Shared with each call reading it, and copied before a change only while one is.
        std::shared_ptr<Values> readings;
    };
    std::mutex lock;
    std::unordered_map<uint64_t, Instance> instances;
};

struct MetalData
{
    std::unique_ptr<FeedbackRenderer> renderer;
    FrameFactsMemo memo;
    ExpressionReadings expressions;
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

// `substepsBefore` of `substeps` spread over `step` before `frameTime`, expressed exactly. AE times are
// integer fractions, so the scale grows when the step doesn't divide evenly. Each lasts no longer than
// a frame, so layers render as they would at their own frame. Returns false if that would overflow
// AE's 32-bit times, which takes over an hour of timeline at the worst settings.
bool MakeSampleTime(PF_InData* in_data, A_long frameTime, A_long step, int substepsBefore, int substeps, SampleTime* out)
{
    const long long multiple = substeps / std::gcd<long long, long long>(step, substeps);
    const long long subStep = step * multiple / substeps;
    const long long value = frameTime * multiple - substepsBefore * subStep;
    const long long scale = (long long)in_data->time_scale * multiple;
    // As short as either, in whichever direction the layer's time runs.
    const long long frameStep = (long long)in_data->time_step * multiple;
    const long long duration = std::llabs(subStep) < std::llabs(frameStep) ? subStep : frameStep;
    if (value > INT_MAX || value < INT_MIN || scale > INT_MAX || duration > INT_MAX)
        return false;
    *out = SampleTime{A_long(value), A_long(duration), A_u_long(scale)};
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

// Where the history a frame at `time` depends on ends. A trail runs back through the comp's earlier
// frames, which in a time-reversed layer (a negative step) are later in its own time, up to where the
// trail starts. There the history read runs from 0 to just past the start, whatever the frame: more
// than it needs, but it can't miss anything.
A_long HistoryEnd(A_long time, A_long step, A_long firstTime)
{
    return step > 0 ? time + step : firstTime - step;
}

struct ParamReader
{
    PF_InData* in_data;
    SampleTime time;
    PF_Err err = PF_Err_NONE;

    // After a failed checkout, this and later reads return zeros, and nothing is checked in.
    template <typename Get>
    auto read(int index, Get get)
    {
        PF_ParamDef def;
        AEFX_CLR_STRUCT(def);
        if (err)
            return get(def);
        err = PF_CHECKOUT_PARAM(in_data, index, time.value, time.step, time.scale, &def);
        auto value = get(def);
        if (!err)
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

    bool checkbox(int index)
    {
        return read(index, [](const PF_ParamDef& d) { return d.u.bd.value != 0; });
    }

    void point(int index, float* x, float* y)
    {
        const auto [px, py] = read(index, [](const PF_ParamDef& d)
        { return std::pair<float, float>(float(d.u.td.x_value) / 65536.0f, float(d.u.td.y_value) / 65536.0f); });
        *x = px;
        *y = py;
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

// Position is how far the trail moves each frame, which AE gives at the current resolution. With
// Uniform Scale on, Scale Height sets both. The step scales, then turns clockwise on screen, where y
// points down, about the centre, then moves along.
void ReadMove(ParamReader& r, float* moveX, float* moveY, float* scaleX, float* scaleY, float* rotationDegrees)
{
    r.point(PARAM_POSITION, moveX, moveY);
    *moveX /= ScaleValue(r.in_data->downsample_x);
    *moveY /= ScaleValue(r.in_data->downsample_y);
    const bool uniform = r.checkbox(PARAM_UNIFORM_SCALE);
    *scaleY = std::max(r.slider(PARAM_SCALE_HEIGHT) / 100.0f, kMinScale);
    *scaleX = uniform ? *scaleY : std::max(r.slider(PARAM_SCALE_WIDTH) / 100.0f, kMinScale);
    *rotationDegrees = r.angleDegrees(PARAM_ROTATION);
}

void SetLinear(double turnDegrees, double scaleX, double scaleY, float* linear)
{
    const double turn = turnDegrees * M_PI / 180.0, c = std::cos(turn), s = std::sin(turn);
    linear[0] = float(c * scaleX);
    linear[1] = float(-s * scaleY);
    linear[2] = float(s * scaleX);
    linear[3] = float(c * scaleY);
}

// `share` of one frame's move. As a sub-step, taken n = 1 / share times, it makes the frame's move: an
// nth root of the linear part, and the move that adds up to the frame's along the way. Between frames,
// a share that isn't such a fraction takes the move part of the way: the linear part to that power, and
// the move that would add up the same way. With an even scale, turning by that share of the angle is
// the power, and keeps a turn of more than half a circle going the way it was set.
void DivideMove(FrameParams* f, double share)
{
    using C = std::complex<double>;
    const double A[4] = {f->linear[0], f->linear[1], f->linear[2], f->linear[3]};
    double B[4];
    auto evenRoot = [&]
    {
        float r[4];
        SetLinear(f->rotationDegrees * share, std::pow(f->scaleX, share), std::pow(f->scaleY, share), r);
        for (int i = 0; i < 4; i++)
            B[i] = r[i];
    };
    if (std::abs(f->scaleX - f->scaleY) < 1e-6f)
        evenRoot();
    else
    {
        // A function of a 2x2 matrix is alpha I + beta A, from its value at the two eigenvalues.
        const C half = (A[0] + A[3]) / 2.0, gap = std::sqrt(C(half * half - (A[0] * A[3] - A[1] * A[2])));
        const C l1 = half + gap, l2 = half - gap;
        const C r1 = std::pow(l1, share), r2 = std::pow(l2, share);
        C alpha, beta;
        if (std::abs(l1 - l2) < 1e-9)
        {
            beta = share * r1 / l1;
            alpha = r1 - beta * l1;
        }
        else
        {
            beta = (r1 - r2) / (l1 - l2);
            alpha = (l1 * r2 - l2 * r1) / (l1 - l2);
        }
        if (std::abs(alpha.imag()) > 1e-6 || std::abs(beta.imag()) > 1e-6)
            evenRoot(); // no real root, as when turned half a circle: the closest real stand-in
        else
            for (int i = 0; i < 4; i++)
                B[i] = beta.real() * A[i] + ((i == 0 || i == 3) ? alpha.real() : 0.0);
    }

    const double mx = f->moveX, my = f->moveY;
    for (int i = 0; i < 4; i++)
        f->linear[i] = float(B[i]);
    const int n = int(std::lround(1.0 / share));
    if (std::abs(double(n) * share - 1.0) > 1e-6)
    {
        // As if 1 / share steps of B came to A: the move is (B - I)(A - I)^-1 times the frame's. Where
        // A - I can't be undone, as for a move with no scale or turn, the same from A's eigenvalues l,
        // each giving (l^share - 1) / (l - 1), or share where l is 1.
        const double D[4] = {A[0] - 1.0, A[1], A[2], A[3] - 1.0};
        const double det = D[0] * D[3] - D[1] * D[2];
        if (std::abs(det) > 1e-9)
        {
            const double yx = (D[3] * mx - D[1] * my) / det, yy = (-D[2] * mx + D[0] * my) / det;
            f->moveX = float((B[0] - 1.0) * yx + B[1] * yy);
            f->moveY = float(B[2] * yx + (B[3] - 1.0) * yy);
            return;
        }
        const C half = (A[0] + A[3]) / 2.0, gap = std::sqrt(C(half * half - (A[0] * A[3] - A[1] * A[2])));
        const C l1 = half + gap, l2 = half - gap;
        auto h = [share](C l) { return std::abs(l - 1.0) < 1e-9 ? C(share) : (std::pow(l, share) - 1.0) / (l - 1.0); };
        C alpha = h(l1), beta = 0.0;
        if (std::abs(l1 - l2) >= 1e-9)
        {
            beta = (h(l1) - h(l2)) / (l1 - l2);
            alpha = (l1 * h(l2) - l2 * h(l1)) / (l1 - l2);
        }
        f->moveX = float(alpha.real() * mx + beta.real() * (A[0] * mx + A[1] * my));
        f->moveY = float(alpha.real() * my + beta.real() * (A[2] * mx + A[3] * my));
        return;
    }

    // n steps move by (I + B + ... + B^(n-1)) step, which must come to the frame's move.
    double sum[4] = {1, 0, 0, 1}, power[4] = {1, 0, 0, 1};
    for (int k = 1; k < n; k++)
    {
        const double p[4] = {power[0] * B[0] + power[1] * B[2], power[0] * B[1] + power[1] * B[3],
                             power[2] * B[0] + power[3] * B[2], power[2] * B[1] + power[3] * B[3]};
        for (int i = 0; i < 4; i++)
        {
            power[i] = p[i];
            sum[i] += p[i];
        }
    }
    const double det = sum[0] * sum[3] - sum[1] * sum[2];
    if (std::abs(det) > 1e-9)
    {
        f->moveX = float((sum[3] * mx - sum[1] * my) / det);
        f->moveY = float((-sum[2] * mx + sum[0] * my) / det);
    }
    else
    {
        f->moveX = float(mx / n);
        f->moveY = float(my / n);
    }
}

FrameParams ReadFrameParams(PF_InData* in_data, const SampleTime& time, PF_Err* err)
{
    ParamReader r{in_data, time};
    FrameParams f{};
    f.compositeMode = r.popup(PARAM_COMPOSITE);
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
    r.point(PARAM_CENTRE, &f.centreX, &f.centreY);
    ReadMove(r, &f.moveX, &f.moveY, &f.scaleX, &f.scaleY, &f.rotationDegrees);
    SetLinear(f.rotationDegrees, f.scaleX, f.scaleY, f.linear);
    f.edgeType = r.popup(PARAM_EDGE_TYPE);
    f.edgeSize = r.slider(PARAM_EDGE_SIZE);
    f.edgeSmoothing = std::clamp(r.slider(PARAM_EDGE_SMOOTHING) / 100.0f, 0.0f, 1.0f);
    f.edgeStrength = r.slider(PARAM_EDGE_STRENGTH);
    f.edgeBlend = r.popup(PARAM_EDGE_BLEND);
    f.inputOpacity = std::clamp(r.slider(PARAM_OPACITY), 0.0f, 1.0f);
    *err = r.err;
    return f;
}

// In all but In Front and Behind, the input builds up with every step rather than replacing what's
// there, so sub-steps each take a share of it.
bool BuildsUpInput(int compositeMode)
{
    return compositeMode != COMPOSITE_INPUT_IN_FRONT && compositeMode != COMPOSITE_INPUT_BEHIND;
}

// Each control describes what happens per frame. A step that makes `share` of that, one of n sub-steps
// or part of the way to a moment between frames, applies that much of each, so neither changes the
// trail's overall look, only how smoothly it gets there. `frameEnd` is the sub-step at the frame's own
// moment, which is where Levels that can't be divided goes. For a share under 1.
void DivideShares(FrameParams& f, float share, bool frameEnd)
{
    if (f.alphaSlope > 0.0f)
    {
        const float subSlope = std::pow(f.alphaSlope, share);
        f.alphaOffset = RootOffset(f.alphaOffset, f.alphaSlope, subSlope, share);
        f.alphaSlope = subSlope;
    }
    // The blur's widths add, so n small ones of radius r / n equal one of radius r.
    f.blurRadius *= share;
    f.amountX *= share;
    f.amountY *= share;
    f.hueDegrees *= share;
    f.saturation = std::pow(f.saturation, share);
    f.lightness *= share;
    DivideMove(&f, share);
    // Edges of edges would build up with every sub-step, so they're found once a frame.
    if (!frameEnd)
        f.edgeStrength = 0.0f;

    // Levels is a linear map plus a gamma. Exact roots of each stand in for the combination. A flat or
    // inverted map has no root that keeps its look, so it's applied whole once a frame instead.
    const float range = std::max(f.inputWhite - f.inputBlack, 1e-5f);
    const float slope = (f.outputWhite - f.outputBlack) / range;
    if (slope > 0.0f)
    {
        const float offset = f.outputBlack - slope * f.inputBlack;
        const float subSlope = std::pow(slope, share);
        const float subOffset = RootOffset(offset, slope, subSlope, share);
        f.gamma = std::pow(f.gamma, share);
        f.inputBlack = 0.0f;
        f.inputWhite = 1.0f;
        f.outputBlack = subOffset;
        f.outputWhite = subSlope + subOffset;
    }
    else if (!frameEnd)
    {
        f.gamma = 1.0f;
        f.inputBlack = f.outputBlack = 0.0f;
        f.inputWhite = f.outputWhite = 1.0f;
    }

    // Each step adds only enough for still content to settle where one step per frame would, and the
    // current frame is laid over the finished trail at full strength when it's displayed. Just below
    // a fade of 1, where the shader's formula is still well behaved.
    if (BuildsUpInput(f.compositeMode))
    {
        f.splitsInput = 1;
        f.frameAlpha = std::min(f.alpha, 0.999f);
        f.stepAlpha = std::pow(f.frameAlpha, share);
    }
}

// Pixel Sampling at Pixel moves the trail by whole pixels of the canvas, whose resolution this is, and
// `carry` keeps what's left over from one step to the next.
struct WholePixels
{
    float resolutionX, resolutionY;
    TrailCarry* carry;
};

// Replaces the move DivideShares made with one of whole pixels only. Resampling by a fraction of a pixel
// mixes neighbours, and fed back step after step that softens the trail. `whole` is the frame undivided.
//
// The frame's move, x -> L (x - centre) + centre + move, is the same as moving by u = L^-1 move and then
// applying L about the centre. Each step takes its share of u, rounded to whole pixels, and `carry` keeps
// what rounding left over for the next, so the steps still add up to the frame's move. L, and
// displacement, which move each pixel differently and can't be carried, apply once a frame, at the
// frame's own moment, as with no sub-steps. The fraction still owed is turned with the trail.
void SnapMove(const FrameParams& whole, float share, bool frameEnd, const WholePixels& pixels, FrameParams& f)
{
    const float resolutionX = pixels.resolutionX, resolutionY = pixels.resolutionY;
    TrailCarry* carry = pixels.carry;
    // In canvas pixels, as TrailTransform has it.
    const double a = whole.linear[0], b = double(whole.linear[1]) * resolutionX / resolutionY,
                 c = double(whole.linear[2]) * resolutionY / resolutionX, d = whole.linear[3];
    const double det = a * d - b * c;
    if (CollapsesTrail(whole) || std::abs(det) < 1e-12)
    {
        *carry = {};
        return;
    }
    const double moveX = double(whole.moveX) * resolutionX, moveY = double(whole.moveY) * resolutionY;
    const double wantX = carry->x + share * (d * moveX - b * moveY) / det;
    const double wantY = carry->y + share * (a * moveY - c * moveX) / det;
    const double stepX = std::floor(wantX + 0.5), stepY = std::floor(wantY + 0.5);
    double owedX = wantX - stepX, owedY = wantY - stepY;

    const bool turns = frameEnd && !(a == 1.0 && b == 0.0 && c == 0.0 && d == 1.0);
    if (turns)
    {
        // x -> L (x + step - centre) + centre: L's move is L times the step.
        std::copy(std::begin(whole.linear), std::end(whole.linear), f.linear);
        f.moveX = float((a * stepX + b * stepY) / resolutionX);
        f.moveY = float((c * stepX + d * stepY) / resolutionY);
        const double turnedX = a * owedX + b * owedY;
        owedY = c * owedX + d * owedY;
        owedX = turnedX;
    }
    else
    {
        f.linear[0] = f.linear[3] = 1.0f;
        f.linear[1] = f.linear[2] = 0.0f;
        f.moveX = float(stepX / resolutionX);
        f.moveY = float(stepY / resolutionY);
    }
    carry->x = float(owedX);
    carry->y = float(owedY);

    f.amountX = frameEnd ? whole.amountX : 0.0f;
    f.amountY = frameEnd ? whole.amountY : 0.0f;
}

// One step's worth of a frame's controls, `share` of it, as DivideShares has it. With `pixels`, the
// trail's move is by whole pixels, as SnapMove has it, at any share.
FrameParams DivideIntoSteps(const FrameParams& frame, float share, bool frameEnd, const WholePixels* pixels = nullptr)
{
    FrameParams f = frame;
    if (share < 1.0f)
        DivideShares(f, share, frameEnd);
    if (pixels)
        SnapMove(frame, share, frameEnd, *pixels, f);
    return f;
}

PF_Err GlobalSetup(PF_OutData* out_data)
{
    // Must match AE_Effect_Version in FeedbackPiPL.r (AE compares the encoded value, build included).
    out_data->my_version = PF_VERSION(1, 0, 0, PF_Stage_DEVELOP, 0);
    // Must match AE_Effect_Global_OutFlags in FeedbackPiPL.r.
    out_data->out_flags = PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_DEEP_COLOR_AWARE |
                          PF_OutFlag_SEND_UPDATE_PARAMS_UI;
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |
                           PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT | PF_OutFlag2_I_MIX_GUID_DEPENDENCIES |
                           PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 | PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
    return PF_Err_NONE;
}

// A control that does nothing while another is set is greyed out, as in AE's Transform effect: Scale
// Width while Uniform Scale is on, and Pre-Roll Limit while Limit Pre-Roll is off.
PF_Err GreyOut(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], int param, bool grey)
{
    PF_ParamDef def = *params[param];
    if (grey == ((def.ui_flags & PF_PUI_DISABLED) != 0))
        return PF_Err_NONE;
    if (grey)
        def.ui_flags |= PF_PUI_DISABLED;
    else
        def.ui_flags &= ~PF_PUI_DISABLED;
    AEFX_SuiteScoper<PF_ParamUtilsSuite3> utils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);
    return utils->PF_UpdateParamUI(in_data->effect_ref, param, &def);
}

PF_Err UpdateParamsUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[])
{
    PF_Err err = GreyOut(in_data, out_data, params, PARAM_SCALE_WIDTH, params[PARAM_UNIFORM_SCALE]->u.bd.value != 0);
    ERR(GreyOut(in_data, out_data, params, PARAM_PREROLL_LIMIT, params[PARAM_LIMIT_PREROLL]->u.bd.value == 0));
    return err;
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
    // Controls that describe the whole loop rather than a moment in it are `fixed`: they can't be keyframed.
    auto popup = [&](const char* name, int choices, int dflt, const char* items, int id, bool fixed = false)
    {
        AEFX_CLR_STRUCT(def);
        if (fixed)
            def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        PF_ADD_POPUP(name, choices, dflt, items, DiskId(id));
        return err;
    };
    // A layer to read, starting at None. Each group's mask: its luminance at each pixel is how much of
    // the group applies there, from none at black to all of it at white. None applies it everywhere.
    auto layerControl = [&](const char* name, int id)
    {
        AEFX_CLR_STRUCT(def);
        PF_ADD_LAYER(name, PF_LayerDefault_NONE, DiskId(id));
        return err;
    };
    auto fixedInteger = [&](const char* name, int min, int max, int sliderMax, int dflt, int id)
    {
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        PF_ADD_SLIDER(name, min, max, min, sliderMax, dflt, DiskId(id));
        return err;
    };

    popup("Composite", 10, COMPOSITE_INPUT_IN_FRONT,
          "Input In Front|Input Behind|Add|Screen|Multiply|Lighten|Darken|Overlay|Add/Subtract|Difference", PARAM_COMPOSITE);
    // How much of the layer goes into the trail, as if it were that transparent. At 0 it adds nothing, as
    // if it were empty, while the trail goes on moving and fading. Keyframed, it fades the layer in and out
    // of the trail, or with Hold keyframes stamps it in at those moments.
    slider("Opacity", 0, 1, 0, 1, 1, 3, PARAM_OPACITY);

    topic("Alpha", PARAM_ALPHA_START);
    slider("Alpha Max", 0, 1, 0, 1, 0.85, 3, PARAM_ALPHA_MAX);
    slider("Alpha Min", 0, 1, 0, 1, 0, 3, PARAM_ALPHA_MIN);
    layerControl("Alpha Mask Layer", PARAM_ALPHA_MASK);
    endTopic(PARAM_ALPHA_END);

    topic("Blur", PARAM_BLUR_START);
    slider("Blur Radius", 0, 500, 0, 50, 0, 2, PARAM_BLUR);
    // Scales Blur Radius at each pixel, so black blurs nothing and white the whole radius.
    layerControl("Blur Mask Layer", PARAM_BLUR_LAYER);
    endTopic(PARAM_BLUR_END);

    topic("Levels", PARAM_LEVELS_START);
    slider("Input Black", -1, 2, 0, 1, 0, 3, PARAM_INPUT_BLACK);
    slider("Input White", -1, 2, 0, 1, 1, 3, PARAM_INPUT_WHITE);
    slider("Gamma", 0.1, 10, 0.5, 2, 1, 3, PARAM_GAMMA);
    slider("Output Black", -1, 2, 0, 1, 0, 3, PARAM_OUTPUT_BLACK);
    slider("Output White", -1, 2, 0, 1, 1, 3, PARAM_OUTPUT_WHITE);
    layerControl("Levels Mask Layer", PARAM_LEVELS_MASK);
    endTopic(PARAM_LEVELS_END);

    topic("HSL", PARAM_HSL_START);
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Hue Shift", 0, DiskId(PARAM_HUE));
    slider("Saturation", 0, 400, 0, 200, 100, 1, PARAM_SATURATION);
    slider("Lightness", -100, 100, -10, 10, 0, 2, PARAM_LIGHTNESS);
    layerControl("HSL Mask Layer", PARAM_HSL_MASK);
    endTopic(PARAM_HSL_END);

    // Moves the trail each frame, before displacement shifts it: scaled and turned about Centre, then
    // moved by Position, so a Position of 0, 0 leaves it in place.
    topic("Transform", PARAM_TRANSFORM_START);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Centre", 50, 50, FALSE, DiskId(PARAM_CENTRE));
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Position", 0, 0, FALSE, DiskId(PARAM_POSITION));
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Uniform Scale", TRUE, PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY,
                     DiskId(PARAM_UNIFORM_SCALE));
    slider("Scale Height", 0, 1000, 50, 200, 100, 2, PARAM_SCALE_HEIGHT);
    slider("Scale Width", 0, 1000, 50, 200, 100, 2, PARAM_SCALE_WIDTH);
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Rotation", 0, DiskId(PARAM_ROTATION));
    // Each pixel moves this much of the way.
    layerControl("Transform Mask Layer", PARAM_TRANSFORM_MASK);
    endTopic(PARAM_TRANSFORM_END);

    topic("Displacement", PARAM_DISPLACE_START);
    layerControl("Map Layer", PARAM_DISPLACE_LAYER);
    popup("Horizontal Channel", 5, MAP_CHANNEL_LUMA, "Red|Green|Blue|Alpha|Luma", PARAM_CHANNEL_X);
    slider("Max Horizontal", -4000, 4000, -100, 100, 0, 2, PARAM_AMOUNT_X);
    popup("Vertical Channel", 5, MAP_CHANNEL_LUMA, "Red|Green|Blue|Alpha|Luma", PARAM_CHANNEL_Y);
    slider("Max Vertical", -4000, 4000, -100, 100, 0, 2, PARAM_AMOUNT_Y);
    layerControl("Displacement Mask Layer", PARAM_DISPLACE_MASK);
    endTopic(PARAM_DISPLACE_END);

    // Found on the moved trail once a frame, then blurred and graded with it. Off at Strength 0.
    topic("Edge", PARAM_EDGE_START);
    popup("Edge Type", 2, EDGE_COLOUR, "Colour|Luminance", PARAM_EDGE_TYPE);
    slider("Edge Strength", 0, 100, 0, 4, 0, 3, PARAM_EDGE_STRENGTH);
    slider("Edge Sample Size", 0.1, 100, 0.5, 10, 1, 2, PARAM_EDGE_SIZE);
    // How much the image is blurred before it's sampled, as a share of the most, half the spacing: less
    // keeps edges crisper, but detail finer than the spacing can then build up into a grid.
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Edge Smoothing", 0, 100, 0, 100, 0, 1, PF_ValueDisplayFlag_PERCENT, 0,
                         DiskId(PARAM_EDGE_SMOOTHING));
    popup("Edge Blend", 10, COMPOSITE_ADD,
          "Over|Behind|Add|Screen|Multiply|Lighten|Darken|Overlay|Add/Subtract|Difference", PARAM_EDGE_BLEND);
    layerControl("Edge Mask Layer", PARAM_EDGE_MASK);
    endTopic(PARAM_EDGE_END);

    topic("Sampling", PARAM_SAMPLING_GROUP_START);
    fixedInteger("Sub-steps", 0, kMaxSubsteps, kMaxSubsteps, 0, PARAM_SUBSTEPS);
    // How faint older frames may get before a rebuild leaves them out, and how far the blur's faint
    // tails are followed: below what the depth can show. The trail itself is always worked in float.
    popup("Cut-off Precision", 3, PRECISION_8_BIT, "8 bit|16 bit|32 bit float", PARAM_PRECISION, true);
    // Bilinear softens a trail moved by fractions of a pixel, a little more with every step; Bicubic
    // keeps it far crisper; Pixel keeps it exactly as sharp, for pixel art and hard-edged graphics.
    popup("Pixel Sampling", 3, FILTER_BILINEAR, "Bilinear|Bicubic|Pixel", PARAM_SAMPLING, true);
    // Off, jumping to a frame recalculates the whole trail before it, however long that takes. On, only
    // the last Pre-Roll Limit frames are, which is quicker but can show a shorter trail.
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Limit Pre-Roll", FALSE, PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY,
                     DiskId(PARAM_LIMIT_PREROLL));
    fixedInteger("Pre-Roll Limit (Frames)", 1, kLongestPreRollLimit, 3000, kDefaultPreRollLimit, PARAM_PREROLL_LIMIT);
    endTopic(PARAM_SAMPLING_GROUP_END);


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

// Where a rectangle goes under a move: the box around its corners. False if a corner has no place.
bool MovedBounds(const Projective& move, const PF_LRect& r, PF_LRect* out)
{
    double lo[2] = {1e18, 1e18}, hi[2] = {-1e18, -1e18};
    for (A_long x : {r.left, r.right})
        for (A_long y : {r.top, r.bottom})
        {
            double p[2];
            if (!move.apply(double(x), double(y), &p[0], &p[1]))
                return false;
            for (int i = 0; i < 2; i++)
            {
                lo[i] = std::min(lo[i], p[i]);
                hi[i] = std::max(hi[i], p[i]);
            }
        }
    const auto clampLong = [](double v) { return A_long(std::clamp(v, -1e8, 1e8)); };
    *out = PF_LRect{clampLong(std::floor(lo[0])), clampLong(std::floor(lo[1])), clampLong(std::ceil(hi[0])),
                    clampLong(std::ceil(hi[1]))};
    return true;
}

#ifdef FEEDBACK_LOG
// Tests turn this on to render as before the canvas followed a comp-space trail and maps were read only
// over the canvas, to check that doing so changes nothing.
bool WholeCanvas()
{
    static const bool whole = getenv("FEEDBACK_WHOLE_CANVAS") != nullptr;
    return whole;
}
#else
bool WholeCanvas()
{
    return false;
}
#endif

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

// What PreRender needs to find layers other renders have fetched. Inactive, everything is checked out.
struct FetchContext
{
    const PF_ParamUtilsSuite3* utils = nullptr;
    FeedbackCache* cache = nullptr;
    uint64_t instance = 0;

    bool active() const { return utils && cache && instance; }
};

// What a layer fetched at a moment is kept under. False if AE can't say what state it's in.
bool MakeFetchTag(PF_InData* in_data, const PF_RenderRequest& request, const FetchContext& fetch, int param,
                  const SampleTime& time, const PF_LRect& area, FetchTag* tag)
{
    *tag = FetchTag{};
    if (!fetch.active())
        return false;
    tag->key = FetchKey{fetch.instance,
                        param,
                        time.value,
                        time.step,
                        time.scale,
                        in_data->downsample_x,
                        in_data->downsample_y,
                        int(in_data->quality),
                        int(request.field),
                        int(request.preserve_rgb_of_zero_alpha),
                        area};
    // Everything the effect reads over that moment, as AE sees it, layers included.
    A_Time start{time.value, time.scale}, duration{std::max<A_long>(std::abs(time.step), 1), time.scale};
    tag->keep = fetch.utils->PF_GetCurrentState(in_data->effect_ref, PF_ParamIndex_CHECK_ALL, &start, &duration,
                                                &tag->state) == PF_Err_NONE;
    tag->tagged = tag->keep;
    return tag->keep;
}

// Far bigger than any layer, so this asks for the whole layer whatever its bounds are in this call.
const A_long kAnyLayer = 1000000;
const PF_LRect kWholeLayer{-kAnyLayer, -kAnyLayer, kAnyLayer, kAnyLayer};

// Checks out `area` of a layer at a moment, unless another render has fetched it, unchanged since.
// Pixels another render fetched for what `tag` names, while the project is as it was then.
bool FindFetched(PF_InData* in_data, const PF_ParamUtilsSuite3& utils, FeedbackCache& cache, const FetchTag& tag,
                 FetchedPixels* pixels)
{
    for (const auto& candidate : cache.fetchCandidates(tag.key))
    {
        A_Boolean same = 0;
        utils.PF_AreStatesIdentical(in_data->effect_ref, &candidate.state, &tag.state, &same);
        if (same)
        {
            *pixels = candidate.pixels;
            return true;
        }
    }
    return false;
}

PF_Err CheckoutOrShare(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    const FetchContext& fetch,
    int param,
    const SampleTime& time,
    const PF_LRect& area,
    A_long id,
    PF_LRect* rect,
    FetchTag* tag)
{
    if (MakeFetchTag(in_data, request, fetch, param, time, area, tag) &&
        FindFetched(in_data, *fetch.utils, *fetch.cache, *tag, &tag->shared))
    {
        tag->keep = false;
        *rect = tag->shared.rect;
        return PF_Err_NONE;
    }
    PF_CheckoutResult result;
    const PF_Err err = CheckoutLayer(in_data, extra, request, area, param, time, id, &result);
    *rect = result.result_rect;
    return err;
}

// The fingerprint of what AE's history doesn't show up to a frame: where a comp-space trail's layer went,
// and what expressions gave. Null when there's nothing to fingerprint.
using MotionAt = std::function<uint64_t(A_long)>;

// Works out what an image of a frame depends on now (ImageValidity), and whether one kept with an earlier
// validity can stand for it. `prints` gives the fingerprint at a frame, and `watch`, on an adjustment
// layer, tells whether anything in the comp has changed since an image was made; one that couldn't be set
// up passes nothing. `made` is the project's timestamp when this render read its inputs.
class ValidityCheck
{
public:
    ValidityCheck(PF_InData* in_data, const PF_ParamUtilsSuite3& utils, const CanvasKey& canvas, A_long step,
                  MotionAt prints, const CompWatch* watch, const AEGP_TimeStamp& made)
        : mInData(in_data), mUtils(utils), mFirstTime(canvas.firstTime), mStep(step), mPrints(std::move(prints)),
          mWatch(watch), mMade(made)
    {
    }

    // What an image of the frame at `time` made now depends on. False if AE's history can't be read.
    bool at(A_long time, ImageValidity* out) const
    {
        if (!ReadHistory(mInData, mUtils, HistoryEnd(time, mStep, mFirstTime), &out->history))
            return false;
        out->prints = mPrints ? mPrints(time) : 0;
        out->made = mMade;
        return true;
    }

    // Whether an image kept with `kept` can stand for the frame at `time`, which now depends on `now`.
    bool same(const ImageValidity& kept, const ImageValidity& now, A_long time) const
    {
        A_Boolean same = 0;
        mUtils.PF_AreStatesIdentical(mInData->effect_ref, &kept.history, &now.history, &same);
        return same && kept.prints == now.prints && (!mWatch || mWatch->unchangedSince(kept.made, time + mStep));
    }

private:
    PF_InData* mInData;
    const PF_ParamUtilsSuite3& mUtils;
    A_long mFirstTime, mStep;
    MotionAt mPrints;
    const CompWatch* mWatch;
    AEGP_TimeStamp mMade;
};

// Finds the image for `time`, if one exists that's still valid, for a render of `now`.
// An image drawn for a later frame sits on that frame's canvas, which follows the trail further on: a
// canvas held below the trail's size leaves out the older end that this frame's canvas still has, so
// such an image is used only if its canvas covers this one. An earlier frame's canvas is behind, where
// the trail is, and is used as before. Later means further along `step`, which runs backwards in a
// time-reversed layer. No buffer if there's none.
TrailImage FindCached(
    FeedbackCache& cache,
    const ValidityCheck& validity,
    const CanvasKey& canvas,
    A_long now,
    A_long time,
    A_long step)
{
    const auto candidates = cache.candidates(canvas, time);
    ImageValidity current;
    if (candidates.empty() || !validity.at(time, &current))
        return {};
    for (const auto& candidate : candidates)
    {
        const bool drawnLater = (step > 0 && candidate.drawnFor > now) || (step < 0 && candidate.drawnFor < now);
        if (validity.same(candidate.validity, current, time) && (!drawnLater || candidate.image.canvas.contains(canvas)))
        {
            cache.touch(candidate.image.buffer);
            return candidate.image;
        }
    }
    return {};
}

// The newest image among the `frames` frames ending at `now`, or -1. Returns how many frames before
// `now` it is.
int FindResume(
    FeedbackCache& cache,
    const ValidityCheck& validity,
    const CanvasKey& canvas,
    A_long now,
    A_long step,
    int frames,
    TrailImage* image)
{
    for (int k = 0; k < frames; k++)
    {
        *image = FindCached(cache, validity, canvas, now, now - k * step, step);
        if (image->buffer)
            return k;
    }
    return -1;
}

// What a render may take from a run: frames before trustedEnd, which its own history matched.
struct RunView
{
    std::shared_ptr<FrameFactsMemo::Run> run;
    A_long trustedEnd = 0;
    unsigned version = 0;
};

// Finds the remembered facts for this history, making room for a new set if there are none.
RunView FindFacts(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    const PF_ParamUtilsSuite3& utils,
    FrameFactsMemo& memo,
    const FrameFactsMemo::Key& key,
    A_long* nextId,
    PF_Err* err)
{
    std::vector<std::shared_ptr<FrameFactsMemo::Run>> runs;
    {
        std::lock_guard<std::mutex> guard(memo.lock);
        runs = memo.runs;
    }
    for (const auto& run : runs)
    {
        if (!(run->key == key))
            continue;
        A_long endExclusive;
        unsigned version;
        PF_State remembered;
        bool hasProbe;
        SampleFacts probe;
        {
            std::lock_guard<std::mutex> guard(memo.lock);
            endExclusive = run->endExclusive;
            version = run->version;
            remembered = run->history;
            hasProbe = run->hasProbe;
            probe = run->probe;
        }
        PF_State history;
        A_Boolean same = 0;
        if (ReadHistory(in_data, utils, endExclusive, &history))
            utils.PF_AreStatesIdentical(in_data->effect_ref, &remembered, &history, &same);
        if (!same)
            continue;
        if (hasProbe)
        {
            PF_LRect bounds{};
            *err = LayerBounds(in_data, extra, request, PARAM_INPUT, probe.time, nextId, &bounds);
            if (*err)
                return {};
            if (!SameRect(bounds, probe.inputRect))
                continue;
        }
        std::lock_guard<std::mutex> guard(memo.lock);
        auto at = std::find(memo.runs.begin(), memo.runs.end(), run);
        if (at != memo.runs.end())
            std::rotate(memo.runs.begin(), at, at + 1);
        return {run, endExclusive, version};
    }
    auto run = std::make_shared<FrameFactsMemo::Run>();
    run->key = key;
    std::lock_guard<std::mutex> guard(memo.lock);
    memo.runs.insert(memo.runs.begin(), run);
    if (memo.runs.size() > kMaxFactRuns)
        memo.runs.pop_back();
    return {run, 0, run->version};
}

// Asks AE about one frame in the history, whose samples spread over the `step` before `time`.
PF_Err ReadFrameFacts(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    A_long time,
    A_long step,
    int samplesPerStep,
    const MapSources& sources,
    A_long* nextId,
    FrameFacts* facts)
{
    ParamReader r{in_data, FrameSample(in_data, time)};
    facts->alpha = r.slider(PARAM_ALPHA_MAX);
    facts->blurRadius = r.slider(PARAM_BLUR);
    facts->amountX = r.slider(PARAM_AMOUNT_X);
    facts->amountY = r.slider(PARAM_AMOUNT_Y);
    const float edgeStrength = r.slider(PARAM_EDGE_STRENGTH);
    facts->edgeSize = edgeStrength > 0.0f ? r.slider(PARAM_EDGE_SIZE) : 0.0f;
    PF_Err err = r.err;

    facts->moves = false;
    for (int before = samplesPerStep - 1; !err && before >= 0; before--)
    {
        SampleFacts sample{};
        if (!MakeSampleTime(in_data, time, step, before, samplesPerStep, &sample.time))
            sample.time = FrameSample(in_data, time - before * step / samplesPerStep);
        // Read once here and kept with the facts, so each step of each rebuild needn't ask AE again. Each
        // sub-step moves by Transform at its own moment, which can differ from the frame's.
        sample.params = ReadFrameParams(in_data, sample.time, &err);
        facts->moves = facts->moves || MovesTrail(sample.params);
        if (!err && sample.params.inputOpacity > 0.0f)
            ERR(LayerBounds(in_data, extra, request, PARAM_INPUT, sample.time, nextId, &sample.inputRect));
        for (int slot = 0; !err && slot < SLOT_COUNT; slot++)
            if (sources[slot] == MAP_LAYER)
                ERR(LayerBounds(in_data, extra, request, kMapParams[slot], sample.time, nextId, &sample.mapRects[slot]));
        facts->samples.push_back(sample);
    }
    return err;
}

// What a map layer control reads: None reads nothing. The effect's own layer with Effects & Masks means
// the trail: that's the layer as this effect leaves it. AE can't render that for the effect itself, as
// it would loop, so it would otherwise hand over the layer with its masks alone. Checking out the layer
// can't tell None apart from a layer that's empty at the moment, which reads nothing too. If this can't
// be read, it reports `fallback`.
//
// It also says which layer, so a comp-space trail can put the map where that layer is in the comp.
struct MapChoice
{
    int source = MAP_LAYER;
    AEGP_LayerIDVal layer = AEGP_LayerIDVal_NONE; // none if it couldn't be read
};

// The project's handle on this plug-in, which AEGP calls that look at the project ask for. Zero if it
// can't be had.
AEGP_PluginID FeedbackPluginId(PF_InData* in_data, PF_OutData* out_data)
{
    static const AEGP_PluginID id = [&]
    {
        AEGP_SuiteHandler suites(in_data->pica_basicP);
        auto* util = suites.UtilitySuite6();
        AEGP_PluginID registered = 0;
        util->AEGP_RegisterWithAEGP(nullptr, "Feedback", &registered);
        return registered;
    }();
    return id;
}

// AEGP effect and stream handles, each disposed of when it goes out of scope, error or not.
struct EffectRef
{
    AEGP_SuiteHandler& suites;
    AEGP_EffectRefH ref = nullptr;
    ~EffectRef()
    {
        if (ref)
            suites.EffectSuite5()->AEGP_DisposeEffect(ref);
    }
};

struct StreamRef
{
    AEGP_SuiteHandler& suites;
    AEGP_StreamRefH ref = nullptr;
    ~StreamRef()
    {
        if (ref)
            suites.StreamSuite7()->AEGP_DisposeStream(ref);
    }
};

// Which instance of the effect this is: its layer, and where it is among that layer's effects. AE's
// history of an adjustment layer's inputs is the same for every one with the same settings, so their
// images are kept apart by this. Zero if it can't be read.
uint64_t EffectInstance(PF_InData* in_data, PF_OutData* out_data)
{
    try
    {
        const AEGP_PluginID plugin = FeedbackPluginId(in_data, out_data);
        AEGP_SuiteHandler suites(in_data->pica_basicP);
        auto* face = suites.PFInterfaceSuite1();
        auto* layers = suites.LayerSuite9();
        auto* streams = suites.StreamSuite7();
        auto* dynamic = suites.DynamicStreamSuite4();
        AEGP_LayerH layer = nullptr;
        AEGP_LayerIDVal id = AEGP_LayerIDVal_NONE;
        if (!plugin || face->AEGP_GetEffectLayer(in_data->effect_ref, &layer) || !layer || layers->AEGP_GetLayerID(layer, &id))
            return 0;
        // The first control's parent is the effect itself, whose place among the layer's effects is its index.
        A_long index = -1;
        EffectRef effect{suites};
        StreamRef control{suites}, parent{suites};
        if (!face->AEGP_GetNewEffectForEffect(plugin, in_data->effect_ref, &effect.ref) &&
            !streams->AEGP_GetNewEffectStreamByIndex(plugin, effect.ref, 1, &control.ref) &&
            !dynamic->AEGP_GetNewParentStreamRef(plugin, control.ref, &parent.ref))
            dynamic->AEGP_GetStreamIndexInParent(parent.ref, &index);
        return uint64_t(uint32_t(id)) << 32 | uint32_t(index + 2);
    }
    catch (...)
    {
        return 0;
    }
}

// The controls of this effect that an expression drives, by index.
std::vector<int> FindExpressions(PF_InData* in_data, PF_OutData* out_data)
{
    std::vector<int> found;
    try
    {
        const AEGP_PluginID plugin = FeedbackPluginId(in_data, out_data);
        AEGP_SuiteHandler suites(in_data->pica_basicP);
        auto* face = suites.PFInterfaceSuite1();
        auto* streams = suites.StreamSuite7();
        EffectRef effect{suites};
        if (!plugin || face->AEGP_GetNewEffectForEffect(plugin, in_data->effect_ref, &effect.ref) || !effect.ref)
            return found;
        for (int index = 1; index < PARAM_COUNT; index++)
        {
            StreamRef stream{suites};
            if (streams->AEGP_GetNewEffectStreamByIndex(plugin, effect.ref, index, &stream.ref) || !stream.ref)
                continue;
            // Only a control that can vary over time can have an expression, and AE stops with an error
            // if asked about any other: group headings, layer controls and those that can't be keyframed.
            A_Boolean canVary = false, enabled = false;
            if (!streams->AEGP_CanVaryOverTime(stream.ref, &canVary) && canVary &&
                !streams->AEGP_GetExpressionState(plugin, stream.ref, &enabled) && enabled)
                found.push_back(index);
        }
    }
    catch (...)
    {
    }
    return found;
}

// Mixes what a control holds into `h`. Layer controls hold no value of their own and are left out.
void MixValue(uint64_t& h, const PF_ParamDef& d)
{
    const auto mix = [&](double v) { Mix(h, v); };
    switch (d.param_type)
    {
    case PF_Param_FLOAT_SLIDER:
        mix(d.u.fs_d.value);
        break;
    case PF_Param_SLIDER:
        mix(d.u.sd.value);
        break;
    case PF_Param_FIX_SLIDER:
        mix(d.u.fd.value);
        break;
    case PF_Param_ANGLE:
        mix(d.u.ad.value);
        break;
    case PF_Param_CHECKBOX:
        mix(d.u.bd.value);
        break;
    case PF_Param_POPUP:
        mix(d.u.pd.value);
        break;
    case PF_Param_POINT:
        mix(d.u.td.x_value);
        mix(d.u.td.y_value);
        break;
    case PF_Param_POINT_3D:
        mix(d.u.point3d_d.x_value);
        mix(d.u.point3d_d.y_value);
        mix(d.u.point3d_d.z_value);
        break;
    case PF_Param_COLOR:
        mix(d.u.cd.value.alpha);
        mix(d.u.cd.value.red);
        mix(d.u.cd.value.green);
        mix(d.u.cd.value.blue);
        break;
    default:
        break;
    }
}

// AE's history of the effect's controls doesn't change when an expression is added or edited, or when
// what one reads changes, so neither AE's cache of finished frames nor the trail's own images can tell
// a trail made before from one made after, and both go on showing what was made before. So what each
// expression gives is read at every moment of the trail and goes into the fingerprint frames are kept
// with, as where the layer went does. Readings are kept until anything in the project changes, so
// playing on reads only the new moments.
class ExpressionPrints
{
public:
    // `instance` is EffectInstance's: readings are kept for each instance of the effect.
    ExpressionPrints(PF_InData* in_data, PF_OutData* out_data, uint64_t instance, ExpressionReadings& shared)
        : mInData(in_data), mShared(shared), mInstance(instance)
    {
        try
        {
            AEGP_SuiteHandler suites(in_data->pica_basicP);
            auto* render = suites.RenderSuite5();
            mHaveMade = render->AEGP_GetCurrentTimestamp(&mMade) == A_Err_NONE;
        }
        catch (...)
        {
            mHaveMade = false;
        }
        if (mHaveMade && mInstance)
        {
            std::lock_guard<std::mutex> guard(shared.lock);
            auto found = shared.instances.find(mInstance);
            if (found != shared.instances.end() && SameStamp(found->second.made, mMade))
            {
                mParams = found->second.params;
                mKnown = found->second.readings;
                mFound = true;
            }
        }
        if (!mFound)
            mParams = FindExpressions(in_data, out_data);
    }

    ~ExpressionPrints()
    {
        if (!mHaveMade || !mInstance || (mFound && mNew.empty()))
            return;
        // Let go first, so the kept readings can be added to in place if no other call holds them.
        mKnown.reset();
        std::lock_guard<std::mutex> guard(mShared.lock);
        ExpressionReadings::Instance& kept = mShared.instances[mInstance];
        if (!SameStamp(kept.made, mMade) || !kept.readings)
            kept.readings = std::make_shared<ExpressionReadings::Values>();
        else if (kept.readings.use_count() > 1)
            kept.readings = std::make_shared<ExpressionReadings::Values>(*kept.readings);
        // Long sessions on one project mustn't grow this without end.
        if (kept.readings->size() > 1000000)
            kept.readings->clear();
        kept.readings->insert(mNew.begin(), mNew.end());
        kept.made = mMade;
        kept.params = mParams;
    }

    bool active() const { return !mParams.empty(); }
    int count() const { return int(mParams.size()); }

    // What the expressions give over a frame's moments, as ReadFrameFacts samples them.
    uint64_t frame(A_long time, A_long step, int samplesPerStep)
    {
        uint64_t print = kHashStart;
        for (int before = samplesPerStep - 1; before >= 0; before--)
        {
            SampleTime t;
            if (!MakeSampleTime(mInData, time, step, before, samplesPerStep, &t))
                t = FrameSample(mInData, time - before * step / samplesPerStep);
            Mix(print, at(t));
        }
        return print;
    }

    // What the expressions give at a moment.
    uint64_t at(const SampleTime& t)
    {
        if (mParams.empty())
            return 0;
        const uint64_t moment = Moment(t);
        if (mKnown)
            if (auto found = mKnown->find(moment); found != mKnown->end())
                return found->second;
        if (auto found = mNew.find(moment); found != mNew.end())
            return found->second;
        uint64_t print = kHashStart;
        ParamReader r{mInData, t};
        for (int index : mParams)
            r.read(index, [&](const PF_ParamDef& def) { MixValue(print, def); return 0; });
        // Unreadable, so never the same as anything kept.
        if (r.err)
            return Unmatched();
        mNew.emplace(moment, print);
        return print;
    }

private:
    static bool SameStamp(const AEGP_TimeStamp& a, const AEGP_TimeStamp& b)
    {
        return std::memcmp(&a, &b, sizeof(a)) == 0;
    }

    // A moment as one key: its time and that time's scale.
    static uint64_t Moment(const SampleTime& t) { return uint64_t(uint32_t(t.value)) << 32 | uint32_t(t.scale); }

    // A different value each time, which no reading can equal.
    static uint64_t Unmatched()
    {
        static std::atomic<uint64_t> next{1};
        return next++ | 1ull << 63;
    }

    PF_InData* mInData;
    ExpressionReadings& mShared;
    uint64_t mInstance;
    AEGP_TimeStamp mMade{};
    bool mHaveMade = false;
    bool mFound = false;
    std::vector<int> mParams;
    std::shared_ptr<const ExpressionReadings::Values> mKnown;
    ExpressionReadings::Values mNew;
};

// Both map layer controls, read through one effect reference. `fallback` is each one's source if it
// can't be read.
std::array<MapChoice, SLOT_COUNT> ReadMapChoices(PF_InData* in_data, PF_OutData* out_data, const MapSources& fallback)
{
    std::array<MapChoice, SLOT_COUNT> choices;
    for (int slot = 0; slot < SLOT_COUNT; slot++)
        choices[slot].source = fallback[slot];
    A_long picked[SLOT_COUNT];
    bool readPicked[SLOT_COUNT];
    AEGP_LayerParamStage stage[SLOT_COUNT];
    std::fill(std::begin(picked), std::end(picked), A_long(1));
    std::fill(std::begin(readPicked), std::end(readPicked), false);
    std::fill(std::begin(stage), std::end(stage), AEGP_LayerParamStage_SOURCE);
    AEGP_LayerIDVal own = AEGP_LayerIDVal_NONE;
    try
    {
        AEGP_SuiteHandler suites(in_data->pica_basicP);
        auto* face = suites.PFInterfaceSuite1();
        auto* streams = suites.StreamSuite7();
        auto* layers = suites.LayerSuite9();

        const AEGP_PluginID pluginId = FeedbackPluginId(in_data, out_data);
        if (!pluginId)
            return choices;

        EffectRef effect{suites};
        if (face->AEGP_GetNewEffectForEffect(pluginId, in_data->effect_ref, &effect.ref))
            return choices;
        for (int slot = 0; slot < SLOT_COUNT; slot++)
        {
            StreamRef stream{suites};
            if (streams->AEGP_GetNewEffectStreamByIndex(pluginId, effect.ref, kMapParams[slot], &stream.ref))
                continue;
            AEGP_StreamValue2 value{};
            A_Time time = {in_data->current_time, static_cast<A_u_long>(in_data->time_scale)};
            if (!streams->AEGP_GetNewStreamValue(pluginId, stream.ref, AEGP_LTimeMode_CompTime, &time, TRUE, &value))
            {
                picked[slot] = value.val.layer_id;
                readPicked[slot] = true;
                streams->AEGP_DisposeStreamValue(&value);
            }
            streams->AEGP_GetStreamLayerParamStageValue(pluginId, stream.ref, &stage[slot]);
        }

        AEGP_LayerH layer = nullptr;
        if (!face->AEGP_GetEffectLayer(in_data->effect_ref, &layer) && layer)
            layers->AEGP_GetLayerID(layer, &own);
    }
    catch (...)
    {
        return choices;
    }
    for (int slot = 0; slot < SLOT_COUNT; slot++)
    {
        if (!readPicked[slot])
            continue;
        MapChoice& choice = choices[slot];
        const bool self =
            own != AEGP_LayerIDVal_NONE && picked[slot] == own && stage[slot] == AEGP_LayerParamStage_ALL_EFFECTS;
        choice.source = picked[slot] == AEGP_LayerIDVal_NONE ? MAP_NONE : self ? MAP_TRAIL : MAP_LAYER;
        if (choice.source == MAP_LAYER)
            choice.layer = picked[slot];
    }
    return choices;
}

// The layer's first frame, which is where its trail begins, however much of the layer lies before it.
// AE supplies input from before an in-point, so this has to be asked for. If it can't be read, the
// trail reaches back to the layer's start. A layer extended before its start also begins there.
A_long TrailStart(PF_InData* in_data, PF_OutData* out_data)
{
    try
    {
        AEGP_SuiteHandler suites(in_data->pica_basicP);
        auto* face = suites.PFInterfaceSuite1();
        auto* layers = suites.LayerSuite9();

        AEGP_LayerH layer = nullptr;
        A_Time inPoint{};
        if (face->AEGP_GetEffectLayer(in_data->effect_ref, &layer) || !layer ||
            layers->AEGP_GetLayerInPoint(layer, AEGP_LTimeMode_LayerTime, &inPoint) || !inPoint.scale)
            return 0;
        // Reversed, AE's in point is the layer's later end in the comp, where its own time is 0. Its trail
        // starts at the other end, the in point plus its duration, which is negative.
        A_Ratio stretch{};
        A_Time compIn{}, length{}, compStart{}, atStart{};
        if (!layers->AEGP_GetLayerStretch(layer, &stretch) && stretch.num < 0 && stretch.den != 0 &&
            !layers->AEGP_GetLayerInPoint(layer, AEGP_LTimeMode_CompTime, &compIn) && compIn.scale &&
            !layers->AEGP_GetLayerDuration(layer, AEGP_LTimeMode_CompTime, &length) && length.scale)
        {
            const double start = std::min(double(compIn.value) / compIn.scale,
                                          double(compIn.value) / compIn.scale + double(length.value) / length.scale);
            compStart = A_Time{A_long(std::lround(start * compIn.scale)), compIn.scale};
            if (!layers->AEGP_ConvertCompToLayerTime(layer, &compStart, &atStart) && atStart.scale)
                return A_long(std::lround(double(atStart.value) * in_data->time_scale / atStart.scale));
        }
        const double time = double(inPoint.value) * in_data->time_scale / inPoint.scale;
        return std::max<A_long>(A_long(std::lround(time)), 0);
    }
    catch (...)
    {
        return 0;
    }
}

// The comp's frames, in the effect's time. If they can't be read, or don't fall on whole units of it,
// the trail steps by what AE asks for, as it always did.
TrailGrid FindGrid(PF_InData* in_data, PF_OutData* out_data)
{
    const A_long now = in_data->current_time;
    TrailGrid grid{now, now, in_data->time_step, 0.0f};
    try
    {
        AEGP_SuiteHandler suites(in_data->pica_basicP);
        auto* face = suites.PFInterfaceSuite1();
        auto* layers = suites.LayerSuite9();
        auto* comps = suites.CompSuite13();
        AEGP_LayerH layer = nullptr;
        AEGP_CompH comp = nullptr;
        A_Time frame{}, at{}, later{};
        const A_u_long scale = A_u_long(in_data->time_scale);
        // A second of the effect's time later shows how fast its layer runs in the comp.
        if (face->AEGP_GetEffectLayer(in_data->effect_ref, &layer) || !layer ||
            layers->AEGP_GetLayerParentComp(layer, &comp) || comps->AEGP_GetCompFrameDuration(comp, &frame) ||
            frame.value <= 0 || !frame.scale || face->AEGP_ConvertEffectToCompTime(in_data->effect_ref, now, scale, &at) ||
            face->AEGP_ConvertEffectToCompTime(in_data->effect_ref, now + A_long(scale), scale, &later) || !at.scale ||
            !later.scale)
            return grid;
        const double seconds = double(at.value) / at.scale;
        const double compPerEffect = double(later.value) / later.scale - seconds;
        const double frameSeconds = double(frame.value) / frame.scale;
        // Negative where the layer's time runs backwards, and its steps are then negative too.
        if (std::abs(compPerEffect) <= 1e-9)
            return grid;
        const double exact = frameSeconds / compPerEffect * scale;
        const A_long step = A_long(std::lround(exact));
        if (step == 0 || std::abs(exact - double(step)) > 0.01)
            return grid;
        // AE rounds the moments it asks for, so one within a unit or so of a frame is that frame.
        const double frames = seconds / frameSeconds;
        const double whole = std::floor(frames + 1.5 / std::abs(exact));
        const A_long back = A_long(std::lround((frames - whole) * exact));
        grid.step = step;
        grid.last = now - back;
        grid.share = std::abs(back) > 1 ? float(double(back) / double(step)) : 0.0f;
        return grid;
    }
    catch (...)
    {
        return grid;
    }
}

// How far a trail reaches: the frames a rebuild would replay, and the canvas to hold it all.
struct TrailWalk
{
    std::vector<const FrameFacts*> window; // [k] is k frames before the current one
    bool hasArea = false;
    PF_LRect area{};
    A_long focusX = 0, focusY = 0; // the centre of the newest area, which matters most if trimmed
    // With Transform, the trail is clipped to this, the same for every frame, so each one cuts it
    // the same way whichever earlier image it resumes from.
    bool limited = false;
    PF_LRect limit{};
    // Facts this render read itself, kept apart from the run until a history vouches for them.
    std::unordered_map<A_long, FrameFacts> read;
    bool hasProbe = false;
    SampleFacts probe{};
    // A remembered frame's expressions gave something else when it was read, so the run is out of date.
    bool stale = false;
};

// Adds what a walk read to its run, under a history read afterwards that covers it. If another render
// has changed the run meanwhile, only the frames both histories agree on go in.
void KeepFacts(
    PF_InData* in_data,
    const PF_ParamUtilsSuite3& utils,
    FrameFactsMemo& memo,
    const RunView& view,
    const TrailGrid& grid,
    A_long firstTime,
    TrailWalk& walk)
{
    FrameFactsMemo::Run& run = *view.run;
    A_long endExclusive = HistoryEnd(grid.last, grid.step, firstTime);
    {
        std::lock_guard<std::mutex> guard(memo.lock);
        if (walk.read.empty() && endExclusive <= run.endExclusive)
            return;
        endExclusive = std::max(endExclusive, run.endExclusive);
    }
    for (const auto& [time, facts] : walk.read)
        endExclusive = std::max(endExclusive, HistoryEnd(time, grid.step, firstTime));

    PF_State history;
    const bool readable = ReadHistory(in_data, utils, endExclusive, &history);

    std::lock_guard<std::mutex> guard(memo.lock);
    const bool unchanged = run.version == view.version;
    if (!readable || walk.stale)
    {
        // A history that can't be read never matches again, and an out-of-date run mustn't: either way,
        // the next call starts a new run.
        if (unchanged)
        {
            run.endExclusive = 0;
            run.version++;
        }
        return;
    }
    // Copied, as the walk's window still points at its own.
    for (const auto& [time, facts] : walk.read)
        if (unchanged || time < view.trustedEnd)
            run.frames.emplace(time, facts);
    if (!unchanged)
        return;
    run.endExclusive = endExclusive;
    run.history = history;
    run.version++;
    if (!run.hasProbe && walk.hasProbe)
    {
        run.hasProbe = true;
        run.probe = walk.probe;
    }
}

// Walks back until earlier frames have faded to nothing or the trail's start. The canvas is whatever
// this reaches: each input's bounds, plus how far later frames can have spread it. It goes past the
// pre-roll limit, because playing forward can still be carrying what's older.
PF_Err WalkTrail(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    FrameFactsMemo& memo,
    const RunView& view,
    const TrailGrid& grid,
    int samplesPerStep,
    int preRollLimit,
    double invisible,
    const MapSources& sources,
    A_long firstTime,
    const std::function<PF_LRect(const SampleFacts&)>& toCanvas,
    bool capSpread,
    float spreadWidest,
    ExpressionPrints* expressions,
    A_long* nextId,
    TrailWalk* walk)
{
    PF_Err err = PF_Err_NONE;
    const A_long now = grid.last;
    const A_long step = grid.step;
    const float scaleX = ScaleValue(in_data->downsample_x);
    const float scaleY = ScaleValue(in_data->downsample_y);
    const A_long framesBefore = step != 0 ? std::max<A_long>((now - firstTime) / step, 0) : 0;

    double remaining = 1.0;
    float shiftX = 0.0f, shiftY = 0.0f;
    // With spreadWidest, the blur's reach is worked out from its widths, which add frame by frame, to
    // where what it carries is below `invisible`. Its widest part, applied frame after frame, widens
    // as the square root of their number.
    float spreadX = 0.0f, spreadY = 0.0f, blurReachX = 0.0f, blurReachY = 0.0f;
    bool oldest = false;
    // Transform can carry the trail anywhere, and working out where exactly is fragile. So while it moves,
    // the canvas is the layer's own rectangle, plus wherever its input reaches beyond that, and the
    // trail is clipped there. Every frame then shares one canvas, and zooming in can't grow it.
    bool moved = false;
    PF_LRect layerArea{0, 0, A_long(std::ceil(in_data->width * scaleX)), A_long(std::ceil(in_data->height * scaleY))};

    for (int k = 0; !err && !oldest; k++)
    {
        const A_long time = now - k * step;
        const FrameFacts* known = nullptr;
        if (time < view.trustedEnd)
        {
            std::lock_guard<std::mutex> guard(memo.lock);
            auto found = view.run->frames.find(time);
            if (found != view.run->frames.end())
                known = &found->second;
        }
        // With expressions, a remembered frame is only of use if they still give what they did.
        const uint64_t expressed = expressions ? expressions->frame(time, step, samplesPerStep) : 0;
        if (known && known->expressions != expressed)
        {
            known = nullptr;
            walk->stale = true;
        }
        if (!known)
        {
            FrameFacts read{};
            ERR(ReadFrameFacts(in_data, extra, request, time, step, samplesPerStep, sources, nextId, &read));
            if (err)
                break;
            read.expressions = expressed;
            known = &walk->read.emplace(time, std::move(read)).first->second;
        }
        // Nothing changes a frame's facts once they're in, and the maps keep them in place.
        const FrameFacts& facts = *known;
        if (k <= preRollLimit)
            walk->window.push_back(&facts);

        // Edges keep finding what's there and adding it back, so a trail with them never fades. Nor
        // does one where an Alpha mask is dark.
        if (facts.edgeSize <= 0.0f && sources[SLOT_ALPHA_MASK] == MAP_NONE)
            remaining *= facts.alpha;
        oldest = remaining < invisible || k >= framesBefore;

        // The oldest frame in the window starts from blank, so only its final moment is used.
        const size_t firstSample = oldest ? facts.samples.size() - 1 : 0;
        bool hasFrameArea = false, hasMap = false;
        PF_LRect frameArea{}, frameInLayer{};
        for (size_t i = firstSample; i < facts.samples.size(); i++)
        {
            const SampleFacts& sample = facts.samples[i];
            if (!IsEmpty(sample.inputRect))
            {
                const PF_LRect placed = toCanvas(sample);
                frameArea = Union(frameArea, placed);
                frameInLayer = Union(frameInLayer, sample.inputRect);
                hasFrameArea = true;
                if (!walk->hasProbe)
                {
                    walk->hasProbe = true;
                    walk->probe = sample;
                }
            }
            hasMap = hasMap || (sources[SLOT_DISPLACE] == MAP_LAYER && !IsEmpty(sample.mapRects[SLOT_DISPLACE]));
        }

        const float sigmaX = BlurSigma(facts.blurRadius) * scaleX;
        const float sigmaY = BlurSigma(facts.blurRadius) * scaleY;
        if (spreadWidest > 0.0f)
        {
            spreadX += sigmaX;
            spreadY += sigmaY;
            const float widest = spreadWidest * std::sqrt(float(k + 1));
            blurReachX = SpreadReach(spreadX, widest, invisible);
            blurReachY = SpreadReach(spreadY, widest, invisible);
        }
        else
        {
            // The blur's reach adds up frame by frame. What its tails carry further is too faint to keep.
            blurReachX += kSpreadReach * sigmaX;
            blurReachY += kSpreadReach * sigmaY;
        }
        if (sources[SLOT_DISPLACE] == MAP_TRAIL || hasMap)
        {
            shiftX += std::abs(facts.amountX) * scaleX;
            shiftY += std::abs(facts.amountY) * scaleY;
        }
        shiftX += EdgeReach(facts.edgeSize * scaleX);
        shiftY += EdgeReach(facts.edgeSize * scaleY);
        moved = moved || facts.moves;

        if (hasFrameArea)
        {
            layerArea = Union(layerArea, frameInLayer);

            // How far the blur, displacement and edges reach, plus a pixel for displacement's filtering.
            const float limitX = capSpread ? kMaxSpread * scaleX : 1e8f, limitY = capSpread ? kMaxSpread * scaleY : 1e8f;
            const A_long padX = A_long(std::ceil(std::min(shiftX + blurReachX, limitX))) + 1;
            const A_long padY = A_long(std::ceil(std::min(shiftY + blurReachY, limitY))) + 1;
            frameArea.left -= padX;
            frameArea.top -= padY;
            frameArea.right += padX;
            frameArea.bottom += padY;
            if (!walk->hasArea)
            {
                walk->focusX = (frameArea.left + frameArea.right) / 2;
                walk->focusY = (frameArea.top + frameArea.bottom) / 2;
            }
            walk->area = Union(walk->area, frameArea);
            walk->hasArea = true;
        }
    }

    if (moved && walk->hasArea)
    {
        walk->area = layerArea;
        walk->limited = true;
        walk->limit = layerArea;
    }
    return err;
}

// The part of the layer that sees the comp-space canvas, which is where its output can have anything:
// the canvas's corners taken back through the layer's move, and the layer itself. A layer scaled far
// down can see far further than it's worth drawing, so this is kept to the longest canvas side.
PF_LRect LayerViewOfComp(const Projective& layerToComp, const PF_LRect& canvas, const PF_LRect& layer)
{
    const A_long focusX = (layer.left + layer.right) / 2, focusY = (layer.top + layer.bottom) / 2;
    PF_LRect view{focusX - kMaxCanvasSide / 2, focusY - kMaxCanvasSide / 2, focusX + kMaxCanvasSide / 2,
                  focusY + kMaxCanvasSide / 2};
    Projective compToLayer;
    PF_LRect seen;
    if (layerToComp.inverse(&compToLayer) && MovedBounds(compToLayer, canvas, &seen))
        view = seen;
    view = Union(view, layer);
    TrimAxis(&view.left, &view.right, focusX, kMaxCanvasSide);
    TrimAxis(&view.top, &view.bottom, focusY, kMaxCanvasSide);
    return view;
}

// Keeps a canvas to `maxPixels`, which this machine's GPU can hold, trimming both sides alike about the
// focus. What lies further out is dropped.
void LimitCanvas(PF_LRect* area, A_long focusX, A_long focusY, size_t maxPixels)
{
    const double w = double(area->right - area->left), h = double(area->bottom - area->top);
    if (w <= 0.0 || h <= 0.0 || w * h <= double(maxPixels))
        return;
    const double shrink = std::sqrt(double(maxPixels) / (w * h));
    TrimAxis(&area->left, &area->right, focusX, std::max<A_long>(A_long(w * shrink), 1));
    TrimAxis(&area->top, &area->bottom, focusY, std::max<A_long>(A_long(h * shrink), 1));
}

// Rounding out means a slowly moving trail reuses the same canvas for many frames.
void SnapCanvas(const TrailWalk& walk, PF_LRect* area)
{
    area->left = (area->left >> kCanvasSnapBits) << kCanvasSnapBits;
    area->top = (area->top >> kCanvasSnapBits) << kCanvasSnapBits;
    area->right = ((area->right + (1 << kCanvasSnapBits) - 1) >> kCanvasSnapBits) << kCanvasSnapBits;
    area->bottom = ((area->bottom + (1 << kCanvasSnapBits) - 1) >> kCanvasSnapBits) << kCanvasSnapBits;
    TrimAxis(&area->left, &area->right, walk.focusX, kMaxCanvasSide);
    TrimAxis(&area->top, &area->bottom, walk.focusY, kMaxCanvasSide);
    // After rounding out, which would otherwise keep a varying margin beyond the limit.
    if (walk.limited)
    {
        *area = Intersect(*area, walk.limit);
        if (IsEmpty(*area))
            *area = walk.limit;
    }
}

// Requests the map layers' pixels at a sample's moment.
// What of each map layer a step reads, in its own pixels: the part over the canvas.
using MapArea = std::function<PF_LRect(int slot, const SampleTime& time)>;

PF_Err CheckOutMaps(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    const FetchContext& fetch,
    const MapArea& mapArea,
    const SampleFacts& sample,
    A_long* nextId,
    SubstepCheckout* sub)
{
    PF_Err err = PF_Err_NONE;
    for (int slot = 0; !err && slot < SLOT_COUNT; slot++)
        if (!IsEmpty(sample.mapRects[slot]))
        {
            SubstepCheckout::Map& map = sub->maps[slot];
            map.id = (*nextId)++;
            ERR(CheckoutOrShare(in_data, extra, request, fetch, kMapParams[slot], sample.time, mapArea(slot, sample.time),
                                map.id, &map.rect, &map.fetch));
        }
    return err;
}

// Requests the pixels of the frames that need computing.
PF_Err CheckOutFrames(
    PF_InData* in_data,
    PF_PreRenderExtra* extra,
    const PF_RenderRequest& request,
    const FetchContext& fetch,
    const MapArea& mapArea,
    const TrailWalk& walk,
    int frames,
    A_long* nextId,
    PreRenderData* data)
{
    PF_Err err = PF_Err_NONE;
    const bool startsBlank = !data->resume.buffer;
    for (int k = 0; !err && k < frames; k++)
    {
        const FrameFacts& facts = *walk.window[k];
        FrameCheckout frame{};
        frame.time = data->grid.last - k * data->grid.step;

        // A rebuild's oldest frame starts from blank, so only its final moment is used.
        const size_t firstSample = startsBlank && k == frames - 1 ? facts.samples.size() - 1 : 0;
        for (size_t i = firstSample; !err && i < facts.samples.size(); i++)
        {
            const SampleFacts& sample = facts.samples[i];
            SubstepCheckout sub{};
            sub.time = sample.time;
            sub.params = sample.params;
            if (k == 0 && i + 1 == facts.samples.size() && !data->grid.between())
            {
                // Checked out either way, as AE needs, but only used while some of the layer goes in.
                sub.inputId = data->currentInputId;
                if (sample.params.inputOpacity > 0.0f)
                    sub.inputRect = data->currentInputRect;
            }
            else if (!IsEmpty(sample.inputRect))
            {
                sub.inputId = (*nextId)++;
                ERR(CheckoutOrShare(in_data, extra, request, fetch, PARAM_INPUT, sample.time, kWholeLayer, sub.inputId,
                                    &sub.inputRect, &sub.inputFetch));
            }
            ERR(CheckOutMaps(in_data, extra, request, fetch, mapArea, sample, nextId, &sub));
            frame.substeps.push_back(sub);
        }
        data->frames.push_back(std::move(frame));
    }
    return err;
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

    const A_long firstTime = TrailStart(in_data, out_data);

    // Every earlier frame affects this one, so AE's cached copy is only valid if they're unchanged.
    {
        PF_State history;
        // AE requires this call on every PreRender. With no earlier frames, the zeroed state goes in.
        ReadHistory(in_data, utils, step > 0 ? now : HistoryEnd(now, step, firstTime), &history);
        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(history), &history));

        // Trimming changes the trail, which the history doesn't show.
        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(firstTime), &firstTime));

        // Each frame is a step further on, even where nothing in the history changes from one frame to
        // the next. Without this, AE shows a still layer's first trailing frame for every later one.
        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(now), &now));

        // AE's disk cache outlives the plugin, so frames saved by a different build mustn't be reused.
        static const char kBuild[] = FEEDBACK_BUILD_STAMP;
        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(kBuild), kBuild));
#ifdef FEEDBACK_LOG
        // Tests set this for each render, so none is served frames an earlier one saved.
        static const std::string salt = [] {
            const char* given = getenv("FEEDBACK_GUID_SALT");
            Log("guid salt=%s", given ? given : "");
            return std::string(given ? given : "");
        }();
        if (!salt.empty())
            ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, A_u_long(salt.size()), salt.data()));
#endif
    }

    PF_RenderRequest request = extra->input->output_request;
    request.channel_mask = PF_ChannelMask_ARGB;

    auto data = std::make_unique<PreRenderData>();
    A_long nextId = 0;
    data->currentInputId = nextId++;

    // Without a GPU, the layer only passes through, so its own pixels are all that matter. AE says
    // which it will use in what_gpu, which can be NONE even when there's GPU data.
    auto* metal = extra->input->what_gpu == PF_GPU_Framework_METAL
                      ? const_cast<MetalData*>(static_cast<const MetalData*>(extra->input->gpu_data))
                      : nullptr;
    if (!metal)
    {
        const SampleTime time = FrameSample(in_data, now);
        ERR(CheckoutWholeLayer(in_data, extra, request, PARAM_INPUT, time, data->currentInputId, &data->currentInputRect));
        extra->output->max_result_rect = data->currentInputRect;
        extra->output->result_rect = Intersect(extra->input->output_request.rect, data->currentInputRect);
        data->outputRect = extra->output->result_rect;
    }
    else
    {
        int preRollLimit = INT_MAX; // unlimited, as Limit Pre-Roll is by default
        double invisible = InvisibleContribution(PRECISION_8_BIT);
        {
            ParamReader r{in_data, FrameSample(in_data, now)};
            data->samplesPerStep = r.integer(PARAM_SUBSTEPS) + 1;
            invisible = InvisibleContribution(r.popup(PARAM_PRECISION));
            data->filter = std::clamp(r.popup(PARAM_SAMPLING), int(FILTER_BILINEAR), int(FILTER_NEAREST));
            if (r.checkbox(PARAM_LIMIT_PREROLL))
                preRollLimit = r.integer(PARAM_PREROLL_LIMIT);
            ERR(r.err);
        }
        data->grid = FindGrid(in_data, out_data);
        const TrailGrid& grid = data->grid;
        // The trail stays where the layer left it in the comp. Layers AE already draws in comp space
        // need nothing moved, but share the comp's canvas too: at reduced resolution AE sometimes
        // reports their bounds at full scale, and a canvas built around those can miss the comp
        // altogether. A 3D layer's trail stays with the layer, which AE puts through the camera after
        // this effect.
        LayerMotion motion(in_data, out_data);
        const CompWatch watch(in_data, out_data);
        // Which instance of the effect this is, which AEGP takes a while to tell.
        const uint64_t instance = EffectInstance(in_data, out_data);
        ExpressionPrints expressions(in_data, out_data, instance, metal->expressions);
        data->watchesComp = watch.active();
        if (data->watchesComp)
            watch.timestamp(&data->made);
        data->canvas.compSpace = motion.usable();
        if (data->watchesComp)
            data->canvas.owner = instance;
        // A missing map displaces nothing, and a missing mask masks nothing.
        MapSources fallback;
        fallback.fill(MAP_NONE);
        fallback[SLOT_DISPLACE] = MAP_LAYER;
        const auto choices = ReadMapChoices(in_data, out_data, fallback);
        // A map layer goes where it is in the comp. If that can't be read, it goes where this layer is.
        std::optional<LayerMotion> mapMotions[SLOT_COUNT];
        for (int slot = 0; slot < SLOT_COUNT; slot++)
        {
            data->sources[slot] = choices[slot].source;
            if (data->canvas.compSpace && choices[slot].layer != AEGP_LayerIDVal_NONE)
            {
                mapMotions[slot].emplace(in_data, out_data, choices[slot].layer);
                if (!mapMotions[slot]->usable())
                    mapMotions[slot].reset();
            }
        }

        const FrameFactsMemo::Key factsKey{
            grid.step,
            static_cast<A_u_long>(in_data->time_scale),
            in_data->downsample_x,
            in_data->downsample_y,
            data->samplesPerStep,
            data->canvas.owner};
        RunView view;
        if (!err)
            view = FindFacts(in_data, extra, request, utils, metal->memo, factsKey, &nextId, &err);

        // Where each moment's input lands on the canvas: in the layer, where it is; in comp space, where
        // the layer was in the comp then. A layer AE draws in comp space can come with its bounds at full
        // scale at reduced resolution, so both readings are covered.
        const double resolutionX = ScaleValue(in_data->downsample_x), resolutionY = ScaleValue(in_data->downsample_y);
        const PF_LRect compBox{A_long(std::floor(-kCompMargin * resolutionX)), A_long(std::floor(-kCompMargin * resolutionY)),
                               A_long(std::ceil((motion.compWidth() + kCompMargin) * resolutionX)),
                               A_long(std::ceil((motion.compHeight() + kCompMargin) * resolutionY))};
        const auto toCanvas = [&](const SampleFacts& sample) -> PF_LRect
        {
            if (!data->canvas.compSpace || IsEmpty(sample.inputRect))
                return sample.inputRect;
            Projective toComp;
            PF_LRect placed;
            if (!motion.layerToComp(sample.time.value, sample.time.scale, &toComp) ||
                !MovedBounds(AtResolution(toComp, in_data), sample.inputRect, &placed))
                return compBox;
            if (motion.drawnInComp() && (resolutionX != 1.0 || resolutionY != 1.0))
            {
                const PF_LRect& r = sample.inputRect;
                const PF_LRect reduced{A_long(std::floor(r.left * resolutionX)), A_long(std::floor(r.top * resolutionY)),
                                       A_long(std::ceil(r.right * resolutionX)), A_long(std::ceil(r.bottom * resolutionY))};
                PF_LRect other;
                if (!MovedBounds(AtResolution(toComp, in_data), reduced, &other))
                    return compBox;
                placed = Union(placed, other);
            }
            return placed;
        };

        if (data->canvas.compSpace)
            data->spreadWidest = float(std::max(compBox.right - compBox.left, compBox.bottom - compBox.top)) * 0.5f;
        TrailWalk walk;
        if (!err)
            ERR(WalkTrail(in_data, extra, request, metal->memo, view, grid, data->samplesPerStep, preRollLimit, invisible,
                          data->sources, firstTime, toCanvas, !data->canvas.compSpace, data->spreadWidest,
                          expressions.active() ? &expressions : nullptr, &nextId, &walk));

        // A failed walk keeps nothing, as what it read may not match any history.
        if (!err)
            KeepFacts(in_data, utils, metal->memo, view, grid, firstTime, walk);

        // Between frames, the layer at the moment asked for goes over the trail, which needs room for it.
        FrameFacts betweenFacts{};
        if (!err && grid.between())
        {
            ERR(ReadFrameFacts(in_data, extra, request, now, grid.step, 1, data->sources, &nextId, &betweenFacts));
            const PF_LRect own = err ? PF_LRect{} : toCanvas(betweenFacts.samples.back());
            if (!walk.limited && !IsEmpty(own))
            {
                walk.area = Union(walk.area, own);
                walk.hasArea = true;
            }
        }

        const size_t pixelLimit = metal->renderer->cache().canvasPixelLimit();
        PF_LRect area = walk.hasArea ? walk.area : extra->input->output_request.rect;
        SnapCanvas(walk, &area);
        LimitCanvas(&area, walk.focusX, walk.focusY, pixelLimit);
        if (data->canvas.compSpace)
        {
            // Where the trail can be, within the comp and a margin: what drifts further out is dropped.
            // While Transform moves the trail, which can carry it anywhere, the whole of that.
            const double sx = resolutionX, sy = resolutionY;
            const PF_LRect fitted = walk.hasArea && !walk.limited && !WholeCanvas() ? Intersect(area, compBox) : PF_LRect{};
            area = IsEmpty(fitted) ? compBox : fitted;
            // A comp too large to hold keeps the part around the layer, as it is now.
            Projective toComp;
            double focusX = 0.5 * (area.left + area.right), focusY = 0.5 * (area.top + area.bottom);
            if (motion.layerToComp(now, A_u_long(in_data->time_scale), &toComp) &&
                toComp.apply(0.5 * in_data->width, 0.5 * in_data->height, &focusX, &focusY))
            {
                focusX *= sx;
                focusY *= sy;
            }
            const auto snap = [](double v) { return A_long(std::lround(v)) >> kCanvasSnapBits << kCanvasSnapBits; };
            LimitCanvas(&area, snap(focusX), snap(focusY), pixelLimit);
        }
        extra->output->max_result_rect = area;
        extra->output->result_rect = Intersect(extra->input->output_request.rect, area);

        data->outputRect = extra->output->result_rect;
        data->canvas.left = area.left;
        data->canvas.top = area.top;
        data->canvas.width = area.right - area.left;
        data->canvas.height = area.bottom - area.top;
        data->canvas.downsampleX = in_data->downsample_x;
        data->canvas.downsampleY = in_data->downsample_y;
        data->canvas.firstTime = firstTime;

        // A step reads a map only over the canvas, so AE renders no more of it than that. In the layer,
        // the canvas's own pixels; in comp space, the canvas taken back to where the map layer is then,
        // or where this one is if that can't be read. Two pixels more each way for filtering.
        const PF_LRect canvasRect{area.left - 2, area.top - 2, area.right + 2, area.bottom + 2};
        const MapArea mapArea = [&](int slot, const SampleTime& time) -> PF_LRect
        {
            if (WholeCanvas())
                return kWholeLayer;
            if (!data->canvas.compSpace)
                return canvasRect;
            Projective toComp, fromComp;
            const bool known = mapMotions[slot] ? mapMotions[slot]->layerToComp(time.value, time.scale, &toComp)
                                                : motion.layerToComp(time.value, time.scale, &toComp);
            PF_LRect seen;
            if (!known || !AtResolution(toComp, in_data).inverse(&fromComp) || !MovedBounds(fromComp, canvasRect, &seen))
                return kWholeLayer;
            return PF_LRect{seen.left - 2, seen.top - 2, seen.right + 2, seen.bottom + 2};
        };

        // The moment shown.
        const SampleFacts* own = err ? nullptr : grid.between() ? &betweenFacts.samples.back() : &walk.window[0]->samples.back();
        if (!err)
        {
            data->opacity = own->params.inputOpacity;
            if ((data->samplesPerStep > 1 || grid.between()) && data->opacity > 0.0f)
            {
                ParamReader r{in_data, FrameSample(in_data, now)};
                const int mode = r.popup(PARAM_COMPOSITE);
                ERR(r.err);
                if (BuildsUpInput(mode))
                    data->overlayMode = mode;
            }
            ERR(CheckoutWholeLayer(in_data, extra, request, PARAM_INPUT, own->time, data->currentInputId, &data->currentInputRect));
        }

        data->fingerprinted = data->canvas.compSpace || expressions.active();
        std::function<uint64_t(const SampleTime&)> expressionsAt;
        if (expressions.active())
            expressionsAt = [&expressions](const SampleTime& t) { return expressions.at(t); };
        MotionPrints prints(in_data, motion, mapMotions, firstTime, grid.step, int(walk.window.size()),
                            data->samplesPerStep, expressionsAt);
        MotionAt motionAt;
        if (!err && data->fingerprinted)
        {
            motionAt = [&prints](A_long t) { return prints.at(t); };
            // Where the layer went, and what expressions gave, change the frame, though AE's history
            // doesn't show either. Between frames, so does where the layer is now.
            const uint64_t moved = prints.at(grid.last);
            const uint64_t movedSince = grid.between() && data->canvas.compSpace ? motion.fingerprint(now) : 0;
            ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(moved), &moved));
            ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(movedSince), &movedSince));
        }
        // With the trail in comp space, the output is the layer's view of the comp, around the layer.
        if (!err && data->canvas.compSpace && !motion.layerToComp(now, A_u_long(in_data->time_scale), &data->layerToCompNow))
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        if (!err && data->canvas.compSpace)
        {
            const PF_LRect output = LayerViewOfComp(AtResolution(data->layerToCompNow, in_data), area, data->currentInputRect);
            extra->output->max_result_rect = output;
            extra->output->result_rect = Intersect(extra->input->output_request.rect, output);
            data->outputRect = extra->output->result_rect;
        }

        // Layers other renders have fetched at the moments needed are used rather than fetched again. Not
        // on an adjustment layer: its input is what's beneath it, which AE's state doesn't follow.
        const FetchContext fetchContext{&utils, &metal->renderer->cache(),
                                        data->watchesComp ? 0 : instance};
        if (!err)
            MakeFetchTag(in_data, request, fetchContext, PARAM_INPUT, own->time, kWholeLayer, &data->currentInputFetch);

        // Only frames newer than the newest usable image need their pixels.
        int resumeK = -1;
        if (!err)
        {
            const ValidityCheck validity(in_data, utils, data->canvas, grid.step, motionAt,
                                         data->watchesComp ? &watch : nullptr, data->made);
            resumeK = FindResume(metal->renderer->cache(), validity, data->canvas, grid.last, grid.step,
                                 int(walk.window.size()), &data->resume);
            ERR(CheckOutFrames(in_data, extra, request, fetchContext, mapArea, walk, resumeK >= 0 ? resumeK : int(walk.window.size()),
                               &nextId, data.get()));
        }
        if (!err && grid.between())
        {
            const SampleFacts& sample = betweenFacts.samples.back();
            SubstepCheckout sub{};
            sub.time = sample.time;
            sub.params = sample.params;
            sub.inputId = data->currentInputId;
            if (sample.params.inputOpacity > 0.0f)
                sub.inputRect = data->currentInputRect;
            ERR(CheckOutMaps(in_data, extra, request, fetchContext, mapArea, sample, &nextId, &sub));
            data->between = sub;
        }
        // Where the layer is at each moment to be computed, and what each frame's image is kept with.
        auto place = [&](SubstepCheckout& sub)
        {
            if (!err && !motion.layerToComp(sub.time.value, sub.time.scale, &sub.layerToComp))
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            for (int slot = 0; slot < SLOT_COUNT; slot++)
            {
                SubstepCheckout::Map& map = sub.maps[slot];
                map.toComp = sub.layerToComp;
                if (!err && mapMotions[slot] && !IsEmpty(map.rect) &&
                    !mapMotions[slot]->layerToComp(sub.time.value, sub.time.scale, &map.toComp))
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
        };
        for (FrameCheckout& frame : data->frames)
        {
            if (err)
                break;
            if (data->fingerprinted)
                frame.motion = prints.at(frame.time);
            if (data->canvas.compSpace)
                for (SubstepCheckout& sub : frame.substeps)
                    place(sub);
        }
        if (!err && data->canvas.compSpace && data->between)
            place(*data->between);

        Log("prerender frame=%d window=%d samples=%d read=%d resume=%d checkouts=%d canvas=%dx%d at %d,%d layer=%dx%d "
            "downsample=%.2f ms=%.2f time=%d step=%d scale=%u first=%d adjust=%d last=%d every=%d share=%.3f expressions=%d",
            step > 0 ? now / step : 0,
            int(walk.window.size()) - 1,
            data->samplesPerStep,
            int(walk.read.size()),
            resumeK,
            nextId,
            data->canvas.width,
            data->canvas.height,
            data->canvas.left,
            data->canvas.top,
            int(in_data->width),
            int(in_data->height),
            ScaleValue(in_data->downsample_x),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - preRenderStart).count(),
            now,
            step,
            unsigned(in_data->time_scale),
            firstTime,
            int(data->watchesComp),
            grid.last,
            grid.step,
            grid.share,
            expressions.count());
    }

    if (!err)
    {
        extra->output->pre_render_data = data.release();
        extra->output->delete_pre_render_data_func = DisposePreRenderData;
    }
    return err;
}

// What the steps of one render share.
struct GpuRender
{
    PF_InData* in_data;
    PF_SmartRenderExtra* extra;
    const PF_GPUDeviceSuite1* gpu;
    const PF_WorldSuite2* worlds;
    FeedbackRenderer& renderer;
    id<MTLCommandQueue> queue;
    const PreRenderData& data;
    PF_EffectWorld* currentInput;

    StepBatch batch;
    std::vector<A_long> checkedOut; // what the queued steps read, which AE must keep until they've run

    // Fetched layers copied for other renders, kept once the batch that copies them has run.
    struct Kept
    {
        FetchKey key;
        PF_State state;
        FetchedPixels pixels;
    };
    std::vector<Kept> keep;

    // Fetched ahead, newest first, by FetchTag; and what the lookups need.
    std::map<const FetchTag*, FetchedPixels> prefetched;
    const PF_ParamUtilsSuite3* utils = nullptr;

    // Pixels for a tag that this render fetched ahead, or another render has kept since PreRender looked.
    bool found(const FetchTag& tag, FetchedPixels* pixels)
    {
        if (tag.shared.buffer)
        {
            *pixels = tag.shared;
            shared++;
            return true;
        }
        auto ahead = prefetched.find(&tag);
        if (ahead != prefetched.end())
        {
            *pixels = ahead->second;
            return true;
        }
        if (!tag.tagged || !utils || !FindFetched(in_data, *utils, renderer.cache(), tag, pixels))
            return false;
        shared++;
        return true;
    }
    int queuedSteps = 0;

    // Timings for the log.
    int fetched = 0, mapsFetched = 0, shared = 0;
    double fetchMs = 0.0, mapMs = 0.0, gpuMs = 0.0; // fetchMs includes mapMs

    // Reads a checked-out world as a placed image. A missing world means nothing there, not failure.
    PF_Err place(PF_EffectWorld* world, const PF_LRect& rect, PlacedImage* image) const
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
            image->offsetX = rect.left - data.canvas.left;
            image->offsetY = rect.top - data.canvas.top;
            image->width = world->width;
            image->height = world->height;
        }
        return e;
    }

    // Places pixels another render fetched, or this one kept.
    PlacedImage placeShared(const FetchedPixels& pixels) const
    {
        PlacedImage image{};
        image.buffer = pixels.buffer;
        image.pitch = pixels.pitch;
        image.offsetX = pixels.rect.left - data.canvas.left;
        image.offsetY = pixels.rect.top - data.canvas.top;
        image.width = pixels.width;
        image.height = pixels.height;
        return image;
    }

    // Copies what AE gave into a buffer of this effect's own, in the batch, and uses that instead, so
    // other renders can have it once the batch has run.
    void keepFetched(const FetchTag& tag, const PF_LRect& rect, PlacedImage* image)
    {
        if (!tag.keep || !image->buffer)
            return;
        id<MTLBuffer> copy = renderer.copyPixels(batch, image->buffer, image->pitch, image->height);
        if (!copy)
            return;
        image->buffer = copy;
        keep.push_back({tag.key, tag.state, FetchedPixels{copy, rect, image->pitch, image->width, image->height}});
    }
};

double MsSince(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// Runs the queued steps and gives AE back what they read.
PF_Err FlushSteps(GpuRender& render)
{
    auto start = std::chrono::steady_clock::now();
    const bool ok = render.renderer.flush(render.batch);
    render.gpuMs += MsSince(start);
    if (ok)
        for (const auto& k : render.keep)
            render.renderer.cache().publishFetch(k.key, k.state, k.pixels);
    render.keep.clear();
    for (A_long id : render.checkedOut)
        render.extra->cb->checkin_layer_pixels(render.in_data->effect_ref, id);
    render.checkedOut.clear();
    render.queuedSteps = 0;
    return ok ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED;
}

// Queues one step of the loop. Sets `next` to the image it will produce.
// `share` is how much of a frame's change it makes. At Pixel Sampling's Pixel, `carry` is what the trail's
// move still owes, which the step takes on and updates.
PF_Err ComputeSample(
    GpuRender& render,
    const SubstepCheckout& sub,
    float share,
    bool frameEnd,
    bool isCurrent,
    id<MTLBuffer> image,
    TrailCarry* carry,
    __strong id<MTLBuffer>* next)
{
    PF_Err err = PF_Err_NONE;
    PF_InData* in_data = render.in_data;
    PF_SmartRenderExtra* extra = render.extra;
    if (!isCurrent)
        ERR(PF_ABORT(in_data));

    const WholePixels pixels{ScaleValue(render.data.canvas.downsampleX), ScaleValue(render.data.canvas.downsampleY), carry};
    FrameParams params =
        DivideIntoSteps(sub.params, share, frameEnd, render.data.filter == FILTER_NEAREST ? &pixels : nullptr);
    params.filter = render.data.filter;
    params.spreadWidest = render.data.spreadWidest;
    std::copy(render.data.sources.begin(), render.data.sources.end(), params.sources);

    // The input, then each map: what it is, how another render may have fetched it, and where its layer
    // is in the comp. The current input is the one AE already gave.
    struct Source
    {
        A_long id;
        PF_LRect rect;
        const FetchTag* fetch;
        const Projective* toComp;
        PlacedImage* placed;
        PF_EffectWorld* world = nullptr; // checked out from AE, or the current input
        bool shared = false;            // or another render's, in `pixels`
        FetchedPixels pixels{};
    };
    PlacedImage input{}, maps[SLOT_COUNT]{};
    Source sources[1 + SLOT_COUNT] = {
        {sub.inputId, sub.inputRect, isCurrent ? &render.data.currentInputFetch : &sub.inputFetch, &sub.layerToComp, &input}};
    for (int slot = 0; slot < SLOT_COUNT; slot++)
        sources[1 + slot] = {sub.maps[slot].id, sub.maps[slot].rect, &sub.maps[slot].fetch, &sub.maps[slot].toComp, &maps[slot]};
    const auto checkedOut = [&](int i) { return sources[i].world && !(i == 0 && isCurrent); };

    auto fetchStart = std::chrono::steady_clock::now();
    for (int i = 0; !err && i < 1 + SLOT_COUNT; i++)
    {
        Source& source = sources[i];
        // At Opacity 0, the layer adds nothing, as if it were empty.
        if (i == 0 && isCurrent)
            source.world = sub.params.inputOpacity > 0.0f ? render.currentInput : nullptr;
        else if (!IsEmpty(source.rect) && !(source.shared = render.found(*source.fetch, &source.pixels)))
        {
            const auto started = std::chrono::steady_clock::now();
            ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, source.id, &source.world));
            if (i == 0)
                render.fetched++;
            else
            {
                render.mapsFetched++;
                render.mapMs += MsSince(started);
            }
        }
    }
    render.fetchMs += MsSince(fetchStart);

    for (Source& source : sources)
    {
        if (err)
            break;
        if (source.shared)
        {
            *source.placed = render.placeShared(source.pixels);
            continue;
        }
        ERR(render.place(source.world, source.rect, source.placed));
        if (!err)
            render.keepFetched(*source.fetch, source.rect, source.placed);
    }

    // In comp space, the input and maps are where their layers are at this moment, and Centre goes with it.
    // Pixels another render fetched cover what it asked for, which can be more than this one did.
    if (!err && render.data.canvas.compSpace)
    {
        for (const Source& source : sources)
            if (!err && source.placed->buffer &&
                !WarpOntoCanvas(source.placed, *source.toComp, source.shared ? source.pixels.rect : source.rect,
                                render.data.canvas, in_data))
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;

        const Projective toComp = AtResolution(sub.layerToComp, in_data);
        double centreX, centreY;
        if (!err && toComp.apply(params.centreX, params.centreY, &centreX, &centreY))
        {
            params.centreX = float(centreX);
            params.centreY = float(centreY);
        }
    }

    if (!err)
    {
        auto gpuStart = std::chrono::steady_clock::now();
        *next = render.renderer.step(render.batch, render.data.canvas, image, input, maps, params);
        render.gpuMs += MsSince(gpuStart);
        render.queuedSteps++;
    }

    for (int i = 0; i < 1 + SLOT_COUNT; i++)
        if (checkedOut(i))
            render.checkedOut.push_back(sources[i].id);
    return err;
}

// How many of the newest frames a render fetches before computing any.
const int kFramesFetchedAhead = 8;

// Fetches the layers for the newest `frames` frames to compute, newest first, before any are
// computed, which needs the oldest first. AE renders several frames at once, which would otherwise
// fetch the same moments at the same time; this way each fetches its own newest moment while the
// render of the frame before fetches the one before that, and finds it kept when it gets there. A
// moment another render is fetching is left until last, and fetched here only if it still isn't kept.
// Each is copied and handed straight back to AE.
PF_Err FetchAhead(GpuRender& render, int frames, bool startsBlank)
{
    PF_Err err = PF_Err_NONE;
    PF_InData* in_data = render.in_data;
    const PreRenderData& data = render.data;
    FeedbackCache& cache = render.renderer.cache();

    struct Need
    {
        const FetchTag* tag;
        A_long id;
        PF_LRect rect;
        bool map;
    };
    std::vector<Need> needs;
    for (int k = 0; k < frames && k < int(data.frames.size()); k++)
    {
        const FrameCheckout& frame = data.frames[size_t(k)];
        for (int s = int(frame.substeps.size()) - 1; s >= 0; s--)
        {
            const SubstepCheckout& sub = frame.substeps[size_t(s)];
            const bool isCurrent = k == 0 && s == int(frame.substeps.size()) - 1 && !data.between;
            // A rebuild from blank skips steps until there's an input; their maps aren't needed.
            if (startsBlank && IsEmpty(sub.inputRect) && !isCurrent)
                continue;
            const auto add = [&](const FetchTag& tag, A_long id, const PF_LRect& rect, bool map)
            {
                if (tag.tagged && !tag.shared.buffer && !IsEmpty(rect))
                    needs.push_back({&tag, id, rect, map});
            };
            if (!isCurrent)
                add(sub.inputFetch, sub.inputId, sub.inputRect, false);
            for (int slot = 0; slot < SLOT_COUNT; slot++)
                add(sub.maps[slot].fetch, sub.maps[slot].id, sub.maps[slot].rect, true);
        }
    }

    const auto fetchOne = [&](const Need& need) -> PF_Err
    {
        PF_Err e = PF_Err_NONE;
        const auto started = std::chrono::steady_clock::now();
        PF_EffectWorld* world = nullptr;
        e = render.extra->cb->checkout_layer_pixels(in_data->effect_ref, need.id, &world);
        const double ms = MsSince(started);
        render.fetchMs += ms;
        if (need.map)
        {
            render.mapMs += ms;
            render.mapsFetched++;
        }
        else
            render.fetched++;
        PlacedImage placed{};
        if (!e && world)
            e = render.place(world, need.rect, &placed);
        if (!e && placed.buffer)
        {
            StepBatch copying{render.queue, nil, {}};
            id<MTLBuffer> copy = render.renderer.copyPixels(copying, placed.buffer, placed.pitch, placed.height);
            if (copy && render.renderer.flush(copying))
            {
                const FetchedPixels pixels{copy, need.rect, placed.pitch, placed.width, placed.height};
                cache.publishFetch(need.tag->key, need.tag->state, pixels);
                render.prefetched[need.tag] = pixels;
            }
        }
        // Copied, or not worth keeping: either way AE can have it back now. If the copy failed, the
        // step fetches it again.
        if (world)
            render.extra->cb->checkin_layer_pixels(in_data->effect_ref, need.id);
        return e;
    };

    // Fetches what another render hasn't already kept. One that another render is fetching goes on
    // `later`, if given, and is otherwise fetched here too.
    const auto tryFetch = [&](const Need& need, std::vector<Need>* later) -> PF_Err
    {
        if (const PF_Err aborted = PF_ABORT(in_data))
            return aborted;
        FetchedPixels pixels;
        if (render.found(*need.tag, &pixels))
        {
            render.prefetched[need.tag] = pixels;
            return PF_Err_NONE;
        }
        const bool mine = cache.beginFetch(need.tag->key);
        if (!mine && later)
        {
            later->push_back(need);
            return PF_Err_NONE;
        }
        const PF_Err e = fetchOne(need);
        if (mine)
            cache.endFetch(need.tag->key);
        return e;
    };

    std::vector<Need> later;
    for (const Need& need : needs)
        if ((err = tryFetch(need, &later)))
            return err;
    for (const Need& need : later)
        if ((err = tryFetch(need, nullptr)))
            return err;
    return err;
}

// Writes the image to AE's output.
PF_Err ShowImage(GpuRender& render, id<MTLBuffer> image, PF_EffectWorld* output)
{
    PF_Err err = PF_Err_NONE;
    const CanvasKey& canvas = render.data.canvas;
    FeedbackRenderer& renderer = render.renderer;

    // The current frame goes on top here, as PreRender decided (overlayMode). It stays sharp, and isn't fed
    // back in twice.
    id<MTLBuffer> shown = image;
    bool shownIsScratch = false;
    if (const int mode = render.data.overlayMode; mode && render.currentInput)
    {
        PlacedImage input{};
        ERR(render.place(render.currentInput, render.data.currentInputRect, &input));
        if (!err && render.data.canvas.compSpace &&
            !WarpOntoCanvas(&input, render.data.layerToCompNow, render.data.currentInputRect, canvas, render.in_data))
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        if (!err)
        {
            shown = renderer.compositeInput(render.queue, canvas, image, input, mode, render.data.filter,
                                            render.data.opacity);
            shownIsScratch = shown != nil;
            if (!shown)
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }
    }

    PlacedImage placedOutput{};
    ERR(render.place(output, render.data.outputRect, &placedOutput));
    // In comp space, each output pixel shows the comp where the layer puts it.
    float canvasFromOutput[9];
    if (render.data.canvas.compSpace)
        Between(AtResolution(render.data.layerToCompNow, render.in_data), render.data.outputRect.left,
                render.data.outputRect.top, canvas.left, canvas.top)
            .toFloats(canvasFromOutput);
    if (!err && !renderer.copyOut(render.queue, canvas, shown, placedOutput,
                                  render.data.canvas.compSpace ? canvasFromOutput : nullptr, render.data.filter))
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    if (shownIsScratch)
        renderer.cache().releaseScratch(shown);
    return err;
}

PF_Err RenderGPU(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    auto* data = static_cast<PreRenderData*>(extra->input->pre_render_data);
    if (!data)
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

    const CanvasKey& canvas = data->canvas;
    const A_long step = data->grid.step;
    const auto startTime = std::chrono::steady_clock::now();
    int framesComputed = 0;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)info.command_queuePV;
    GpuRender render{in_data, extra, gpu.get(), worlds.get(), renderer, queue, *data, nullptr, StepBatch{queue, nil, {}}, {}};

    @autoreleasepool
    {
        PF_EffectWorld* currentWorld = nullptr;
        PF_EffectWorld* output = nullptr;
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, data->currentInputId, &currentWorld));
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
        render.currentInput = !IsEmpty(data->currentInputRect) ? currentWorld : nullptr;

        // A newer image than PreRender found can have appeared since, from a rebuild that kept its
        // last few frames. Otherwise, what it found is the place to start.
        const int frameCount = int(data->frames.size());
        // A fingerprinted image is only of use if the layer went where it did, and expressions gave what
        // they did, when PreRender looked.
        MotionAt motionAt;
        if (data->fingerprinted)
            motionAt = [data, step](A_long t)
            {
                // [k] is k frames before the current one.
                const auto& frames = data->frames;
                if (step != 0 && !frames.empty() && (frames.front().time - t) % step == 0)
                {
                    const A_long k = (frames.front().time - t) / step;
                    if (k >= 0 && k < A_long(frames.size()) && frames[size_t(k)].time == t)
                        return frames[size_t(k)].motion;
                }
                return uint64_t(1); // never a match: zero is a layer-space image
            };
        std::optional<CompWatch> watch;
        if (data->watchesComp)
            watch.emplace(in_data, out_data);
        const ValidityCheck validity(in_data, utils, canvas, step, motionAt, watch ? &*watch : nullptr, data->made);
        TrailImage resume;
        int resumeK = err ? -1 : FindResume(cache, validity, canvas, data->grid.last, step, frameCount, &resume);
        if (resumeK < 0)
        {
            resume = data->resume;
            resumeK = frameCount;
        }
        framesComputed = resumeK;
        id<MTLBuffer> image = resume.buffer;
        TrailCarry carry = resume.carry;

        // Set while `image` is a scratch buffer of ours rather than one the cache owns.
        bool imageIsScratch = false;

        if (image && !(resume.canvas == canvas))
        {
            image = renderer.reframe(render.queue, resume.canvas, canvas, image);
            imageIsScratch = image != nil;
            if (!image)
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }

        render.utils = &utils;
        if (!err && resumeK > 0)
            ERR(FetchAhead(render, std::min(resumeK, kFramesFetchedAhead), image == nil));

        // A step's result becomes the image, and the one before it can be reused once the GPU is done.
        const auto adopt = [&](id<MTLBuffer> next)
        {
            if (!next)
                return;
            if (imageIsScratch)
                render.batch.freeBuffers.push_back(image);
            image = next;
            imageIsScratch = true;
        };
        // Runs the queued steps, keeping the first error.
        const auto flush = [&]
        {
            const PF_Err flushErr = FlushSteps(render);
            if (!err)
                err = flushErr;
        };

        for (int k = resumeK - 1; !err && k >= 0; k--)
        {
            const FrameCheckout& frame = data->frames[k];
            const int subCount = int(frame.substeps.size());

            for (int s = 0; !err && s < subCount; s++)
            {
                const SubstepCheckout& sub = frame.substeps[s];
                const bool isCurrent = k == 0 && s == subCount - 1 && !data->between;

                // Until there's an image or an input, every step would only produce blank again.
                if (!image && !isCurrent && IsEmpty(sub.inputRect))
                    continue;

                // Command buffers are autoreleased and hold on to what they used, so a long rebuild
                // mustn't keep them all until it ends.
                @autoreleasepool
                {
                    id<MTLBuffer> next = nil;
                    ERR(ComputeSample(render, sub, 1.0f / float(data->samplesPerStep), s == subCount - 1, isCurrent,
                                      image, &carry, &next));
                    adopt(next);
                    if (render.queuedSteps >= kStepsPerBatch)
                        flush();
                }
            }

            if (err)
                break;

            // Keep only the frames most likely to be asked for next; the rest of a long rebuild
            // would otherwise push useful images out of the cache.
            ImageValidity kept;
            if (image && k < kRebuildFramesKept && validity.at(frame.time, &kept))
            {
                ERR(FlushSteps(render));
                if (!err)
                {
                    cache.publish({{image, canvas, carry}, kept, data->grid.last}, frame.time);
                    imageIsScratch = false;
                }
            }
        }

        // Between frames, the part of a step that takes the trail on from the last frame to now. It's
        // only shown: the next frame starts again from the last.
        if (!err && data->between)
        {
            @autoreleasepool
            {
                id<MTLBuffer> next = nil;
                // Only shown, so what it owes is left with the last frame's image.
                TrailCarry shownCarry = carry;
                ERR(ComputeSample(render, *data->between, data->grid.share, false, true, image, &shownCarry, &next));
                adopt(next);
            }
        }

        flush();

        if (!err && image && output)
            ERR(ShowImage(render, image, output));

        if (imageIsScratch)
            cache.releaseScratch(image);
        extra->cb->checkin_layer_pixels(in_data->effect_ref, data->currentInputId);
    }

    Log("frame=%d computed=%d samples=%d fetched=%d fetchMs=%.1f gpuMs=%.1f %dx%d out=%dx%d took=%.1fms err=%d maps=%d mapMs=%.1f shared=%d",
        step > 0 ? in_data->current_time / step : 0,
        framesComputed,
        data->samplesPerStep,
        render.fetched,
        render.fetchMs,
        render.gpuMs,
        canvas.width,
        canvas.height,
        data->outputRect.right - data->outputRect.left,
        data->outputRect.bottom - data->outputRect.top,
        MsSince(startTime),
        err,
        render.mapsFetched,
        render.mapMs,
        render.shared);
    return err;
}

// Without a GPU the effect can't run, so the layer passes through unchanged.
PF_Err RenderCPU(PF_InData* in_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    auto* data = static_cast<PreRenderData*>(extra->input->pre_render_data);
    if (!data || IsEmpty(data->currentInputRect))
        return PF_Err_NONE;
    const PF_LRect& inputRect = data->currentInputRect;

    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, data->currentInputId, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (!err && input && output)
    {
        // The output can be larger than the layer now, so the copy may only cover part of it.
        const PF_LRect overlap = Intersect(inputRect, data->outputRect);
        if (!IsEmpty(overlap))
        {
            PF_Rect source, dest;
            source.left = overlap.left - inputRect.left;
            source.top = overlap.top - inputRect.top;
            source.right = overlap.right - inputRect.left;
            source.bottom = overlap.bottom - inputRect.top;
            dest.left = overlap.left - data->outputRect.left;
            dest.top = overlap.top - data->outputRect.top;
            dest.right = overlap.right - data->outputRect.left;
            dest.bottom = overlap.bottom - data->outputRect.top;
            ERR(PF_COPY(input, output, &source, &dest));
        }
    }
    Log("CPU render time=%d: passing through, err=%d", in_data->current_time, err);
    extra->cb->checkin_layer_pixels(in_data->effect_ref, data->currentInputId);
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
                       "Limit Pre-Roll: off, jumping to a frame recalculates the whole trail before it. On, "
                       "only the last Pre-Roll Limit frames are, which is quicker but can show a shorter trail. "
                       "Playing forward keeps the full trail either way.");
            break;
        case PF_Cmd_GLOBAL_SETUP:
            err = GlobalSetup(out_data);
            break;
        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_data, out_data);
            break;
        case PF_Cmd_UPDATE_PARAMS_UI:
        case PF_Cmd_USER_CHANGED_PARAM:
            err = UpdateParamsUI(in_data, out_data, params);
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
