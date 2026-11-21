// Recent feedback images, identified by what produced them rather than by which effect instance made
// them, except on adjustment layers (CanvasKey::owner). An entry is only reused when AE confirms the
// entire input and parameter history up to its frame is unchanged, so a stale or foreign entry can
// never be picked up.
#pragma once

#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

#include "AE_EffectSuites.h"
#include "AE_GeneralPlug.h"

inline bool SameScale(const PF_RationalScale& a, const PF_RationalScale& b)
{
    return a.num == b.num && a.den == b.den;
}

inline float ScaleValue(const PF_RationalScale& r)
{
    return float(r.num) / float(std::max<A_u_long>(r.den, 1));
}

// Where an image sits, at what resolution, and when its trail began. Canvases follow the trail, so
// consecutive frames usually differ in position; only images that otherwise match can be used together.
struct CanvasKey
{
    A_long left, top, width, height;
    PF_RationalScale downsampleX, downsampleY;
    A_long firstTime;
    bool compSpace = false; // in comp pixels rather than the layer's
    // Zero for most layers, whose images any instance with the same history can use. An adjustment
    // layer's history doesn't take in what's beneath it, so its images are kept to the instance that
    // made them, as EffectInstance tells them apart.
    uint64_t owner = 0;

    bool compatible(const CanvasKey& o) const
    {
        return SameScale(downsampleX, o.downsampleX) && SameScale(downsampleY, o.downsampleY) &&
               firstTime == o.firstTime && compSpace == o.compSpace && owner == o.owner;
    }

    bool operator==(const CanvasKey& o) const
    {
        return left == o.left && top == o.top && width == o.width && height == o.height && compatible(o);
    }

    bool contains(const CanvasKey& o) const
    {
        return left <= o.left && top <= o.top && left + width >= o.left + o.width && top + height >= o.top + o.height;
    }

    size_t bytes() const { return size_t(width) * size_t(height) * 16; }
};

// Pixel Sampling at Pixel moves the trail by whole pixels and owes what's left over: how far, in canvas
// pixels, the image still has to move to be where the trail should be. Kept with each image, so a trail
// resumed from it moves on exactly as one that never stopped. Always zero at Bilinear.
struct TrailCarry
{
    float x = 0.0f, y = 0.0f;
};

// A layer as one render fetched it from AE, kept so that other renders can use it rather than ask AE
// for it again. AE renders several frames at once, and each reads layers at moments the others read too.
struct FetchKey
{
    uint64_t instance; // the effect, as EffectInstance tells them apart
    int param;         // which of its layer controls
    A_long value, step;
    A_u_long scale; // the moment, exactly, and how long it lasts
    PF_RationalScale downsampleX, downsampleY;
    int quality, field, preserveRgb;
    PF_LRect area; // what was asked for, in the layer's pixels

    // The same layer at the same moment, however much of it.
    bool sameSource(const FetchKey& o) const
    {
        // Moments are fractions, and the same one can come with different scales.
        const auto same = [](A_long a, A_u_long as, A_long b, A_u_long bs)
        { return __int128(a) * __int128(bs) == __int128(b) * __int128(as); };
        return instance == o.instance && param == o.param && same(value, scale, o.value, o.scale) &&
               same(step, scale, o.step, o.scale) && SameScale(downsampleX, o.downsampleX) &&
               SameScale(downsampleY, o.downsampleY) && quality == o.quality && field == o.field &&
               preserveRgb == o.preserveRgb;
    }

    // What was fetched for `o` would do for this: all of what this asks for is in it.
    bool coveredBy(const FetchKey& o) const
    {
        return sameSource(o) && o.area.left <= area.left && o.area.top <= area.top && o.area.right >= area.right &&
               o.area.bottom >= area.bottom;
    }
};

// A copy of what AE gave: `rect` is where it sits, in the layer, and `pitch` is in pixels.
struct FetchedPixels
{
    id<MTLBuffer> buffer;
    PF_LRect rect;
    int pitch, width, height;
};

class FeedbackCache
{
public:
    // `motion` fingerprints where the layer went while a comp-space trail was built, which AE's history
    // doesn't cover. Zero for trails kept in the layer.
    struct Candidate
    {
        CanvasKey key;
        PF_State history;
        uint64_t motion;
        AEGP_TimeStamp made; // the project's timestamp when the image's inputs were read
        A_long drawnFor;     // the frame whose canvas the image was drawn on, which may be later than its own
        id<MTLBuffer> buffer;
        TrailCarry carry;
    };

    explicit FeedbackCache(id<MTLDevice> device);

    // The most pixels a canvas may have, so that a step's buffers fit comfortably in what the GPU can
    // use on this machine. Memory is shared with the rest of the system, so a canvas that doesn't fit
    // would page rather than fail, and slow everything to a crawl.
    size_t canvasPixelLimit() const { return mCanvasPixelLimit; }

    // All images for this frame; the caller decides which, if any, matches the current history.
    std::vector<Candidate> candidates(const CanvasKey& key, A_long time);

    // Fetched layers: all kept for this key, with the state of the project they were fetched in, for
    // the caller to match against its own. Kept to a budget of their own, oldest-used dropped first.
    struct FetchCandidate
    {
        PF_State state;
        FetchedPixels pixels;
    };
    std::vector<FetchCandidate> fetchCandidates(const FetchKey& key);
    void publishFetch(const FetchKey& key, const PF_State& state, const FetchedPixels& pixels);
    // Says a render is fetching what `key` names, so others can fetch something else meanwhile rather
    // than the same. False if another already is. Each true is followed by endFetch, kept or not.
    bool beginFetch(const FetchKey& key);
    void endFetch(const FetchKey& key);
    void touch(id<MTLBuffer> buffer);

    // Takes ownership of the buffer. Evicted buffers are dropped rather than recycled, because
    // a render further up the stack may still be reading them.
    void publish(const CanvasKey& key, A_long time, const PF_State& history, uint64_t motion, const AEGP_TimeStamp& made,
                 A_long drawnFor, id<MTLBuffer> buffer, TrailCarry carry);

    id<MTLBuffer> acquireScratch(size_t bytes);
    void releaseScratch(id<MTLBuffer> buffer);

private:
    struct Entry
    {
        CanvasKey key;
        A_long time;
        PF_State history;
        uint64_t motion;
        AEGP_TimeStamp made;
        A_long drawnFor;
        id<MTLBuffer> buffer;
        TrailCarry carry;
        uint64_t lastUsed;
    };

    void evictLocked();

    id<MTLDevice> mDevice;
    size_t mBudgetBytes; // for kept images
    size_t mCanvasPixelLimit;
    std::mutex mMutex;
    std::vector<Entry> mEntries;
    std::vector<id<MTLBuffer>> mScratch;
    uint64_t mClock = 0;
    size_t mEntryBytes = 0;

    struct FetchEntry
    {
        FetchKey key;
        PF_State state;
        FetchedPixels pixels;
        uint64_t lastUsed;
    };
    std::vector<FetchEntry> mFetches; // guarded by mMutex
    std::vector<FetchKey> mInFlight;  // guarded by mMutex
    size_t mFetchBytes = 0;
    size_t mFetchBudgetBytes;
};
