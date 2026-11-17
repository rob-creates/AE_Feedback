// Recent feedback images, identified by what produced them rather than by which effect
// instance made them. An entry is only reused when AE confirms the entire input and parameter
// history up to its frame is unchanged, so a stale or foreign entry can never be picked up.
#pragma once

#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

#include "AE_EffectSuites.h"

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

    bool compatible(const CanvasKey& o) const
    {
        return SameScale(downsampleX, o.downsampleX) && SameScale(downsampleY, o.downsampleY) &&
               firstTime == o.firstTime && compSpace == o.compSpace;
    }

    bool operator==(const CanvasKey& o) const
    {
        return left == o.left && top == o.top && width == o.width && height == o.height && compatible(o);
    }

    size_t bytes() const { return size_t(width) * size_t(height) * 16; }
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
        id<MTLBuffer> buffer;
    };

    explicit FeedbackCache(id<MTLDevice> device);

    // All images for this frame; the caller decides which, if any, matches the current history.
    std::vector<Candidate> candidates(const CanvasKey& key, A_long time);
    void touch(id<MTLBuffer> buffer);

    // Takes ownership of the buffer. Evicted buffers are dropped rather than recycled, because
    // a render further up the stack may still be reading them.
    void publish(const CanvasKey& key, A_long time, const PF_State& history, uint64_t motion, id<MTLBuffer> buffer);

    id<MTLBuffer> acquireScratch(size_t bytes);
    void releaseScratch(id<MTLBuffer> buffer);

private:
    struct Entry
    {
        CanvasKey key;
        A_long time;
        PF_State history;
        uint64_t motion;
        id<MTLBuffer> buffer;
        uint64_t lastUsed;
    };

    void evictLocked();

    id<MTLDevice> mDevice;
    std::mutex mMutex;
    std::vector<Entry> mEntries;
    std::vector<id<MTLBuffer>> mScratch;
    uint64_t mClock = 0;
    size_t mEntryBytes = 0;
};
