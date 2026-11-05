// Recent feedback images, identified by what produced them rather than by which effect
// instance made them. An entry is only reused when AE confirms the entire input and parameter
// history up to its frame is unchanged, so a stale or foreign entry can never be picked up.
#pragma once

#import <Metal/Metal.h>
#include <cstdint>
#include <mutex>
#include <vector>

#include "AE_EffectSuites.h"

// Where an image sits, and at what resolution. Canvases follow the trail, so consecutive frames
// usually differ in position; only images of the same resolution can be used together.
struct CanvasKey
{
    A_long left, top, width, height;
    A_long downsampleXNum;
    A_u_long downsampleXDen;
    A_long downsampleYNum;
    A_u_long downsampleYDen;

    bool operator==(const CanvasKey& o) const
    {
        return left == o.left && top == o.top && width == o.width && height == o.height &&
               downsampleXNum == o.downsampleXNum && downsampleXDen == o.downsampleXDen &&
               downsampleYNum == o.downsampleYNum && downsampleYDen == o.downsampleYDen;
    }

    bool sameResolution(const CanvasKey& o) const
    {
        return downsampleXNum == o.downsampleXNum && downsampleXDen == o.downsampleXDen &&
               downsampleYNum == o.downsampleYNum && downsampleYDen == o.downsampleYDen;
    }

    size_t bytes() const { return size_t(width) * size_t(height) * 16; }
};

class FeedbackCache
{
public:
    struct Candidate
    {
        CanvasKey key;
        PF_State history;
        id<MTLBuffer> buffer;
    };

    explicit FeedbackCache(id<MTLDevice> device);

    // All images for this frame; the caller decides which, if any, matches the current history.
    std::vector<Candidate> candidates(const CanvasKey& key, A_long time);
    bool hasFrame(const CanvasKey& key, A_long time);
    void touch(id<MTLBuffer> buffer);

    // Takes ownership of the buffer. Evicted buffers are dropped rather than recycled, because
    // a render further up the stack may still be reading them.
    void publish(const CanvasKey& key, A_long time, const PF_State& history, id<MTLBuffer> buffer);

    id<MTLBuffer> acquireScratch(size_t bytes);
    void releaseScratch(id<MTLBuffer> buffer);

private:
    struct Entry
    {
        CanvasKey key;
        A_long time;
        PF_State history;
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
