#include "FeedbackCache.h"

#include <algorithm>
#include <cstdlib>

namespace
{
// Holds about 30 frames at 4K or 120 at 1080p across all instances of the effect, given room.
const size_t kMostBudgetBytes = size_t(4) * 1024 * 1024 * 1024;
// Kept images take no more than this share of what the GPU can use, and a step's buffers no more
// than kStepShare, at about kStepCanvases canvases' worth: the trail, its result and the scratch for
// displacement, edges and the blur's smaller images. AE's own images take the rest.
const size_t kBudgetShare = 4;
const size_t kStepShare = 3;
const size_t kStepCanvases = 12;
// A canvas side never goes beyond this, however much memory there is.
const size_t kMostCanvasPixels = size_t(8192) * 8192;
// Nor below about 4K, which a machine that can run AE at all can hold.
const size_t kLeastCanvasPixels = size_t(4096) * 2304;
// A step with displacement and a downsampled blur has seven free at rest.
const size_t kMaxScratchBuffers = 8;
} // namespace

FeedbackCache::FeedbackCache(id<MTLDevice> device)
    : mDevice(device)
{
    const size_t usable = size_t(device.recommendedMaxWorkingSetSize);
    mBudgetBytes = std::min(kMostBudgetBytes, usable / kBudgetShare);
    mCanvasPixelLimit =
        std::clamp(usable / kStepShare / (kStepCanvases * 16), kLeastCanvasPixels, kMostCanvasPixels);
#ifdef FEEDBACK_LOG
    // Tests stand in for a machine with less memory.
    if (const char* given = getenv("FEEDBACK_CANVAS_PIXELS"))
        mCanvasPixelLimit = std::max<size_t>(strtoull(given, nullptr, 10), 1);
#endif
}

std::vector<FeedbackCache::Candidate> FeedbackCache::candidates(const CanvasKey& key, A_long time)
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::vector<Candidate> out;
    for (const Entry& e : mEntries)
        if (e.time == time && e.key.compatible(key))
            out.push_back({e.key, e.history, e.motion, e.made, e.buffer});
    return out;
}

void FeedbackCache::touch(id<MTLBuffer> buffer)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (Entry& e : mEntries)
        if (e.buffer == buffer)
            e.lastUsed = ++mClock;
}

void FeedbackCache::publish(
    const CanvasKey& key, A_long time, const PF_State& history, uint64_t motion, const AEGP_TimeStamp& made,
    id<MTLBuffer> buffer)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mEntries.push_back({key, time, history, motion, made, buffer, ++mClock});
    mEntryBytes += buffer.length;
    evictLocked();
}

void FeedbackCache::evictLocked()
{
    // The newest entry always survives, however large it is.
    while (mEntryBytes > mBudgetBytes && mEntries.size() > 1)
    {
        auto oldest = std::min_element(
            mEntries.begin(), mEntries.end(), [](const Entry& a, const Entry& b) { return a.lastUsed < b.lastUsed; });
        mEntryBytes -= oldest->buffer.length;
        mEntries.erase(oldest);
    }
}

id<MTLBuffer> FeedbackCache::acquireScratch(size_t bytes)
{
    {
        // Canvas sizes change as a trail moves, so a slightly larger buffer is good enough. Kernels
        // address pixels by pitch and never rely on a buffer's length.
        std::lock_guard<std::mutex> lock(mMutex);
        auto best = mScratch.end();
        for (auto it = mScratch.begin(); it != mScratch.end(); ++it)
        {
            const size_t length = (*it).length;
            if (length >= bytes && length <= bytes * 2 && (best == mScratch.end() || length < (*best).length))
                best = it;
        }
        if (best != mScratch.end())
        {
            id<MTLBuffer> buffer = *best;
            mScratch.erase(best);
            return buffer;
        }
    }
    return [mDevice newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
}

void FeedbackCache::releaseScratch(id<MTLBuffer> buffer)
{
    if (!buffer)
        return;
    std::lock_guard<std::mutex> lock(mMutex);
    mScratch.push_back(buffer);
    if (mScratch.size() > kMaxScratchBuffers)
        mScratch.erase(mScratch.begin());
}
