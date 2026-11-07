#include "FeedbackCache.h"

#include <algorithm>

namespace
{
// Holds about 30 frames at 4K or 120 at 1080p across all instances of the effect.
const size_t kBudgetBytes = size_t(4) * 1024 * 1024 * 1024;
// A step with displacement and a downsampled blur has seven free at rest.
const size_t kMaxScratchBuffers = 8;
} // namespace

FeedbackCache::FeedbackCache(id<MTLDevice> device)
    : mDevice(device)
{
}

std::vector<FeedbackCache::Candidate> FeedbackCache::candidates(const CanvasKey& key, A_long time)
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::vector<Candidate> out;
    for (const Entry& e : mEntries)
        if (e.time == time && e.key.compatible(key))
            out.push_back({e.key, e.history, e.buffer});
    return out;
}

void FeedbackCache::touch(id<MTLBuffer> buffer)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (Entry& e : mEntries)
        if (e.buffer == buffer)
            e.lastUsed = ++mClock;
}

void FeedbackCache::publish(const CanvasKey& key, A_long time, const PF_State& history, id<MTLBuffer> buffer)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mEntries.push_back({key, time, history, buffer, ++mClock});
    mEntryBytes += buffer.length;
    evictLocked();
}

void FeedbackCache::evictLocked()
{
    // The newest entry always survives, however large it is.
    while (mEntryBytes > kBudgetBytes && mEntries.size() > 1)
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
