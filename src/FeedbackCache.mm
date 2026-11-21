#include "FeedbackCache.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

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
// Scratch buffers kept for reuse come to no more than a step's worth: kStepCanvases of the largest kept,
// however many smaller ones, such as a blur's, that takes.
} // namespace

FeedbackCache::FeedbackCache(id<MTLDevice> device)
    : mDevice(device)
{
    const size_t usable = size_t(device.recommendedMaxWorkingSetSize);
    mBudgetBytes = std::min(kMostBudgetBytes, usable / kBudgetShare);
    // Enough for the moments a few frames rendered at once share. Kept small: it comes on top of the
    // trail's images, and memory the GPU pages to and from slows everything down.
    mFetchBudgetBytes = mBudgetBytes / 4;
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
        if (e.time == time && e.kept.image.canvas.compatible(key))
            out.push_back(e.kept);
    return out;
}

void FeedbackCache::touch(id<MTLBuffer> buffer)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (Entry& e : mEntries)
        if (e.kept.image.buffer == buffer)
            e.lastUsed = ++mClock;
}

void FeedbackCache::publish(const Candidate& kept, A_long time)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mEntries.push_back({kept, time, ++mClock});
    mEntryBytes += kept.image.buffer.length;
    evictLocked();
}

void FeedbackCache::evictLocked()
{
    // The newest entry always survives, however large it is.
    while (mEntryBytes > mBudgetBytes && mEntries.size() > 1)
    {
        auto oldest = std::min_element(
            mEntries.begin(), mEntries.end(), [](const Entry& a, const Entry& b) { return a.lastUsed < b.lastUsed; });
        mEntryBytes -= oldest->kept.image.buffer.length;
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
            mScratchBytes -= buffer.length;
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
    mScratchBytes += buffer.length;
    // The oldest go first. Once a larger canvas's buffers have gone, the limit comes down with them.
    const auto largest = [this]
    {
        size_t most = 0;
        for (id<MTLBuffer> kept : mScratch)
            most = std::max<size_t>(most, kept.length);
        return most;
    };
    while (mScratch.size() > 1 && mScratchBytes > kStepCanvases * largest())
    {
        mScratchBytes -= mScratch.front().length;
        mScratch.erase(mScratch.begin());
    }
}

std::vector<FeedbackCache::FetchCandidate> FeedbackCache::fetchCandidates(const FetchKey& key)
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::vector<FetchCandidate> out;
    for (FetchEntry& e : mFetches)
        if (key.coveredBy(e.key))
        {
            e.lastUsed = ++mClock;
            out.push_back({e.state, e.pixels});
        }
    return out;
}

void FeedbackCache::publishFetch(const FetchKey& key, const PF_State& state, const FetchedPixels& pixels)
{
    std::lock_guard<std::mutex> lock(mMutex);
    // Another render may have kept the same already. Only to save memory, so comparing bytes is enough.
    for (const FetchEntry& e : mFetches)
        if (key.coveredBy(e.key) && memcmp(&e.state, &state, sizeof(state)) == 0)
            return;
    mFetches.push_back({key, state, pixels, ++mClock});
    mFetchBytes += pixels.buffer.length;
    while (mFetchBytes > mFetchBudgetBytes && !mFetches.empty())
    {
        auto oldest = std::min_element(mFetches.begin(), mFetches.end(),
                                       [](const FetchEntry& a, const FetchEntry& b) { return a.lastUsed < b.lastUsed; });
        mFetchBytes -= oldest->pixels.buffer.length;
        mFetches.erase(oldest);
    }
}

namespace
{
bool SameFetch(const FetchKey& a, const FetchKey& b)
{
    return a.coveredBy(b) && b.coveredBy(a);
}
} // namespace

bool FeedbackCache::beginFetch(const FetchKey& key)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (const FetchKey& k : mInFlight)
        if (SameFetch(k, key))
            return false;
    mInFlight.push_back(key);
    return true;
}

void FeedbackCache::endFetch(const FetchKey& key)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto it = mInFlight.begin(); it != mInFlight.end(); ++it)
        if (SameFetch(*it, key))
        {
            mInFlight.erase(it);
            return;
        }
}
