// Readings taken at moments, such as where a layer is or what an expression gives, kept between calls
// while the project's timestamp stays the same. AE's timestamp goes up with every change that affects
// rendering, so readings from before a change are never taken for current ones.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "AE_GeneralPlug.h"

inline bool SameStamp(const AEGP_TimeStamp& a, const AEGP_TimeStamp& b)
{
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

// A moment as one key: a time and that time's scale.
inline uint64_t MomentKey(A_long value, A_u_long scale)
{
    return uint64_t(uint32_t(value)) << 32 | uint32_t(scale);
}

template <typename V>
class MomentReadings
{
public:
    using Values = std::unordered_map<uint64_t, V>; // by MomentKey

    // What's kept for `key` at timestamp `made`, if anything. It stays as it is for as long as it's held.
    std::shared_ptr<const Values> find(uint64_t key, const AEGP_TimeStamp& made)
    {
        std::lock_guard<std::mutex> guard(mLock);
        auto found = mEntries.find(key);
        if (found == mEntries.end() || !SameStamp(found->second.made, made))
            return nullptr;
        return found->second.values;
    }

    // Adds `fresh`, read at timestamp `made`, to what's kept for `key`, replacing what was kept at another.
    // The kept readings are added to in place, unless another call holds them, so let go of them first.
    void add(uint64_t key, const AEGP_TimeStamp& made, const Values& fresh)
    {
        std::lock_guard<std::mutex> guard(mLock);
        Entry& kept = mEntries[key];
        if (!SameStamp(kept.made, made) || !kept.values)
            kept.values = std::make_shared<Values>();
        else if (kept.values.use_count() > 1)
            kept.values = std::make_shared<Values>(*kept.values);
        // Long sessions on one project mustn't grow this without end.
        if (kept.values->size() > kMostReadings)
            kept.values->clear();
        kept.values->insert(fresh.begin(), fresh.end());
        kept.made = made;
    }

private:
    static constexpr size_t kMostReadings = 1000000;

    struct Entry
    {
        AEGP_TimeStamp made{};
        std::shared_ptr<Values> values;
    };
    std::mutex mLock;
    std::unordered_map<uint64_t, Entry> mEntries;
};
