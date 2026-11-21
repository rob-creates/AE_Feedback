// The one way fingerprints are made here: FNV-1a over the bytes of what's read, and splitmix64's
// finaliser where a fingerprint is combined with others by adding.
#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint64_t kHashStart = 0xCBF29CE484222325ull; // FNV-1a's offset basis

// Mixes `size` bytes at `data` into `h`.
inline void Mix(uint64_t& h, const void* data, size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; i++)
        h = (h ^ bytes[i]) * 1099511628211ull;
}

// Mixes a value's bytes into `h`.
template <typename T>
inline void Mix(uint64_t& h, const T& value)
{
    Mix(h, &value, sizeof(value));
}

// splitmix64's finaliser: every bit of the result depends on every bit of `x`.
inline uint64_t Scramble(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ x >> 31;
}
