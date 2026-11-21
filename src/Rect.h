// Rectangles of pixels, as AE gives them: left and top inclusive, right and bottom exclusive.
#pragma once

#include <algorithm>

#include "AE_Effect.h"

inline bool IsEmpty(const PF_LRect& r)
{
    return r.left >= r.right || r.top >= r.bottom;
}

inline PF_LRect Intersect(const PF_LRect& a, const PF_LRect& b)
{
    return PF_LRect{std::max(a.left, b.left), std::max(a.top, b.top), std::min(a.right, b.right),
                    std::min(a.bottom, b.bottom)};
}

// The box around both. An empty one adds nothing, so a union can start from an empty rectangle.
inline PF_LRect Union(const PF_LRect& a, const PF_LRect& b)
{
    if (IsEmpty(a))
        return b;
    if (IsEmpty(b))
        return a;
    return PF_LRect{std::min(a.left, b.left), std::min(a.top, b.top), std::max(a.right, b.right),
                    std::max(a.bottom, b.bottom)};
}

inline bool SameRect(const PF_LRect& a, const PF_LRect& b)
{
    return a.left == b.left && a.top == b.top && a.right == b.right && a.bottom == b.bottom;
}

// Whether all of `inner` is in `outer`.
inline bool Contains(const PF_LRect& outer, const PF_LRect& inner)
{
    return outer.left <= inner.left && outer.top <= inner.top && outer.right >= inner.right && outer.bottom >= inner.bottom;
}
