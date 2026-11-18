#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// Pixels are stored BGRA: .x blue, .y green, .z red, .w alpha.
// Layers from AE and the final output are straight alpha; the feedback image is premultiplied.

namespace
{
// Stops a runaway loop poisoning the cached image forever with NaN or infinity.
inline float4 sanitize(float4 c)
{
    c = select(c, float4(0.0), isnan(c));
    c.xyz = clamp(c.xyz, -1024.0f, 1024.0f);
    c.w = saturate(c.w);
    return c;
}

inline float4 bilinearClamped(device const float4* src, int pitch, int w, int h, float2 pos)
{
    pos = clamp(pos, float2(0.0), float2(w - 1, h - 1));
    int2 i0 = int2(floor(pos));
    int2 i1 = min(i0 + 1, int2(w - 1, h - 1));
    float2 f = pos - float2(i0);
    float4 top = mix(src[i0.y * pitch + i0.x], src[i0.y * pitch + i1.x], f.x);
    float4 bottom = mix(src[i1.y * pitch + i0.x], src[i1.y * pitch + i1.x], f.x);
    return mix(top, bottom, f.y);
}

// Straight colour from a premultiplied pixel; nothing where it's transparent.
inline float3 unpremultiply(float4 c)
{
    return c.w > 1e-5f ? c.xyz / c.w : float3(0.0);
}

inline float luma(float3 bgr)
{
    return dot(bgr, float3(0.0722, 0.7152, 0.2126));
}

// Returns the 0-1 channel value, weighted so transparent map pixels cause no shift.
inline float mapShift(float4 m, int channel, bool isStraight)
{
    float3 c = isStraight ? m.xyz : unpremultiply(m);
    float v;
    switch (channel)
    {
    case MAP_CHANNEL_RED: v = c.z; break;
    case MAP_CHANNEL_GREEN: v = c.y; break;
    case MAP_CHANNEL_BLUE: v = c.x; break;
    case MAP_CHANNEL_ALPHA: v = m.w; break;
    default: v = luma(c); break;
    }
    return (v - 0.5f) * 2.0f * m.w;
}

inline float3 rgbToHsl(float3 rgb)
{
    float hi = max3(rgb.r, rgb.g, rgb.b);
    float lo = min3(rgb.r, rgb.g, rgb.b);
    float l = (hi + lo) * 0.5f;
    float d = hi - lo;
    if (d < 1e-6f)
        return float3(0.0, 0.0, l);
    float s = d / max(1.0f - abs(2.0f * l - 1.0f), 1e-6f);
    float h;
    if (hi == rgb.r)
        h = fmod((rgb.g - rgb.b) / d + 6.0f, 6.0f);
    else if (hi == rgb.g)
        h = (rgb.b - rgb.r) / d + 2.0f;
    else
        h = (rgb.r - rgb.g) / d + 4.0f;
    return float3(h / 6.0f, s, l);
}

inline float3 hslToRgb(float3 hsl)
{
    float a = hsl.y * min(hsl.z, 1.0f - hsl.z);
    float3 k = fmod(float3(0.0, 8.0, 4.0) + hsl.x * 12.0f, 12.0f);
    return hsl.z - a * max(float3(-1.0), min(min(k - 3.0f, 9.0f - k), float3(1.0)));
}

// One straight colour channel set blended onto another, as a layer above with that mode.
inline float3 blendColours(float3 bottom, float3 top, int mode)
{
    switch (mode)
    {
    case COMPOSITE_MULTIPLY: return bottom * top;
    case COMPOSITE_LIGHTEN: return max(bottom, top);
    case COMPOSITE_DARKEN: return min(bottom, top);
    case COMPOSITE_OVERLAY:
        return select(1.0f - 2.0f * (1.0f - bottom) * (1.0f - top), 2.0f * bottom * top, bottom <= 0.5f);
    case COMPOSITE_ADD_SUBTRACT: return bottom + 2.0f * top - 1.0f;
    case COMPOSITE_DIFFERENCE: return abs(bottom - top);
    default: return top;
    }
}

// Puts `top` over `bottom`, both premultiplied. The blend modes follow the usual layer rule: the mode
// applies where both are present, and elsewhere each shows as itself.
inline float4 blendImages(float4 bottom, float4 top, int mode)
{
    switch (mode)
    {
    case COMPOSITE_INPUT_IN_FRONT: return top + bottom * (1.0f - top.w);
    case COMPOSITE_INPUT_BEHIND: return bottom + top * (1.0f - bottom.w);
    case COMPOSITE_ADD: return bottom + top;
    case COMPOSITE_SCREEN: return bottom + top - bottom * top;
    default:
    {
        const float b = bottom.w, t = top.w;
        const float3 mixed = blendColours(unpremultiply(bottom), unpremultiply(top), mode);
        return float4((1.0f - t) * bottom.xyz + (1.0f - b) * top.xyz + t * b * mixed, t + b - t * b);
    }
    }
}

// A pixel position through a projective map. False where it falls behind the camera.
inline bool project(constant float* m, float2 p, thread float2& out)
{
    const float w = m[6] * p.x + m[7] * p.y + m[8];
    if (w <= 1e-6f)
        return false;
    out = float2(m[0] * p.x + m[1] * p.y + m[2], m[3] * p.x + m[4] * p.y + m[5]) / w;
    return true;
}

// Bilinear, with each tap premultiplied first if it's straight, and nothing beyond the image, so
// colour hidden in transparent pixels can't bleed in.
inline float4 tap(device const float4* src, int pitch, int w, int h, int2 i, bool straight)
{
    if (i.x < 0 || i.y < 0 || i.x >= w || i.y >= h)
        return float4(0.0);
    float4 v = src[i.y * pitch + i.x];
    if (straight)
        v.xyz *= v.w;
    return v;
}

inline float4 bilinearPremultiplied(device const float4* src, int pitch, int w, int h, float2 pos, bool straight)
{
    const int2 i0 = int2(floor(pos));
    const float2 f = pos - float2(i0);
    const float4 top = mix(tap(src, pitch, w, h, i0, straight), tap(src, pitch, w, h, i0 + int2(1, 0), straight), f.x);
    const float4 bottom =
        mix(tap(src, pitch, w, h, i0 + int2(0, 1), straight), tap(src, pitch, w, h, i0 + int2(1, 1), straight), f.x);
    return mix(top, bottom, f.y);
}

// Where a canvas pixel falls in a map. False where it has no place in it.
inline bool mapPosition(constant MapLookup& m, uint2 xy, thread float2& at)
{
    if (m.warped != 0)
        return project(m.fromCanvas, float2(xy), at);
    at = float2(int2(xy) - int2(m.offsetX, m.offsetY));
    return true;
}

// Transparent beyond the image, rather than repeating its edge.
inline float4 sampleOrClear(device const float4* src, int pitch, int w, int h, float2 pos)
{
    if (pos.x > -1.0f && pos.y > -1.0f && pos.x < float(w) && pos.y < float(h))
        return bilinearClamped(src, pitch, w, h, pos);
    return float4(0.0);
}
} // namespace

kernel void displace(
    device const float4* src [[buffer(0)]],
    device const float4* map [[buffer(1)]],
    device float4* dst [[buffer(2)]],
    constant DisplaceParams& p [[buffer(3)]],
    uint2 xy [[thread_position_in_grid]])
{
    int2 canvas = int2(xy);
    float2 at;
    const int2 m = mapPosition(p.map, xy, at) ? int2(floor(at + 0.5f)) : int2(-1);
    float2 shift = float2(0.0);
    if (m.x >= 0 && m.y >= 0 && m.x < p.map.width && m.y < p.map.height)
    {
        float4 v = map[m.y * p.map.pitch + m.x];
        shift = float2(
            mapShift(v, p.channelX, p.map.isStraight != 0) * p.amountX,
            mapShift(v, p.channelY, p.map.isStraight != 0) * p.amountY);
    }
    // Displacement says where each pixel comes from, and Transform where that was before it moved.
    const float2 shifted = float2(canvas) + shift;
    const float2 pos = float2(
        p.inverse[0] * shifted.x + p.inverse[1] * shifted.y + p.offsetX,
        p.inverse[2] * shifted.x + p.inverse[3] * shifted.y + p.offsetY);
    // Anything brought in from beyond the canvas is transparent.
    dst[xy.y * p.dstPitch + xy.x] = sampleOrClear(src, p.srcPitch, p.width, p.height, pos);
}

// Finds edges with a Sobel filter whose taps are `distance` apart, and puts them over the image.
// A sharp step gives an edge of 1 at Strength 1, however far apart the taps are.
kernel void edge(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant EdgeParams& p [[buffer(2)]],
    uint2 xy [[thread_position_in_grid]])
{
    const float2 c = float2(xy);
    const float2 d = float2(p.distanceX, p.distanceY);
#define AT(dx, dy) sampleOrClear(src, p.srcPitch, p.width, p.height, c + float2(dx, dy) * d)
    const float4 nw = AT(-1, -1), n = AT(0, -1), ne = AT(1, -1);
    const float4 w = AT(-1, 0), e = AT(1, 0);
    const float4 sw = AT(-1, 1), s = AT(0, 1), se = AT(1, 1);
#undef AT
    const float4 gx = ((ne + 2.0f * e + se) - (nw + 2.0f * w + sw)) * 0.25f;
    const float4 gy = ((sw + 2.0f * s + se) - (nw + 2.0f * n + ne)) * 0.25f;

    float4 edges;
    if (p.type == EDGE_LUMINANCE)
    {
        const float g = length(float2(luma(gx.xyz), luma(gy.xyz))) * p.strength;
        edges = float4(float3(g), saturate(g));
    }
    else
    {
        const float4 g = sqrt(gx * gx + gy * gy) * p.strength;
        edges = float4(g.xyz, saturate(max(g.w, max3(g.x, g.y, g.z))));
    }
    // The blend modes work on straight colour, which only stays in range if the edges are valid
    // premultiplied colour. Strong edges otherwise turn Multiply into a brightening.
    if (p.blend >= COMPOSITE_MULTIPLY)
        edges.xyz = min(edges.xyz, float3(edges.w));
    const float4 image = src[xy.y * p.srcPitch + xy.x];
    dst[xy.y * p.dstPitch + xy.x] = sanitize(blendImages(image, edges, p.blend));
}

// Box-averages factor x factor blocks, keeping large blurs cheap.
kernel void blurDown(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant BlurResizeParams& p [[buffer(2)]],
    uint2 xy [[thread_position_in_grid]])
{
    int2 start = int2(xy) * p.factor;
    int2 end = min(start + p.factor, int2(p.srcWidth, p.srcHeight));
    float4 sum = float4(0.0);
    for (int y = start.y; y < end.y; y++)
        for (int x = start.x; x < end.x; x++)
            sum += src[y * p.srcPitch + x];
    int count = max((end.x - start.x) * (end.y - start.y), 1);
    dst[xy.y * p.dstPitch + xy.x] = sum / float(count);
}

kernel void blurAxis(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant BlurAxisParams& p [[buffer(2)]],
    uint2 xy [[thread_position_in_grid]])
{
    int2 c = int2(xy);
    int2 size = int2(p.width, p.height);
    int2 dir = int2(p.dirX, p.dirY);
    float4 sum = src[c.y * p.srcPitch + c.x] * p.weights[0];
    // Beyond the canvas is transparent; repeating the edge would smear it inwards.
    for (int i = 1; i <= p.radius; i++)
    {
        int2 a = c + dir * i;
        int2 b = c - dir * i;
        if (all(a >= 0) && all(a < size))
            sum += src[a.y * p.srcPitch + a.x] * p.weights[i];
        if (all(b >= 0) && all(b < size))
            sum += src[b.y * p.srcPitch + b.x] * p.weights[i];
    }
    dst[xy.y * p.dstPitch + xy.x] = sum;
}

// Adds a weighted share of a blurred image, scaled up from its smaller size, to the sum.
kernel void addScaled(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant AddScaledParams& p [[buffer(2)]],
    uint2 xy [[thread_position_in_grid]])
{
    float4 v;
    if (p.factor > 1)
        v = bilinearClamped(src, p.srcPitch, p.srcWidth, p.srcHeight, (float2(xy) + 0.5f) / float(p.factor) - 0.5f);
    else
        v = src[xy.y * p.srcPitch + xy.x];
    const uint i = xy.y * p.dstPitch + xy.x;
    dst[i] = p.first != 0 ? v * p.weight : dst[i] + v * p.weight;
}

// Each pixel takes its share of the blur from the map's luminance, blending the two blurs either side.
kernel void blurByMap(
    device const float4* sharp [[buffer(0)]],
    device const float4* eighth [[buffer(1)]],
    device const float4* quarter [[buffer(2)]],
    device const float4* halfway [[buffer(3)]],
    device const float4* whole [[buffer(4)]],
    device const float4* map [[buffer(5)]],
    device float4* dst [[buffer(6)]],
    constant BlurMapParams& p [[buffer(7)]],
    uint2 xy [[thread_position_in_grid]])
{
    // Premultiplied, so transparent parts of the map blur nothing. Beyond it, nothing is blurred.
    float2 at;
    const float4 v = mapPosition(p.map, xy, at)
                         ? bilinearPremultiplied(map, p.map.pitch, p.map.width, p.map.height, at, p.map.isStraight != 0)
                         : float4(0.0);
    const float share = saturate(luma(v.xyz));
    const uint i = xy.y * p.pitch + xy.x;
    float4 result;
    if (share <= 0.125f)
        result = mix(sharp[i], eighth[i], share / 0.125f);
    else if (share <= 0.25f)
        result = mix(eighth[i], quarter[i], (share - 0.125f) / 0.125f);
    else if (share <= 0.5f)
        result = mix(quarter[i], halfway[i], (share - 0.25f) / 0.25f);
    else
        result = mix(halfway[i], whole[i], (share - 0.5f) / 0.5f);
    dst[i] = result;
}

namespace
{
// Levels, then HSL, then Alpha.
inline float4 applyGrade(float4 c, constant GradeParams& p)
{
    if ((p.applyLevels != 0 || p.applyHSL != 0) && c.w > 1e-5f)
    {
        float a = c.w;
        float3 rgb = c.zyx / a;

        if (p.applyLevels != 0)
        {
            float3 v = max((rgb - p.inputBlack) / max(p.inputWhite - p.inputBlack, 1e-5f), 0.0f);
            v = pow(v, p.inverseGamma);
            rgb = p.outputBlack + v * (p.outputWhite - p.outputBlack);
        }

        if (p.applyHSL != 0)
        {
            // HSL is only defined on 0-1, so HDR values are scaled into range and back.
            rgb = max(rgb, 0.0f);
            float scale = max(max3(rgb.r, rgb.g, rgb.b), 1.0f);
            float3 hsl = rgbToHsl(rgb / scale);
            hsl.x = fract(hsl.x + p.hueShift);
            hsl.y = saturate(hsl.y * p.saturation);
            hsl.z = saturate(hsl.z + p.lightness);
            rgb = hslToRgb(hsl) * scale;
        }

        c = float4(rgb.bgr * a, a);
    }

    // With no offset this is a plain multiply, which also keeps colour held at zero alpha, as in Add.
    float gain = p.alphaSlope;
    if (p.alphaOffset != 0.0f)
        gain = c.w > 1e-5f ? saturate(c.w * p.alphaSlope + p.alphaOffset) / c.w : 0.0f;

    return sanitize(c * gain);
}

// Apart from In Front and Behind, the input builds up with every sub-step rather than replacing what's there. This
// is the strength at which still content settles where one step per frame would settle, before the
// current frame is laid over the top. A and a are the fades per frame and per sub-step.
inline float4 inputShare(float4 i, constant CompositeParams& p)
{
    const float A = p.frameAlpha, a = p.stepAlpha;
    // Add's share, which as an opacity also serves the blend modes: exactly for Add/Subtract, and
    // within a few percent for the rest.
    if (p.mode != COMPOSITE_SCREEN)
        return i * A * (1.0f - a) / (1.0f - A);
    float4 c = saturate(i);
    float4 settled = A * c / max(1.0f - A + A * c, 1e-5f);
    return settled * (1.0f - a) / max(1.0f - a * settled, 1e-5f);
}

// Combines this frame's input with the processed feedback image.
inline float4 applyComposite(float4 f, device const float4* input, int2 xy, constant CompositeParams& p)
{
    float4 i = float4(0.0);
    int2 m = xy - int2(p.inputOffsetX, p.inputOffsetY);
    if (p.inputWarped != 0)
    {
        float2 at;
        if (project(p.inputFromCanvas, float2(xy), at))
            i = bilinearPremultiplied(input, p.inputPitch, p.inputWidth, p.inputHeight, at, true);
    }
    else if (m.x >= 0 && m.y >= 0 && m.x < p.inputWidth && m.y < p.inputHeight)
    {
        // A straight pixel can hold any colour where it's transparent, and that must not show.
        i = input[m.y * p.inputPitch + m.x];
        i.xyz *= i.w;
    }
    if (p.splitsInput != 0)
        i = inputShare(i, p);

    return sanitize(blendImages(f, i, p.mode));
}
} // namespace

// The end of one step: grades the feedback image and combines it with the input, in one pass.
kernel void gradeAndComposite(
    device const float4* src [[buffer(0)]],
    device const float4* input [[buffer(1)]],
    device float4* dst [[buffer(2)]],
    constant StepParams& p [[buffer(3)]],
    uint2 xy [[thread_position_in_grid]])
{
    const float4 graded = applyGrade(src[xy.y * p.grade.srcPitch + xy.x], p.grade);
    dst[xy.y * p.composite.dstPitch + xy.x] = applyComposite(graded, input, int2(xy), p.composite);
}

// Lays the input over an image, with no grading.
kernel void composite(
    device const float4* prev [[buffer(0)]],
    device const float4* input [[buffer(1)]],
    device float4* dst [[buffer(2)]],
    constant CompositeParams& p [[buffer(3)]],
    uint2 xy [[thread_position_in_grid]])
{
    dst[xy.y * p.dstPitch + xy.x] = applyComposite(prev[xy.y * p.prevPitch + xy.x], input, int2(xy), p);
}

// Moves an image onto a differently placed canvas; anything newly in view is transparent.
kernel void reframe(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant ReframeParams& p [[buffer(2)]],
    uint2 xy [[thread_position_in_grid]])
{
    int2 s = int2(xy) + int2(p.offsetX, p.offsetY);
    float4 out = float4(0.0);
    if (s.x >= 0 && s.y >= 0 && s.x < p.srcWidth && s.y < p.srcHeight)
        out = src[s.y * p.srcPitch + s.x];
    dst[xy.y * p.dstPitch + xy.x] = out;
}

kernel void copyOut(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant CopyOutParams& p [[buffer(2)]],
    uint2 xy [[thread_position_in_grid]])
{
    float4 v;
    if (p.warped != 0)
    {
        float2 at;
        v = project(p.canvasFromOutput, float2(xy), at)
                ? bilinearPremultiplied(src, p.srcPitch, p.srcPitch, p.canvasHeight, at, false)
                : float4(0.0);
    }
    else
    {
        int2 c = int2(xy) + int2(p.offsetX, p.offsetY);
        v = src[c.y * p.srcPitch + c.x];
    }
    v.xyz = unpremultiply(v);
    dst[xy.y * p.dstPitch + xy.x] = v;
}
