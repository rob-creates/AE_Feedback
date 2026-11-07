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
} // namespace

kernel void displace(
    device const float4* src [[buffer(0)]],
    device const float4* map [[buffer(1)]],
    device float4* dst [[buffer(2)]],
    constant DisplaceParams& p [[buffer(3)]],
    uint2 xy [[thread_position_in_grid]])
{
    int2 canvas = int2(xy);
    int2 m = canvas - int2(p.mapOffsetX, p.mapOffsetY);
    float2 shift = float2(0.0);
    if (m.x >= 0 && m.y >= 0 && m.x < p.mapWidth && m.y < p.mapHeight)
    {
        float4 v = map[m.y * p.mapPitch + m.x];
        shift = float2(
            mapShift(v, p.channelX, p.mapIsStraight != 0) * p.amountX,
            mapShift(v, p.channelY, p.mapIsStraight != 0) * p.amountY);
    }
    float2 pos = float2(canvas) + shift;
    float4 out = float4(0.0);
    // Anything displaced in from beyond the layer is transparent.
    if (pos.x > -1.0f && pos.y > -1.0f && pos.x < float(p.width) && pos.y < float(p.height))
        out = bilinearClamped(src, p.srcPitch, p.width, p.height, pos);
    dst[xy.y * p.dstPitch + xy.x] = out;
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

kernel void blurUp(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant BlurResizeParams& p [[buffer(2)]],
    uint2 xy [[thread_position_in_grid]])
{
    float2 pos = (float2(xy) + 0.5f) / float(p.factor) - 0.5f;
    dst[xy.y * p.dstPitch + xy.x] = bilinearClamped(src, p.srcPitch, p.srcWidth, p.srcHeight, pos);
}

// Levels, then HSL, then Alpha.
kernel void grade(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant GradeParams& p [[buffer(2)]],
    uint2 xy [[thread_position_in_grid]])
{
    float4 c = src[xy.y * p.srcPitch + xy.x];

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

    dst[xy.y * p.dstPitch + xy.x] = sanitize(c * gain);
}

// Combines this frame's input with the processed feedback image.
kernel void composite(
    device const float4* prev [[buffer(0)]],
    device const float4* input [[buffer(1)]],
    device float4* dst [[buffer(2)]],
    constant CompositeParams& p [[buffer(3)]],
    uint2 xy [[thread_position_in_grid]])
{
    float4 f = prev[xy.y * p.prevPitch + xy.x];
    float4 i = float4(0.0);
    int2 m = int2(xy) - int2(p.inputOffsetX, p.inputOffsetY);
    if (m.x >= 0 && m.y >= 0 && m.x < p.inputWidth && m.y < p.inputHeight)
    {
        // A straight pixel can hold any colour where it's transparent, and that must not show.
        i = input[m.y * p.inputPitch + m.x];
        i.xyz *= i.w;
        i *= p.inputGain;
    }

    // Screen builds up non-linearly, so each sub-step's strength depends on the input's value. This
    // is the strength at which still content settles where one step per frame would settle, before
    // the current frame is laid over the top: settled = c / (1 - A + A c), feedback = A * settled.
    if (p.screenGain != 0)
    {
        float4 c = saturate(i);
        float4 feedback = p.frameAlpha * c / max(1.0f - p.frameAlpha + p.frameAlpha * c, 1e-5f);
        i = feedback * (1.0f - p.stepAlpha) / max(1.0f - p.stepAlpha * feedback, 1e-5f);
    }

    float4 out;
    switch (p.mode)
    {
    case COMPOSITE_INPUT_BEHIND: out = f + i * (1.0f - f.w); break;
    case COMPOSITE_ADD: out = f + i; break;
    case COMPOSITE_SCREEN: out = f + i - f * i; break;
    default: out = i + f * (1.0f - i.w); break;
    }
    dst[xy.y * p.dstPitch + xy.x] = sanitize(out);
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
    int2 c = int2(xy) + int2(p.offsetX, p.offsetY);
    float4 v = src[c.y * p.srcPitch + c.x];
    v.xyz = unpremultiply(v);
    dst[xy.y * p.dstPitch + xy.x] = v;
}
