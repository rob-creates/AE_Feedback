#include "LayerMotion.h"

#include <cmath>
#include <cstring>


Projective Projective::after(const Projective& first) const
{
    Projective r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r.m[i * 3 + j] = m[i * 3] * first.m[j] + m[i * 3 + 1] * first.m[3 + j] + m[i * 3 + 2] * first.m[6 + j];
    return r;
}

bool Projective::inverse(Projective* out) const
{
    const double* a = m;
    const double c0 = a[4] * a[8] - a[5] * a[7], c1 = a[5] * a[6] - a[3] * a[8], c2 = a[3] * a[7] - a[4] * a[6];
    const double det = a[0] * c0 + a[1] * c1 + a[2] * c2;
    if (std::abs(det) < 1e-12)
        return false;
    const double d = 1.0 / det;
    out->m[0] = c0 * d;
    out->m[1] = (a[2] * a[7] - a[1] * a[8]) * d;
    out->m[2] = (a[1] * a[5] - a[2] * a[4]) * d;
    out->m[3] = c1 * d;
    out->m[4] = (a[0] * a[8] - a[2] * a[6]) * d;
    out->m[5] = (a[2] * a[3] - a[0] * a[5]) * d;
    out->m[6] = c2 * d;
    out->m[7] = (a[1] * a[6] - a[0] * a[7]) * d;
    out->m[8] = (a[0] * a[4] - a[1] * a[3]) * d;
    return true;
}

bool Projective::apply(double x, double y, double* outX, double* outY) const
{
    const double w = m[6] * x + m[7] * y + m[8];
    if (w <= 1e-9)
        return false;
    *outX = (m[0] * x + m[1] * y + m[2]) / w;
    *outY = (m[3] * x + m[4] * y + m[5]) / w;
    return true;
}

void Projective::toFloats(float out[9]) const
{
    for (int i = 0; i < 9; i++)
        out[i] = float(m[i]);
}

LayerMotion::LayerMotion(PF_InData* in_data, PF_OutData* out_data, AEGP_LayerIDVal other)
    : mInData(in_data)
    , mOutData(out_data)
{
    try
    {
        mFace.emplace(in_data, kAEGPPFInterfaceSuite, kAEGPPFInterfaceSuiteVersion1, out_data);
        mLayers.emplace(in_data, kAEGPLayerSuite, kAEGPLayerSuiteVersion9, out_data);
        auto& face = *mFace;
        auto& layers = *mLayers;
        AEFX_SuiteScoper<AEGP_CompSuite13> comps(in_data, kAEGPCompSuite, kAEGPCompSuiteVersion13, out_data);
        AEFX_SuiteScoper<AEGP_ItemSuite9> items(in_data, kAEGPItemSuite, kAEGPItemSuiteVersion9, out_data);

        AEGP_CompH comp = nullptr;
        AEGP_ItemH item = nullptr;
        AEGP_LayerFlags flags = 0;
        AEGP_ObjectType type = AEGP_ObjectType_NONE;
        if (face->AEGP_GetEffectLayer(in_data->effect_ref, &mLayer) || !mLayer ||
            layers->AEGP_GetLayerFlags(mLayer, &flags) || layers->AEGP_GetLayerObjectType(mLayer, &type) ||
            layers->AEGP_GetLayerParentComp(mLayer, &comp) || comps->AEGP_GetItemFromComp(comp, &item) ||
            items->AEGP_GetItemDimensions(item, &mCompWidth, &mCompHeight))
            return;
        if (other != AEGP_LayerIDVal_NONE &&
            (layers->AEGP_GetLayerFromLayerID(comp, other, &mLayer) || !mLayer || layers->AEGP_GetLayerFlags(mLayer, &flags) ||
             layers->AEGP_GetLayerObjectType(mLayer, &type)))
            return;
        mDrawnInComp =
            type == AEGP_ObjectType_TEXT || type == AEGP_ObjectType_VECTOR || (flags & AEGP_LayerFlag_COLLAPSE) != 0;
        mUsable = mCompWidth > 0 && mCompHeight > 0 && (flags & AEGP_LayerFlag_LAYER_IS_3D) == 0;
    }
    catch (...)
    {
        mUsable = false;
    }
}

bool LayerMotion::layerToComp(A_long value, A_u_long scale, Projective* out)
{
    if (!mUsable)
        return false;
    if (mDrawnInComp)
    {
        *out = Projective{};
        return true;
    }
    const auto key = std::make_pair(value, scale);
    if (auto found = mRead.find(key); found != mRead.end())
    {
        *out = found->second;
        return true;
    }
    A_Time compTime{};
    A_Matrix4 world{};
    if ((*mFace)->AEGP_ConvertEffectToCompTime(mInData->effect_ref, value, scale, &compTime) ||
        (*mLayers)->AEGP_GetLayerToWorldXform(mLayer, &compTime, &world))
        return false;
    // A 2D layer's world is the comp itself. AE's matrices act on row vectors, (x, y, z, 1) * M.
    *out = Projective{{world.mat[0][0], world.mat[1][0], world.mat[3][0], world.mat[0][1], world.mat[1][1],
                       world.mat[3][1], 0, 0, 1}};
    mRead.emplace(key, *out);
    return true;
}

uint64_t LayerMotion::fingerprint(A_long value, A_u_long scale)
{
    Projective p;
    if (!layerToComp(value, scale, &p))
        return 0;
    uint64_t hash = 1469598103934665603ull; // FNV-1a
    unsigned char bytes[sizeof(p.m)];
    std::memcpy(bytes, p.m, sizeof(p.m));
    for (unsigned char b : bytes)
        hash = (hash ^ b) * 1099511628211ull;
    return hash;
}

CompWatch::CompWatch(PF_InData* in_data, PF_OutData* out_data)
    : mInData(in_data)
{
    try
    {
        mFace.emplace(in_data, kAEGPPFInterfaceSuite, kAEGPPFInterfaceSuiteVersion1, out_data);
        AEFX_SuiteScoper<AEGP_LayerSuite9> layers(in_data, kAEGPLayerSuite, kAEGPLayerSuiteVersion9, out_data);
        AEFX_SuiteScoper<AEGP_CompSuite13> comps(in_data, kAEGPCompSuite, kAEGPCompSuiteVersion13, out_data);
        AEGP_LayerH layer = nullptr;
        AEGP_LayerFlags flags = 0;
        AEGP_CompH comp = nullptr;
        AEGP_ItemH item = nullptr;
        if ((*mFace)->AEGP_GetEffectLayer(in_data->effect_ref, &layer) || !layer ||
            layers->AEGP_GetLayerFlags(layer, &flags) || (flags & AEGP_LayerFlag_ADJUSTMENT_LAYER) == 0 ||
            layers->AEGP_GetLayerParentComp(layer, &comp) || comps->AEGP_GetItemFromComp(comp, &item))
            return;
        mRender.emplace(in_data, kAEGPRenderSuite, kAEGPRenderSuiteVersion5, out_data);
        mHaveNow = (*mRender)->AEGP_GetCurrentTimestamp(&mNow) == A_Err_NONE;
        mCompItem = item;
    }
    catch (...)
    {
        mCompItem = nullptr;
    }
}

bool CompWatch::timestamp(AEGP_TimeStamp* out) const
{
    if (!mHaveNow)
        return false;
    *out = mNow;
    return true;
}

bool CompWatch::unchangedSince(const AEGP_TimeStamp& made, A_long endExclusive) const
{
    if (!mCompItem)
        return false;
    if (mHaveNow && std::memcmp(&made, &mNow, sizeof(made)) == 0)
        return true;
    A_Time end{};
    if ((*mFace)->AEGP_ConvertEffectToCompTime(mInData->effect_ref, endExclusive, A_u_long(mInData->time_scale), &end))
        return false;
    // From the comp's start, which takes in any time before the layer's.
    const A_Time start{0, end.scale};
    A_Boolean changed = true;
    if ((*mRender)->AEGP_HasItemChangedSinceTimestamp(mCompItem, &start, &end, &made, &changed))
        return false;
    return !changed;
}
