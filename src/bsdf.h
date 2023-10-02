#pragma once

#include "bsdfStruct.h"
#include "texture.h"

#define Clamp(x, min, max) (x < min ? min : (x > max ? max : x))

// BSDF Inline Functions
__device__ inline float CosTheta(const glm::vec3& w) { return w.z; }
__device__ inline float Cos2Theta(const glm::vec3& w) { return w.z * w.z; }
__device__ inline float AbsCosTheta(const glm::vec3& w) { return std::abs(w.z); }
__device__ inline float Sin2Theta(const glm::vec3& w) {
    return __max((float)0, (float)1 - Cos2Theta(w));
}

__device__ inline float SinTheta(const glm::vec3& w) { return std::sqrt(Sin2Theta(w)); }

__device__ inline float TanTheta(const glm::vec3& w) { return SinTheta(w) / CosTheta(w); }

__device__ inline float Tan2Theta(const glm::vec3& w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

__device__ inline float CosPhi(const glm::vec3& w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);
}

__device__ inline float SinPhi(const glm::vec3& w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);
}

__device__ inline float Cos2Phi(const glm::vec3& w) { return CosPhi(w) * CosPhi(w); }

__device__ inline float Sin2Phi(const glm::vec3& w) { return SinPhi(w) * SinPhi(w); }

__device__ inline float CosDPhi(const glm::vec3& wa, const glm::vec3& wb) {
    float waxy = wa.x * wa.x + wa.y * wa.y;
    float wbxy = wb.x * wb.x + wb.y * wb.y;
    if (waxy == 0 || wbxy == 0)
        return 1;
    return Clamp((wa.x * wb.x + wa.y * wb.y) / std::sqrt(waxy * wbxy), -1, 1);
}


__device__ float Lambda(const glm::vec3& w, const float alpha) {
    float absTanTheta = std::abs(TanTheta(w));
    if (glm::isinf(absTanTheta)) return 0.;
        float alpha_lambda = glm::sqrt(Cos2Phi(w) +
            Sin2Phi(w)) * alpha;

    float a = 1 / (alpha_lambda * absTanTheta);
    if (a >= 1.6f)
        return 0;
    return (1 - 1.259f * a + 0.396f * a * a) /
        (3.535f * a + 2.181f * a * a);
    
    // Beckmann Spizzichino 
    //float theta = acos(w.z);
    //float a = 1.0 / (alpha * tan(theta));
    //return 0.5f * (erf(a) - 1.0f + exp(-a * a) / (a * sqrt(PI)));
}

__device__ float microfacetBSDF_G(const glm::vec3 & wo, const glm::vec3 & wi, const float alpha) {
    return 1.0 / (1.0 + Lambda(wo, alpha) + Lambda(wi, alpha));
}

__device__ float microfacetBSDF_D(const glm::vec3 & h, const float alpha) {
    float alpha_square = alpha * alpha;
    float cos_theta_h_square = h.z * h.z;
    float tan_theta_h_square = 1.0f / (cos_theta_h_square)-1.0f;
    return exp(-tan_theta_h_square / alpha_square) / (PI * alpha_square * cos_theta_h_square * cos_theta_h_square);
}

template<typename T>
__device__ T Schlick(const T & R0, float cos_theta) {
	return R0 + (1.0f - R0) * pow(1.0f - cos_theta, 5.0f);
}

__device__ float roughnessToAlpha(float roughness) {
    roughness = __max(roughness, 1e-3);
    float x = glm::log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x +
        0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

__device__ glm::vec3 microfacetBSDF_F(const glm::vec3& h, const float ior, const glm::vec3 & baseColor, const float metallic) {
    auto F0 = glm::mix(glm::vec3(0.04f), baseColor, metallic);
    return Schlick(F0, h.z);
    //if (metallic > 0.1f) {
    //    auto F0 = glm::mix(glm::vec3(0.4f), baseColor, metallic);
    //    return Schlick(F0, h.z);
    //}
    //else {
    //    float R0 = powf((1.0f - ior) / (1.0f + ior), 2);
    //    return glm::vec3(Schlick(R0, h.z));

    //}
    //auto eta2_add_k2 = eta * eta + k * k;
    //auto cosine_square = wi.z * wi.z;
    //auto eta_times_cosine = eta * wi.z;
    //auto Rs = (eta2_add_k2 - 2.0f * eta_times_cosine + cosine_square) / ((eta2_add_k2 + 2.0f * eta_times_cosine + cosine_square) + EPSILON);
    //auto Rp = ((eta2_add_k2)*cosine_square - 2.0f * eta_times_cosine + 1.0f) / (((eta2_add_k2)*cosine_square + 2.0f * eta_times_cosine + 1.0f) + EPS_D);
    //return (Rs + Rp) / 2.;
}

__device__ glm::vec3 f(BSDFStruct& bsdfStruct, const glm::vec3& wo, glm::vec3& wi, const glm::vec2 & uv) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
        if (bsdfStruct.baseColorTextureID != -1) {
            return sampleTextureRGB(*bsdfStruct.baseColorTexture, uv) * INV_PI;
		}
		else
			return bsdfStruct.reflectance * INV_PI;
    case MICROFACET:
        glm::vec3 baseColor;
        glm::vec2 metallicRoughness;
        if (bsdfStruct.baseColorTextureID != -1) {
            baseColor = sampleTextureRGB(*bsdfStruct.baseColorTexture, uv);
		}
		else
			baseColor = bsdfStruct.reflectance;

        if (bsdfStruct.metallicRoughnessTextureID != -1) {
            metallicRoughness = sampleTextureRG(*bsdfStruct.metallicRoughnessTexture, uv);
        }
        else {
            metallicRoughness = glm::vec2(bsdfStruct.metallicFactor, bsdfStruct.roughnessFactor);
        }
        auto h = glm::normalize(wo + wi);
        auto F = microfacetBSDF_F(h, bsdfStruct.ior, baseColor, metallicRoughness.x);
        auto alpha = roughnessToAlpha(metallicRoughness.y);
        auto D = microfacetBSDF_D(h, alpha);
        auto G = microfacetBSDF_G(wo, wi, alpha);
        auto result = baseColor * F * D * G / (4.0f * wo.z * wi.z);
        //printf("F: %f, D: %f, G: %f\n", F, D, G);
        //printf("result: %f, %f, %f\n", result.x, result.y, result.z);
        //return result;
        return baseColor * INV_PI;
    case EMISSIVE:
        return bsdfStruct.emissiveFactor * bsdfStruct.strength;
    case SPECULAR:
        if (bsdfStruct.baseColorTextureID != -1) {
            return sampleTextureRGB(*bsdfStruct.baseColorTexture, uv) * INV_PI;
        }
        else
            return bsdfStruct.reflectance * INV_PI;
    default:
        printf("BSDF not implemented!\n");
        assert(0);
        return glm::vec3();
    }
}

__device__ glm::vec3 sample_f(BSDFStruct& bsdfStruct, const glm::vec3& wo, glm::vec3& wi, float* pdf, thrust::default_random_engine & rng, const glm::vec2 & uv) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
    {
        wi = hemiSphereRandomSample(rng, pdf);
        // Encapsulate a sampler class
        return f(bsdfStruct, wo, wi, uv);
    }
    case MICROFACET:
        // Uniform sampling for now
        // TODO: Add importance sampling
        wi = hemiSphereRandomSample(rng, pdf);
        return f(bsdfStruct, wo, wi, uv);

        auto s = hemiSphereRandomSample(rng, pdf);
        float alpha = bsdfStruct.roughnessFactor * bsdfStruct.roughnessFactor;
        auto theta_h = atan(sqrt(-alpha * alpha * log(1 - s.x)));
        auto phi_h = 2 * PI * s.y;
        auto sin_theta_h = sin(theta_h);
        auto cos_theta_h = cos(theta_h);
        glm::vec3 h(sin_theta_h * cos(phi_h), sin_theta_h * sin(phi_h), cos_theta_h);
        auto proj = dot(wo, h) * h;
        wi = 2.0f * proj - wo;
        if (wi.z < 0) return glm::vec3();

        auto p_theta = 2 * sin_theta_h * exp(-pow((sin_theta_h/cos_theta_h) / alpha, 2)) / (alpha * alpha * pow(cos_theta_h, 3));
        auto p_phi = INV_2PI;
        auto p_h = p_theta * p_phi / sin_theta_h;
        *pdf = p_h / (4 * dot(wi, h));
        
        return f(bsdfStruct, wo, wi, uv);
        
    case EMISSIVE:
        return f(bsdfStruct, wo, wi, uv);
    case SPECULAR:
		wi = wo;
		wi.x = -wi.x;
		wi.y = -wi.y;
        *pdf = 0.1f;
		return f(bsdfStruct, wo, wi, uv);
    default:
        return glm::vec3();
        break;
    }
}

__device__ glm::vec3 get_debug_color(BSDFStruct& bsdfStruct) {
	switch (bsdfStruct.bsdfType)
	{
	case DIFFUSE:
		return glm::vec3(1.0f, 0.0f, 0.0f);
	case EMISSIVE:
		return glm::vec3(0.0f, 1.0f, 0.0f);
	default:
		return glm::vec3(0.0f, 0.0f, 1.0f);
	}
}