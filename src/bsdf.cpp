#include "bsdf.h"
#include "bsdf.h"


__host__ __device__ DiffuseBSDF(const glm::vec3& reflectance) : reflectance(reflectance), rng(thrust::default_random_engine()) {
    bsdfType = DIFFUSE;
}

//__host__ __device__ DiffuseBSDF(const glm::vec3& reflectance) : reflectance(reflectance), rng(thrust::default_random_engine()) {
//    bsdfType = DIFFUSE;
//}

__device__ glm::vec3 DiffuseBSDF::f(const glm::vec3& wo, const glm::vec3& wi)
{
	return reflectance * INV_PI;
}

__device__ glm::vec3 DiffuseBSDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float* pdf)
{
	wi = hemiSphereRandomSample(rng, pdf);
	return f(wo, wi);
}

__host__ __device__ EmissionBSDF(const glm::vec3 radiance) : radiance(radiance) {
	bsdfType = EMISSIVE;
}

__device__ glm::vec3 EmissionBSDF::f(const glm::vec3& wo, const glm::vec3& wi)
{
	return glm::vec3();
}

__device__ glm::vec3 EmissionBSDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float* pdf)
{
	*pdf = INV_PI;
	wi = hemiSphereRandomSample(rng, pdf);
	return glm::vec3();
}
