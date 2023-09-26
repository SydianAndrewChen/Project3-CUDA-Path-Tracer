#pragma once
#include "sceneStructs.h"
#include <glm/glm.hpp>
#include <thrust/random.h>


enum BSDFType {
  UNIMPLEMENTED=-1,
  DIFFUSE=0,
  SPECULAR=1,
  REFRACTIVE=2,
  MICROFACET=3,
  EMISSIVE=4
};

struct BSDFStruct {
    glm::vec3 reflectance;
    float strength;
    BSDFType bsdfType;
};

class BSDF {
public:
    BSDFType bsdfType = UNIMPLEMENTED;
    __host__ __device__ BSDF(): bsdfType(UNIMPLEMENTED) {};
    __host__ __device__ BSDF(BSDFType type):bsdfType(type) {};

    static __host__ __device__ glm::vec3 hemiSphereRandomSample(thrust::default_random_engine& rng, float* pdf) {
        thrust::uniform_real_distribution<float> u01(0, 1);

        //float up = sqrt(u01(rng)); // cos(theta)
        //float over = sqrt(1 - up * up); // sin(theta)
        //float around = u01(rng) * TWO_PI;
        //*pdf = sqrt(1 - up) / PI;
        //return glm::vec3(cos(around) * over, sin(around) * over, up);

        float Xi1 = u01(rng);
        float Xi2 = u01(rng);

        float r = sqrt(Xi1);
        float theta = 2. * PI * Xi2;
        *pdf = sqrt(1 - Xi1) / PI;
        return glm::vec3(r * cos(theta), r * sin(theta), sqrt(1 - Xi1));
    }

  /**
   * Evaluate BSDF.
   * Given incident light direction wi and outgoing light direction wo. Note
   * that both wi and wo are defined in the local coordinate system at the
   * point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi incident light direction in local space of point of intersection
   * \return reflectance in the given incident/outgoing directions
   */
    __device__ virtual glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi) {
        return glm::vec3();
    };

  /**
   * Evaluate BSDF.
   * Given the outgoing light direction wo, samplea incident light
   * direction and store it in wi. Store the pdf of the sampled direction in pdf.
   * Again, note that wo and wi should both be defined in the local coordinate
   * system at the point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi address to store incident light direction
   * \param pdf address to store the pdf of the sampled incident direction
   * \return reflectance in the output incident and given outgoing directions
   */
  __device__ virtual glm::vec3 sample_f (const glm::vec3 & wo, glm::vec3 & wi, float* pdf) {
    return glm::vec3();
  };

  /**
   * Get the emission value of the surface material. For non-emitting surfaces
   * this would be a zero energy Vector3D.
   * \return emission Vector3D of the surface material
   */
  __device__ virtual glm::vec3 get_emission () const {
    return glm::vec3();
  };

  /**
   * If the BSDF is a delta distribution. Materials that are perfectly specular,
   * (e.g. water, glass, mirror) only scatter light from a single incident angle
   * to a single outgoing angle. These BSDFs are best described with alpha
   * distributions that are zero except for the single direction where light is
   * scattered.
   */
  __device__ virtual bool is_delta() const { return false; };

  __device__ virtual void debug() const {
	  printf("BSDF: %d\n", bsdfType);
  }

  __device__ virtual glm::vec3 get_debug_color() const { return glm::vec3(1.0f, 0.0f, 1.0f); }

  //__device__ virtual void render_debugger_node() {};
};

class DiffuseBSDF : public BSDF {
    glm::vec3 reflectance;
    thrust::default_random_engine rng;
public:
    __host__ __device__ DiffuseBSDF(const glm::vec3& reflectance) : reflectance(reflectance), rng(thrust::default_random_engine()) {
        bsdfType = DIFFUSE;
    }
    __device__ glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi) override {
        return reflectance * INV_PI;
    };
    __device__ glm::vec3 sample_f(const glm::vec3& wo, glm::vec3& wi, float* pdf) override {
        wi = BSDF::hemiSphereRandomSample(rng, pdf);
        return f(wo, wi);
    };
    __device__ glm::vec3 get_emission() const override { return glm::vec3(0.0f); };
    __device__ bool is_delta() const override { return false; };

    __device__ void debug() const override {
		printf("BSDF: %d\n", bsdfType);
		printf("Reflectance: %f %f %f\n", reflectance.x, reflectance.y, reflectance.z);
	}

    __device__ glm::vec3 get_debug_color() const override { return reflectance; }
};

/**
 * Emission BSDF.
 */
class EmissionBSDF : public BSDF {
public:

    __host__ __device__ EmissionBSDF(const glm::vec3 radiance):radiance(radiance) {
        bsdfType = EMISSIVE;
    };

    __device__ glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi) override {
        return glm::vec3();
    };
    __device__ glm::vec3 sample_f(const glm::vec3& wo, glm::vec3& wi, float* pdf) override {
        *pdf = INV_PI;
        wi = hemiSphereRandomSample(rng, pdf);
        return glm::vec3();
    };

    __device__ glm::vec3 get_emission() const override{ return radiance; }
    __device__ bool is_delta() const { return false; }

    __device__ void debug() const override {
    	printf("BSDF: %d\n", bsdfType);
        printf("Radiance: %f %f %f\n", radiance.x, radiance.y, radiance.z);
    }

    __device__ glm::vec3 get_debug_color() const override { return radiance;}
private:

    glm::vec3 radiance;
    thrust::default_random_engine rng;

}; // class EmissionBSDF