#pragma once

#include <thrust/random.h>
#include <glm/glm.hpp>
#include "utilities.h"

__host__ __device__ glm::vec3 hemiSphereRandomSample(thrust::default_random_engine& rng, float* pdf);

__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng);