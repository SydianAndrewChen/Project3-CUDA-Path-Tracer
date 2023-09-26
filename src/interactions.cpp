#include "interactions.h"


__host__ __device__
glm::vec3 hemiSphereRandomSample(thrust::default_random_engine& rng, float* pdf) {
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