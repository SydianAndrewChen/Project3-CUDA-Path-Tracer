#pragma once

#include <glm/glm.hpp>
#include <thrust/random.h>
#include "utilities.h"
#include "textureStruct.h"

enum BSDFType {
    UNIMPLEMENTED = -1,
    DIFFUSE = 0,
    SPECULAR = 1,
    REFRACTIVE = 2,
    MICROFACET = 3,
    EMISSIVE = 4,
    GLOSSY = 5,

};

struct BSDFStruct {
    glm::vec3 reflectance;
    glm::vec3 emissiveFactor;
    float strength;
    BSDFType bsdfType;
    Texture * baseColorTexture;
    int baseColorTextureID = -1;
    Texture * metallicRoughnessTexture;
    int metallicRoughnessTextureID = -1;
    Texture * normalTexture;
    int normalTextureID = -1;
    float metallicFactor;
    float roughnessFactor;
    float ior;
};