#pragma once
#include <vector>
#include "scene.h"
#include "sceneStructs.h"
#include "bsdf.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Scene {
    std::vector<Sphere> spheres;
    std::vector<BSDF*> bsdfs;
    std::vector<BSDFStruct> bsdfStructs;
    BSDFStruct * dev_bsdfStructs;
    void applyNodeTransform(const tinygltf::Node& node, glm::mat4x4& parentTransform);
    void traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4x4 & localTransform);

    void processMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4x4 & parentTransform);
    tinygltf::Model model;

    BSDF** dev_bsdfs;
public:
    Scene(const char* filename);
    std::vector<Triangle> triangles;
    void movePrimitivesToDevice();
    Triangle* dev_triangles;
    Sphere* dev_spheres;
    Primitive** dev_primitives;
    int getPrimitiveSize() const{
        return triangles.size() + spheres.size();
    }
    void assembleScenePrimitives();
    void loadMaterials();
    void initBSDFs();
    void freeBuffer();
};