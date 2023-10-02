#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include "tinygltf/tiny_gltf.h"

#include "sceneStructs.h"
#include "bsdfStruct.h"
#include "textureStruct.h"
#include "camera.h"

enum GLTFDataType {
	GLTF_DATA_TYPE_SIGNED_BYTE = 5120,
    GLTF_DATA_TYPE_UNSIGNED_BYTE = 5121,
    GLTF_DATA_TYPE_SHORT = 5122,
    GLTF_DATA_TYPE_UNSIGNED_SHORT = 5123,
    GLTF_DATA_TYPE_UNSIGNED_INT = 5125,
    GLTF_DATA_TYPE_FLOAT = 5126
};

class Scene {
    std::vector<Sphere> spheres;

    void applyNodeTransform(const tinygltf::Node& node, glm::mat4x4& parentTransform);
    void traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4x4 & localTransform);

    void processMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4x4 & parentTransform);
    void processCamera(const tinygltf::Model & model, const tinygltf::Camera & sceneCamera, const glm::mat4x4 & parentTransform);
    tinygltf::Model model;
public:
    Scene(const char * filename);
    RenderState state;
    Camera camera;
    std::vector<Triangle> triangles;
    std::vector<BSDFStruct> bsdfStructs;
    std::vector<TextureInfo> textures;
    Triangle* dev_triangles;
    Sphere* dev_spheres;
    Primitive** dev_primitives;
    int getPrimitiveSize() const{
        return triangles.size() + spheres.size();
    }
    void initTriangles();
    void initBSDFs();
    void initTextures();
    void initCamera();
};



