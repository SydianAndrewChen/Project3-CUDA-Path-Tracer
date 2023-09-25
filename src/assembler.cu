#include "assembler.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
//// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tinygltf/tiny_gltf.h"

template<typename T>
__global__ void loadPrimitive(int childPrimitivesSize, int offset, Primitive** primitives, T* childPrimitives) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < childPrimitivesSize) {
        primitives[index + offset] = childPrimitives + index;
    }
}

__global__ void loadTriangle(int childPrimitivesSize, int offset, Primitive** primitives, Triangle* childPrimitives) {
	//int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 //   if (index < childPrimitivesSize) {
 //       auto triangle = childPrimitives[index];
 //       Primitive * p = &childPrimitives[index];
 //       //p->test();
	//	primitives[index + offset] = new Triangle(triangle.p1, triangle.p2, triangle.p3, triangle.n1, triangle.n2, triangle.n3);
 //       //primitives[index + offset] = childPrimitives + index;

 //       Primitive* p2 = primitives[index + offset];
 //       p2->test();
 //       //printf("Init\n");
	//}
}

void Scene::traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4x4 & parentTransform)
{
    if (nodeIndex < 0 || nodeIndex >= model.nodes.size()) {
        return;
    }

    const auto& node = model.nodes[nodeIndex];
    
    glm::mat4 nodeTransform = parentTransform;
    applyNodeTransform(node, nodeTransform);

    if (node.mesh >= 0) {
        const auto& mesh = model.meshes[node.mesh];
        processMesh(model, mesh, nodeTransform);
    }

    // Recursively process child nodes
    for (int childIndex : node.children) {
        traverseNode(model, childIndex, nodeTransform);
    }
}

void Scene::processMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4x4 & transform)
{
    std::cout << "Loading mesh: " << mesh.name << std::endl;
    for (const auto& primitive : mesh.primitives) {
        int p_size = mesh.primitives.size();
        const auto& indicesAccessor = model.accessors[primitive.indices];
        const auto& positionsAccessor = model.accessors[primitive.attributes.at("POSITION")];
        const auto& normalsAccessor = model.accessors[primitive.attributes.at("NORMAL")];
        const auto& uvAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];

        const auto& indicesView = model.bufferViews[indicesAccessor.bufferView];
        const auto& positionsView = model.bufferViews[positionsAccessor.bufferView];
        const auto& normalsView = model.bufferViews[normalsAccessor.bufferView];
        const auto& uvView = model.bufferViews[uvAccessor.bufferView];

        // TODO: Dynamic type array according to componentType
        const unsigned short* indexData = reinterpret_cast<const unsigned short*>(&model.buffers[indicesView.buffer].data[indicesAccessor.byteOffset + indicesView.byteOffset]);
        const float* positionData = reinterpret_cast<const float*>(&model.buffers[positionsView.buffer].data[positionsAccessor.byteOffset + positionsView.byteOffset]);
        const float* normalData = reinterpret_cast<const float*>(&model.buffers[normalsView.buffer].data[normalsAccessor.byteOffset + normalsView.byteOffset]);
        const float* uvData = reinterpret_cast<const float*>(&model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]);

        const size_t vertexStride = 3;
        const size_t normalStride = 3;
        const size_t uvStride = 3;

        const size_t numIndices = indicesAccessor.count;
        glm::mat4x4 normalTransform = glm::transpose(glm::inverse(transform));
        int materialID = primitive.material;
        if (materialID < 0) {
            materialID = 0;
        }
        printf("Material: %d\n", materialID);
        // Iterate through indices and create triangles
        //for (size_t i = 0; i < numIndices; i += 3) {
        for (size_t i = 0; i < numIndices; i += 3) {
            Triangle triangle;
            triangle.p1 = glm::vec3(positionData[indexData[i] * vertexStride], positionData[indexData[i] * vertexStride + 1], positionData[indexData[i] * vertexStride + 2]);
            triangle.p2 = glm::vec3(positionData[indexData[i + 1] * vertexStride], positionData[indexData[i + 1] * vertexStride + 1], positionData[indexData[i + 1] * vertexStride + 2]);
            triangle.p3 = glm::vec3(positionData[indexData[i + 2] * vertexStride], positionData[indexData[i + 2] * vertexStride + 1], positionData[indexData[i + 2] * vertexStride + 2]);
            
            triangle.p1 = glm::vec3(transform * glm::vec4(triangle.p1, 1.0f));
            triangle.p2 = glm::vec3(transform * glm::vec4(triangle.p2, 1.0f));
            triangle.p3 = glm::vec3(transform * glm::vec4(triangle.p3, 1.0f));

            triangle.n1 = glm::vec3(normalData[indexData[i] * normalStride], normalData[indexData[i] * normalStride + 1], normalData[indexData[i] * normalStride + 2]);
            triangle.n2 = glm::vec3(normalData[indexData[i + 1] * normalStride], normalData[indexData[i + 1] * normalStride + 1], normalData[indexData[i + 1] * normalStride + 2]);
            triangle.n3 = glm::vec3(normalData[indexData[i + 2] * normalStride], normalData[indexData[i + 2] * normalStride + 1], normalData[indexData[i + 2] * normalStride + 2]);

            triangle.n1 = glm::vec3(normalTransform * glm::vec4(triangle.n1, 0.0f));
            triangle.n1 = glm::vec3(normalTransform * glm::vec4(triangle.n1, 0.0f));
            triangle.n1 = glm::vec3(normalTransform * glm::vec4(triangle.n1, 0.0f));

            triangle.uv1 = glm::vec2(uvData[indexData[i] * uvStride], uvData[indexData[i] * uvStride + 1]);
            triangle.uv2 = glm::vec2(uvData[indexData[i + 1] * uvStride], uvData[indexData[i + 1] * uvStride + 1]);
            triangle.uv3 = glm::vec2(uvData[indexData[i + 2] * uvStride], uvData[indexData[i + 2] * uvStride + 1]);
            //auto index0 = indexData[i];
            //auto index1 = indexData[i + 1];
            //auto index2 = indexData[i + 2];
            triangle.materialID = materialID;
            triangles.push_back(triangle);
        }
    }
}

void Scene::applyNodeTransform(const tinygltf::Node & node, glm::mat4x4& parentTransform)
{
    glm::mat4 localTransform(1.0f);
    glm::mat4 T(1.0f), R(1.0f), S(1.0f);

    if (!node.translation.empty()) {
        const auto& translation = node.translation;
        T = glm::translate(localTransform, glm::vec3(translation[0], translation[1], translation[2]));
    }

    if (!node.rotation.empty()) {
        const auto& rotation = node.rotation;
        // Please note: 
        // tinygltf stores quaternions in the order w, x, y, z
        // glm::mat constructor in the order x, y, z, w (It seems that glm quat module is not even internally consistent...)
        // Ref: https://stackoverflow.com/questions/48348509/glmquat-why-the-order-of-x-y-z-w-components-are-mixed
        glm::quat rotationQuaternion(rotation[3], rotation[0], rotation[1], rotation[2]);
        R = glm::mat4_cast(rotationQuaternion);
    }

    if (!node.scale.empty()) {
        const auto& scale = node.scale;
        S = glm::scale(localTransform, glm::vec3(scale[0], scale[1], scale[2]));
    }
    localTransform = T * R * S;
    // Update the parent transformation matrix with the node's transformation
    parentTransform = parentTransform * localTransform;
}

Scene::Scene(const char* filename)
{
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    //bool success = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    bool success = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) {
        std::cout << "Warn: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "Error: " << err << std::endl;
        assert(0);
    }

    if (!success) {
        std::cerr << "Failed to load glTF model." << std::endl;
        assert(0);
    }
    loadMaterials();
}

void Scene::movePrimitivesToDevice()
{
    cudaMalloc(&dev_primitives, sizeof(Primitive*) * getPrimitiveSize());
    int offset = 0;
    cudaMalloc(&dev_triangles, sizeof(Triangle) * triangles.size());
    cudaMemcpy(dev_triangles, triangles.data(), sizeof(Triangle) * triangles.size(), cudaMemcpyHostToDevice);
    int blockSize = 128;
    dim3 loadPrimitiveBlock((triangles.size() + (blockSize - 1)) / blockSize);
    //loadPrimitive << <loadPrimitiveBlock, blockSize >> > (triangles.size(), offset, dev_primitives, dev_triangles);
    loadTriangle << <loadPrimitiveBlock, blockSize >> > (triangles.size(), offset, dev_primitives, dev_triangles);
    offset += triangles.size();
    //cudaFree(dev_triangles);
    //cudaMalloc(&dev_spheres, sizeof(Sphere) * spheres.size());
    //cudaMemcpy(dev_spheres, spheres.data(), sizeof(Sphere) * spheres.size(), cudaMemcpyHostToDevice);
    //loadPrimitiveBlock = dim3(spheres.size() + (blockSize - 1) / blockSize, blockSize);
    //loadPrimitive << <loadPrimitiveBlock, blockSize >> > (spheres.size(), offset, dev_primitives, dev_spheres);
    //offset += spheres.size();
}

void Scene::assembleScenePrimitives()
{
    //auto model = scene->model;
    auto initTransform = glm::mat4x4(1.0f);
    for (auto node : model.scenes[0].nodes) {
        traverseNode(model, node, initTransform);
    }
    return;
}

void Scene::loadMaterials()
{
    for ( const auto & material:model.materials)
    {
        //BSDF* bsdf = nullptr;
        BSDFStruct bsdfStruct;
        if (*std::max_element(material.emissiveFactor.begin(), material.emissiveFactor.end()) > DBL_EPSILON) {
            auto ext = material.extensions.find("KHR_materials_emissive_strength");
            float strength;
            if (ext != material.extensions.end()) {
                auto strengthObject = ext->second.Get<tinygltf::Value::Object>().find("emissiveStrength");
                if (strengthObject != ext->second.Get<tinygltf::Value::Object>().end()) {
					float strength = static_cast<float>(strengthObject->second.Get<double>());
                    bsdfStruct.reflectance = glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]);
                    bsdfStruct.strength = strength;
                    bsdfStruct.bsdfType = EMISSIVE;
                    bsdfStructs.push_back(bsdfStruct);
				}
            }
            //bsdf = new EmissionBSDF(strength * glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]));
        }

        else {
			//bsdf = new DiffuseBSDF(glm::vec3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1], material.pbrMetallicRoughness.baseColorFactor[2]));
            bsdfStruct.bsdfType = DIFFUSE;
            bsdfStruct.reflectance = glm::vec3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1], material.pbrMetallicRoughness.baseColorFactor[2]);
            bsdfStructs.push_back(bsdfStruct);
        }
        //bsdfs.push_back(bsdf);

    }
}


__global__ void initBSDF(int size, BSDF** bsdfs, BSDFStruct * bsdfStructs) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size) {
        BSDFType type = bsdfStructs[index].bsdfType;
        printf("bsdfStructs[index].bsdfType: %d\n", bsdfStructs[index].bsdfType);
        printf("bsdfStructs[index].reflectance: %f, %f, %f\n", bsdfStructs[index].reflectance.x, bsdfStructs[index].reflectance.y, bsdfStructs[index].reflectance.z);
        printf("bsdfStructs[index].strength: %f\n", bsdfStructs[index].strength);
        switch (type)
        {
        case UNIMPLEMENTED:
            break;
        case DIFFUSE:
            bsdfs[index] = new DiffuseBSDF(bsdfStructs[index].reflectance);
            break;
        case SPECULAR:
            assert(0);
            break;
        case REFRACTIVE:
            assert(0);
            break;
        case MICROFACET:
            assert(0);
            break;
        case EMISSIVE:
            bsdfs[index] = new EmissionBSDF(bsdfStructs[index].reflectance * bsdfStructs[index].strength);
            break;
        default:
            assert(0);
            break;
        }
    }
}

void Scene::initBSDFs()
{
    cudaMalloc(&dev_bsdfs, bsdfStructs.size() * sizeof(BSDF*));
    cudaMalloc(&dev_bsdfStructs, bsdfStructs.size() * sizeof(BSDFStruct));
    cudaMemcpy(dev_bsdfStructs, bsdfStructs.data(), bsdfStructs.size() * sizeof(BSDFStruct), cudaMemcpyHostToDevice);
    int blockSize = 128;
    dim3 initBSDFBlocks((bsdfStructs.size() + (blockSize-1))/blockSize);
    initBSDF<<<initBSDFBlocks, blockSize>>>(bsdfStructs.size(), dev_bsdfs, dev_bsdfStructs);
    cudaFree(dev_bsdfStructs);
}

__global__ void freePrimitive(Primitive** primitives, int size) {
    for (size_t i = 0; i < size; i++)
    {
        delete primitives[i];
    }
}

void Scene::freeBuffer()
{
    int blockSize = 128;
    dim3 loadPrimitiveBlock((triangles.size() + (blockSize - 1)) / blockSize);
    //freePrimitive << <loadPrimitiveBlock, blockSize >> > (dev_primitives, getPrimitiveSize());
    freePrimitive << <1, 1>> > (dev_primitives, getPrimitiveSize());
    //for (size_t i = 0; i < getPrimitiveSize(); i++)
    //{
    //    delete dev_primitives[i];
    //}
    cudaFree(dev_primitives);
}
