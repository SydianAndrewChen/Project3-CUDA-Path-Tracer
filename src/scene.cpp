#include "scene.h"

//// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
//// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tinygltf/tiny_gltf.h"


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

    initTextures();
    initBSDFs();
    initTriangles();
    initCamera();
}

void Scene::initTriangles()
{
    auto initTransform = glm::mat4x4(1.0f);
    for (auto node : model.scenes[0].nodes) {
        traverseNode(model, node, initTransform);
    }
}

void Scene::initBSDFs() {
    for (const auto& material : model.materials)
    {
        printf("material name: %s\n", material.name.c_str());
        //BSDF* bsdf = nullptr;
        BSDFStruct bsdfStruct;
        bsdfStruct.normalTextureID = material.normalTexture.index;
        if (*std::max_element(material.emissiveFactor.begin(), material.emissiveFactor.end()) > DBL_EPSILON) {
            auto ext = material.extensions.find("KHR_materials_emissive_strength");
            float strength;
            if (ext != material.extensions.end()) {
                auto strengthObject = ext->second.Get<tinygltf::Value::Object>().find("emissiveStrength");
                if (strengthObject != ext->second.Get<tinygltf::Value::Object>().end()) {
                    bsdfStruct.bsdfType = EMISSIVE;
                    float strength = static_cast<float>(strengthObject->second.Get<double>());
                    bsdfStruct.emissiveFactor = glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]);
                    bsdfStruct.strength = strength;
                    bsdfStructs.push_back(bsdfStruct);
                }
            }
            //bsdf = new EmissionBSDF(strength * glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]));
        }
		else if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
            bsdfStruct.bsdfType = MICROFACET;
			bsdfStruct.reflectance = glm::vec3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1], material.pbrMetallicRoughness.baseColorFactor[2]);
			bsdfStruct.baseColorTextureID = material.pbrMetallicRoughness.baseColorTexture.index;
			bsdfStruct.metallicRoughnessTextureID = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            bsdfStruct.metallicFactor = material.pbrMetallicRoughness.metallicFactor;
            bsdfStruct.roughnessFactor = material.pbrMetallicRoughness.roughnessFactor;
            bsdfStruct.ior = 1.5f;
			bsdfStructs.push_back(bsdfStruct);
		}
        else if (/* TODO: Add condition for specular material here */false) {
            bsdfStruct.bsdfType = SPECULAR;
            bsdfStruct.reflectance = glm::vec3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1], material.pbrMetallicRoughness.baseColorFactor[2]);
            bsdfStruct.baseColorTextureID = material.pbrMetallicRoughness.baseColorTexture.index;
            bsdfStruct.metallicRoughnessTextureID = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            bsdfStructs.push_back(bsdfStruct);
        }
        else if ( /* TODO: Refraction */ false) {

        }
        else {
            //bsdf = new DiffuseBSDF(glm::vec3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1], material.pbrMetallicRoughness.baseColorFactor[2]));
            bsdfStruct.bsdfType = DIFFUSE;
            bsdfStruct.reflectance = glm::vec3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1], material.pbrMetallicRoughness.baseColorFactor[2]);
            bsdfStruct.baseColorTextureID = material.pbrMetallicRoughness.baseColorTexture.index;
            bsdfStruct.metallicRoughnessTextureID = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            bsdfStructs.push_back(bsdfStruct);
        }
        //bsdfs.push_back(bsdf);

    }
}

void Scene::initTextures()
{
    for (const auto & texture: model.textures)
    {
        auto name = texture.name;
        auto image = model.images[texture.source];
        auto uri = image.uri;
        auto width = image.width;
        auto height = image.height;
        auto component = image.component;
        auto data = image.image.data();
        auto size = image.image.size();
        auto type = image.pixel_type;
        auto format = image.pixel_type;

        printf("Loading Texture size: %d data: %d\n", size, data);
        std::vector<unsigned char> textureData;
        // TODO: Resize texture.data to unsigned char based on texture.type
        // To save bandwith, we only support unsigned char for now
        switch (type)
        {
        case GLTFDataType::GLTF_DATA_TYPE_UNSIGNED_BYTE:
            textureData = image.image;
            break;
        case GLTFDataType::GLTF_DATA_TYPE_UNSIGNED_SHORT:
            for (size_t i = 0; i < image.image.size(); i++)
            {
                //printf("image.image: %d\n", image.image[i]);
                textureData.push_back(static_cast<unsigned char>(image.image[i]));
            }
            //assert(0);
            break;
        default:
            /* Unsupported data type! */
            assert(0);
            break;
        }
        TextureInfo texture;
        texture.width = width;
        texture.height = height;
        texture.data = textureData;
        texture.nrChannels = component;
        textures.push_back(texture);
    }
}

void Scene::initCamera()
{
    if (model.cameras.size() == 0) {
        printf("WARNING: NO CAMERA FOUND IN THE MODEL\n");
    }
    else if (model.cameras[0].type != "perspective"){
		printf("ERROR: Only perspective camera is supported for now\n");
		assert(0);
	}
    else {
        auto sceneCamera = model.cameras[0].perspective;
        auto aspectRatio = sceneCamera.aspectRatio;
        auto yfov = sceneCamera.yfov;
        auto f = 1.0f / tan(yfov * 0.5f);
        auto znear = sceneCamera.znear;
        auto zfar = sceneCamera.zfar;
        camera.resolution = glm::vec2(800, 800);
        
    }



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
    else if (node.camera >= 0) {
        const auto& sceneCamera = model.cameras[node.camera];
        processCamera(model, sceneCamera, nodeTransform);
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
        const auto& indicesAccessor = model.accessors[primitive.indices];
        const auto& positionsAccessor = model.accessors[primitive.attributes.at("POSITION")];
        const auto& normalsAccessor = model.accessors[primitive.attributes.at("NORMAL")];

        const auto& indicesView = model.bufferViews[indicesAccessor.bufferView];
        const auto& positionsView = model.bufferViews[positionsAccessor.bufferView];
        const auto& normalsView = model.bufferViews[normalsAccessor.bufferView];

        // TODO: Dynamic type array according to componentType
        const unsigned short* indexData = reinterpret_cast<const unsigned short*>(&model.buffers[indicesView.buffer].data[indicesAccessor.byteOffset + indicesView.byteOffset]);
        const float* positionData       = reinterpret_cast<const float*>(&model.buffers[positionsView.buffer].data[positionsAccessor.byteOffset + positionsView.byteOffset]);
        const float* normalData         = reinterpret_cast<const float*>(&model.buffers[normalsView.buffer].data[normalsAccessor.byteOffset + normalsView.byteOffset]);

        const size_t vertexStride = 3;
        const size_t normalStride = 3;
        const size_t uvStride = 2;

        const float* uvData;
        bool hasUV = (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end());
        if (hasUV) {
            const auto& uvAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
            const auto& uvView = model.bufferViews[uvAccessor.bufferView];
            uvData = reinterpret_cast<const float*>(&model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]);
        }

        const size_t numIndices = indicesAccessor.count;
        glm::mat4x4 normalTransform = glm::transpose(glm::inverse(transform));

        int materialID = primitive.material;
        if (primitive.material < 0) {
            printf("WARNING: No material specified for mesh %s\n", mesh.name.c_str());
            if (model.materials.size() == 0) {
				printf("ERROR: No material found in the model\n");
                assert(0);
			}
			materialID = 0;
		}
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

            triangle.n1 = glm::normalize(glm::vec3(normalTransform * glm::vec4(triangle.n1, 0.0f)));
            triangle.n2 = glm::normalize(glm::vec3(normalTransform * glm::vec4(triangle.n2, 0.0f)));
            triangle.n3 = glm::normalize(glm::vec3(normalTransform * glm::vec4(triangle.n3, 0.0f)));
            
            if (hasUV) {
                triangle.uv1 = glm::vec2(uvData[indexData[i] * uvStride], uvData[indexData[i] * uvStride + 1]);
                triangle.uv2 = glm::vec2(uvData[indexData[i + 1] * uvStride], uvData[indexData[i + 1] * uvStride + 1]);
                triangle.uv3 = glm::vec2(uvData[indexData[i + 2] * uvStride], uvData[indexData[i + 2] * uvStride + 1]);
            }

            //printf("triangle.uv1: %f %f\n", triangle.uv1.x, triangle.uv1.y);
            //printf("triangle.uv2: %f %f\n", triangle.uv2.x, triangle.uv2.y);
            //printf("triangle.uv3: %f %f\n", triangle.uv3.x, triangle.uv3.y);
            //auto index0 = indexData[i];
            //auto index1 = indexData[i + 1];
            //auto index2 = indexData[i + 2];
            triangle.materialID = materialID;
            triangle.normalTextureID = bsdfStructs[materialID].normalTextureID;

            triangles.push_back(triangle);
        }
    }
}

void Scene::processCamera(const tinygltf::Model& model, const tinygltf::Camera& sceneCamera, const glm::mat4x4& parentTransform)
{
    if (sceneCamera.type != "perspective") {
        printf("ERROR: Only perspective camera is supported for now\n");
        assert(0);
    }
    else {
        auto sceneCamera = model.cameras[0].perspective;
        auto aspectRatio = sceneCamera.aspectRatio;
        auto yfov = sceneCamera.yfov;
        auto f = 1.0f / tan(yfov * 0.5f);
        auto znear = sceneCamera.znear;
        auto zfar = sceneCamera.zfar;
        camera.resolution = glm::vec2(800, 800);
        camera.fov = glm::vec2(yfov * sceneCamera.aspectRatio, yfov) * 90.0f;
        camera.position = glm::vec3(parentTransform * glm::vec4(0, 0, 0, 1));
        camera.view = glm::normalize(glm::vec3(parentTransform * glm::vec4(0,0,1,0)));
        if (abs(glm::dot(camera.view, glm::vec3(0, 1, 0))) < EPSILON) {
            camera.up = glm::vec3(0, 0, 1);
        }
        else {
			camera.up = glm::vec3(0, 1, 0);
        }
        camera.right = glm::normalize(glm::cross(camera.view, camera.up));
        camera.up = glm::normalize(glm::cross(camera.view, camera.right));

        camera.pixelLength = 2.0f * glm::vec2(1.0f / camera.resolution.x, 1.0f / camera.resolution.y);
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
