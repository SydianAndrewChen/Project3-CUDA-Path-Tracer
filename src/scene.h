#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "light.h"
#include <thrust/device_vector.h>


using namespace std;

class HostScene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();

    void initLightFromObject();
public:
    HostScene(string filename);
    ~HostScene() {};

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<std::shared_ptr<Light>> lights;
    RenderState state;
    int geoms_size;
};

//template<typename T>
//void FreePtrVector(thrust::device_vector<T> v, int v_size) {
//    for (size_t i = 0; i < v_size; i++)
//    {
//        cudaFree(v[i]);
//    }
//}

class DeprecatedScene {
public:
    thrust::device_vector<Geom*> geoms;
    thrust::device_vector<Material*> mats;
    Light ** lights;
    int light_size = 0;
    std::vector<glm::vec3> triangles;
    DeprecatedScene(const char* filename);
    ~DeprecatedScene() {
        //FreePtrVector<<<1,1>>>(geoms);
        //FreePtrVector<<<1,1>>>(mats);
        //FreePtrVector(lights, lights.size());
        for (size_t i = 0; i < light_size; i++)
        {
            cudaFree(lights[i]);
        }
        cudaFree(lights);
    }

};


