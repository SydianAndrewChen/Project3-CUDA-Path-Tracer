#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    float min_t;
    float max_t;
    __host__ __device__ glm::vec3 at(float t) const{
        return origin + direction * t;
    }
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct BoundingBox {
    __host__ __device__ BoundingBox(){}
    __host__ __device__ BoundingBox(glm::vec3 bbmin, glm::vec3 bbmax) : bbmin(bbmin), bbmax(bbmax){}
    __host__ __device__ BoundingBox(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) : bbmin(xmin, ymin, zmin), bbmax(xmax, ymax, zmax){}
    glm::vec3 bbmin, bbmax;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};


struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    glm::vec3 constantTerm;
};

class Primitive;

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  const Primitive* primitive;
};



class Primitive {
public:
    int materialID;
    __device__ virtual const BoundingBox& getBoundingBox() const = 0;
    __host__ __device__ virtual void test() const {
        printf("Primitive::test\n");
    };

    __device__ virtual bool hasIntersection(const Ray& r) const {
        return false;
    }

    __device__ virtual bool intersect(Ray& r, ShadeableIntersection* isect) const {
        printf("Primitive::intersect\n");
        return false;
    };
    //__device__ virtual void applyTransformation(const glm::mat3x3 modelMatrix);
};

class Sphere : public Primitive {
    glm::vec3 c;
    float r;
    BoundingBox bb;
public:
    __host__ __device__ Sphere(const glm::vec3& c, const float r) : c(c), r(r), bb(c - r, c + r) {}
    __device__ const BoundingBox& getBoundingBox() const override {
        return bb;
    }

    // TODO: fix intellisense bug? intellisense will tell me that I am wrong when adding override
    __device__ bool hasIntersection(const Ray& r, ShadeableIntersection& intersection) const {
        // TODO

        return true;
    }
};

class Cube : public Primitive {
    glm::vec3 cubeMin;
    glm::vec3 cubeMax;
    BoundingBox bb;
public:
    __host__ __device__ Cube(const glm::vec3& cubeMin, const glm::vec3& cubeMax) :cubeMin(cubeMin), cubeMax(cubeMax), bb(glm::min(cubeMin, cubeMax), glm::max(cubeMin, cubeMax)) {}
    __device__ const BoundingBox& getBoundingBox() const override { return bb; }
};

//class Triangle : public Primitive {
class Triangle{
public:
    glm::vec3 p1;
    glm::vec3 p2;
    glm::vec3 p3;
    glm::vec3 n1;
    glm::vec3 n2;
    glm::vec3 n3;
    glm::vec2 uv1;
    glm::vec2 uv2;
    glm::vec2 uv3;
    BoundingBox bb;
    glm::vec3 N;
    int materialID;
    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) : 
        p1(p1), p2(p2), p3(p3) {
        N = cross(p2 - p1, p3 - p1);
        bb = BoundingBox(glm::min(p1, glm::min(p2, p3)), glm::max(p1, glm::max(p2, p3)));
    }
    __host__ __device__ Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3) :
        p1(p1), p2(p2), p3(p3), n1(n1), n2(n2), n3(n3) {
        N = cross(p2 - p1, p3 - p1);
        bb = BoundingBox(glm::min(p1, glm::min(p2, p3)), glm::max(p1, glm::max(p2, p3)));
    }
    __host__ __device__ Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec2 uv1, glm::vec2 uv2, glm::vec2 uv3): 
        p1(p1), p2(p2), p3(p3), uv1(uv1), uv2(uv2), uv3(uv3) {
        N = cross(p2 - p1, p3 - p1);
        bb = BoundingBox(glm::min(p1, glm::min(p2, p3)), glm::max(p1, glm::max(p2, p3)));
    }

    __host__ __device__ virtual void test() const {
        //printf("Triangle::test\n");
        //printf("Pos: %f %f %f\n", p1[0], p1[1], p1[2]);
    };

    __device__ virtual bool has_intersection(const Ray& r) const {
        //assert("Not implemented!");
        return true;
    }


    __device__ bool Triangle::intersect(Ray& r, ShadeableIntersection* isect) const {
        //printf("Triangle::intersect\n");
        glm::vec3 bary;
        bool hasHit = glm::intersectRayTriangle(r.origin, r.direction, p1, p2, p3, bary);
        if (hasHit) {
            glm::vec3 intersection = bary.x * p1 + bary.y * p2 + bary.z * p3;
            float t = (intersection - r.origin).length();
            if (r.min_t < t && t < r.max_t) {
                r.max_t = t;
                isect->t = t;
                //isect->primitive = this;
                isect->surfaceNormal = bary.x * n1 + bary.y * n2 + bary.z * n3;
            }
        }
        return hasHit;
    }

    __device__ const BoundingBox& getBoundingBox() const { return bb; }
};
