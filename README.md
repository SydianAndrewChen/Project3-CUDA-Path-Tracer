CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

- [x] Load arbitrary scene(only geom)
    - [x] Triangle
    - [x] ~~Primitive assemble phase~~(This will not work, see `README` of this commit)
    - [x] Use tinygltf
        Remember to check [data type]((https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types)) before using accessor
    ![Alt text](img/accessor_data_types.png)
    - [x] Done with loading a scene with node tree!
            ![blender_reference](img/blender_reference.png)
            ![rendered](img/first_scene.png)
        > Can't tell how excited I am! Now my raytracer is open to most of the scenes!
        - Scene with parenting relationship
            ![with_parenting](img/scene_with_parenting.png)
- [ ] More BSDF
    - [x] Diffuse
    - [x] Emissive
    - [ ] Reflective
    - [ ] Refractive
    - [ ] Microfacet

- [ ] BVH
    - [ ] BoundingBox Array
    - [ ] Construct BVH
    - [ ] Traverse BVH
    - [ ] Better Heuristics

- [ ] Light (probably not gonna do a lot about it because gltf has a poor support over area light)


### Log

09.26
Finally, finish gltf loading and basic bsdf.

- A brief trial
    - Note that this difference might be due to different bsdf we are using right now. For convenience, we are using the most naive Diffuse BSDF, while Blender use a standard BSDF by default.
![Alt text](img/first_trial_glb_scene.png)