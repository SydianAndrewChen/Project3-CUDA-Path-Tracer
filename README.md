CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

- [x] Load mesh within arbitrary scene
    - [x] Triangle
    - [x] Integrate `tinygltf`
    - [x] Scene Node Tree

- [ ] Core
    - [x] G Buffer
    - [x] Sort by material
    - [ ] Russian Roulette

- [ ] More BSDF
    - [x] Diffuse
    - [x] Emissive
    - [ ] Reflective
    - [ ] Refractive
    - [ ] Microfacet
    - [ ] Disney

- [ ] BVH
    - [x] Basic BVH
        - [x] BoundingBox Array
        - [x] Construct BVH
        - [x] Traverse BVH
    - [x] Better Heuristics
        - [x] SAH 
    - [ ] MTBVH

- [ ] Texture
    - [x] Naive texture sampling
        - A Resource Mananger to help get the handle to texture?
    - [x] Bump mapping
    - [ ] Displacement mapping
    - [ ] Deal with antialiasing

- [ ] Better sampler
    - [ ] Encapsulate a sampler class
        - Gotta deal with cmake issue
    - [ ] Monte carlo sampling
    - [ ] Importance sampling


- [ ] Light 
    - ~~Probably not gonna do a lot about it because gltf has a poor support over area light~~ 
    - Might need to implement it for direct lighting to speed up the converging speed of ray tracing

- [ ] Camera
    - [x] Jitter
    - [ ] Field of depth
    - [ ] Motion blur 


- [ ] Denoiser 
    - [ ] Use Intel OpenImage Denoiser for now


### Log
09.20
- Basic raytracer
- Refactor integrator
- First triangle!

09.21-22
- Load arbitrary scene(only geom)
    - Triangle
    - ~~Primitive assemble phase~~(This will not work, see `README` of this commit)
    - Use tinygltf
        Remember to check [data type]((https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types)) before using accessor
    ![Alt text](img/accessor_data_types.png)
    - Done with loading a scene with node tree!
            ![blender_reference](img/blender_reference.png)
            ![rendered](img/first_scene.png)
        > Can't tell how excited I am! Now my raytracer is open to most of the scenes!
        - Scene with parenting relationship
            ![with_parenting](img/scene_with_parenting.png)

09.23-09.26
> Waste too much time on OOP. Eventually used C-style coding.

09.26
Finally, finish gltf loading and basic bsdf.

- A brief trial
    - Note that this difference might be due to different bsdf we are using right now. For convenience, we are using the most naive Diffuse BSDF, while Blender use a standard BSDF by default.
![Alt text](img/first_trial_glb_scene.png)


09.27

Naive BVH (probably done...)
Scene with 1k faces
- One bounce, normal shading
    - Without BVH: FPS 10.9
    ![Alt text](img/normal_shading_without_bvh.png)

    - With BVH: FPS 53.4
    ![Alt text](img/normal_shading_with_bvh.png)
    - 5 times faster
- Multiple bounces
    - Without BVH: FPS 7.6
        ![Alt text](img/multiple_bounces_without_bvh.png)
    - With BVH: FPS 22.8
        ![Alt text](img/multiple_bounces_with_bvh.png)


09.28

- SAH BVH(probably done...)

- Texture sampling
    - Try billboarding
    ![Alt text](img/billboarding.png)

09.29

- Texture mapping
    - Texture mapping test(only baseColor shading)
    ![Alt text](img/texture_mapping_test.png)
    - Bump mapping
        - Normals in world coordinate
    ![](img/bump_mapping_world_coord.png)
        - Before bump mapping
    ![](img/bump_mapping_before_annotated.png)
        - After bump mapping
    ![](img/bump_mapping_after_annotated.png)
    It might be a little difficult to notice the difference before and after bump mapping. **Please observe the logo on the box, more details are added.**


    - [ ] Texture aliasing is indeed quite serious!
        - [ ] However, to implement antialiasing for texture mapping, I may need to consider implementing mipmapping. 

- [ ] Microfacet model pbr failed...
    - [ ] Need to read through microfacet part and think about how to use roughness and metallic indeed

09.30

- [ ] Microfacet
    - [x] Metal Fresnel hack
    - [ ] Conductor
        - [ ] After mixing, need to consider how to sample

- [x] Camera 
    - [x] Antialiasing

- [ ] Denoising
    - [x] OpenImage Denoiser built
        - CPU only for now
        - [ ] Figure out how to build `oidn` for cuda
    - [ ] Integrate it into project

- [x] Sort by shader
    - No speedup observed?

10.1

- [ ] Reflective
- [ ] Refractive

- [ ] Camera refactor
    - [ ] Abstract the camera class
    - [ ] Export camera from gltf

- [ ] Field of depth


- [ ] Light refactor
    - [ ] Implement a light class
        - [ ] Consider how to make it compatible with emissive bsdfs

- [ ] Material refactor
    - [ ] May need to refactor material as a combination of different bsdfs
        - according to pbrt


- [ ] Primitve refactor
    - [ ] Support more primitive besdies triangles(not so urgent)