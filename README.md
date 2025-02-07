CUDA Path Tracer
================
[pbrt]: https://pbrt.org/

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yian Chen
  * [LinkedIn](https://www.linkedin.com/in/yian-chen-33a31a1a8/), [personal website](https://sydianandrewchen.github.io/) etc.
* Tested on: Windows 10, AMD Ryzen 5800 HS with Radeon Graphics CPU @ 3.20GHz 16GB, NVIDIA GeForce RTX3060 Laptop 8GB

<div align="center">
    <img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/emissive_robot_car_2.png?raw=true" width="45%"/>
    <img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/result_bunny.png?raw=true" width="45%"/>
    <img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/bump_mapping_after.png?raw=true" width="45%"/>
    <img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/robots.png?raw=true" width="45%"/>
    <img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/indoor.png?raw=true" width="45%"/>
<video controls>
  <source src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/realtime.mp4?raw=true" type="video/mp4">
</video>

</div>



### Implemeted Feature 

- [x] Core(as required in Project 3)
    - [x] Diffuse & Specular
    - [x] Jittering (Antialiasing)
    - [x] First Bounce Cache
    - [x] Sort by material
- [x] Load gltf
- [x] BVH && SAH
- [x] Texture mapping & bump mapping
- [x] Environment Mapping
- [x] Microfacet BSDF
- [x] Emissive BSDF (with Emissive Texture)
- [x] Direct Lighting
- [x] Multiple Importance Sampling
- [x] Depth of Field
- [x] Tone mapping && Gamma Correction


### Core features (As required by project instruction)
- Diffuse & Specular

![Diffuse & Specular Demo](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/basic_pathtracer.png?raw=true)

- Jittering
<table>
    <tr>
        <th>Before jittering</th>
        <th>After jittering</th>
    </tr>
    <tr>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/without_jittering.png?raw=true" /></th>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/with_jittering.png?raw=true" /></th>
    </tr>
</table>

### `gltf` Load & A Better Workflow (?)

In this pathtracer, supported scene format is `gltf` for its high expressive capability of 3D scenes. Please view [this page](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) for more details about gltf. 

Eventually, during development, most scenes used for testing is directly exported from Blender. This enables a much higher flexibility for testing. 

- `scenes/pathtracer_robots_demo.glb` [Link](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/tree/main/scenes/pathtracer_robots_demo.glb)

![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/blender_ref.png?raw=true)
### BVH
On host, we can construct and traverse BVH recursively. While in this project, our code run on GPU. Though recent cuda update allows recursive function execution on device, we cannot take that risk as raytracer is very performance-oriented. Recursive execution will slow down the kernel function, as it may bring dynamic stack size. 

Thanks to [this paper](https://arxiv.org/pdf/1505.06022.pdf), a novel BVH constructing and traversing algorithm called MTBVH is adopted in this pathtracer. This method is **stack-free**.

This pathtracer only implements a simple version of MTBVH. Instead of constructing 6 BVHs and traversing one of them at runtime, only 1 BVH is constructed. *It implies that this pathtracer still has the potential of speeding up*.

- With BVH & Without BVH:
<table>
    <tr>
        <th>With BVH</th>
        <th>Without BVH</th>
    </tr>
    <tr>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/with_bvh_bunny.png?raw=true" /></th>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/without_bvh_bunny.png?raw=true" /></th>
    </tr>
</table>

As expected, speedup is huge up to 40 times. With a more complex scene, BVH should give a higher speedup.

### Texture Mapping & Bump Mapping

To enhance the details of mesh surfaces and gemometries, texture mapping is a must. *Here we have not implemented mipmap on GPU, though it should not be that difficult to do so*.

- `scenes/pathtracer_test_texture.glb` [Link](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/tree/main/scenes/pathtracer_test_texture.glb)

<table>
    <tr>
        <th>Before bump mapping</th>
        <th>After bump mapping</th>
    </tr>
    <tr>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/bump_mapping_before.png?raw=true"/></th>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/bump_mapping_after.png?raw=true"/></th>
    </tr>
</table>

### Microfact BSDF

To use various material, bsdfs that are more complicated than diffuse/specular are required. Here, we will first implement the classic microfacet BSDF to extend the capability of material in this pathtracer.

This pathtracer uses the Microfacet implementation basd on [pbrt].

Metallness = 1. Roughness 0 to 1 from left to right.

> Please note that the sphere used here is not an actual sphere but an icosphere. 

- `scenes/pathtracer_test_microfacet.glb` [Link](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/tree/main/scenes/pathtracer_test_microfacet.glb)

![Microfacet Demo](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/microfacet.png?raw=true)

With texture mapping implemented, we can use `metallicRoughness` texture now. Luckily, `gltf` has a good support over metallic workflow.

- `scenes/pathtracer_robot.glb` [Link](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/tree/main/scenes/pathtracer_robot.glb)

![Metallic Workflow Demo](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/metallic_workflow.png?raw=true)

### Direct Lighting & MIS

> To stress the speed up of convergence in MIS, Russian-Roulette is disabled in this part's rendering.

> The tiny dark stripe is visible in some rendering result. This is because by default we do not allow double-sided lighting in this pathtracer.

> By default, number of light sample is set to 3.

When sampling for the direction of next bounce, we have adopted importance sampling for bsdf for most of the time. It enhances the convergence speed for specular materials, as the sampling strategies greatly aligned with the expected radiance distribution on hemisphere. However, for diffuse/matte surfaces, this sampling strategies can be optimized, as the most affecting factors for the radiance distribution of these sort of materials is light instead of outgoing rays. Thus, sampling from light is also a valuable strategy to speedup convergence speed of raytracing rough surfaces.

In this demo scene, 3 metal plane are allocated with 4 cube lights. When we only sample bsdf, we can see that the expected radiance on the surface of metal plane converges. When we only sample light, we can see how the rougher part of the scene, the back white wall, has better converging speed. Hence, we are looking forward to a sampling strategy that combines the advantages of these two, which is multiple importance sampling.

- `scenes/pathtracer_mis_demo.glb` [Link](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/tree/main/scenes/pathtracer_mis_demo.glb)

<table>
    <tr>
        <th>Only sample bsdf 500spp</th>
        <th>Only sample light 500spp</th>
        <th>MIS 500spp</th>
    </tr>
    <tr>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/mis_bsdf_500spp.png?raw=true"/></th>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/mis_light_500spp.png?raw=true"/></th>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/mis_mis_500spp.png?raw=true"/></th>
    </tr>
</table>


To see more details about this part, see [this part of pbrt](https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Importance_Sampling) or [this post](https://sydianandrewchen.github.io/blog/PBRT-Direct-Lighting) of mine.


Test on bunny scene. Faster convergence speed can be observed.

- `scenes/pathtracer_bunny_mis.glb` [Link](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/tree/main/scenes/pathtracer_bunny_mis.glb)

<table>
    <tr>
        <th>Without MIS 256spp</th>
        <th>With MIS 256spp</th>
    </tr>
    <tr>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/mis_without_mis_bunny_256spp.png?raw=true"/></th>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/mis_with_mis_bunny_256spp.png?raw=true"/></th>
    </tr>
    <tr>
        <th>Without MIS 5k spp</th>
        <th>With MIS 5k spp</th>
    </tr>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/mis_without_mis_bunny_5000spp.png?raw=true"/></th>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/mis_with_mis_bunny_5000spp.png?raw=true"/></th>
    </tr>
</table>




### Depth of Field

In depth of field, we define two variables. `focal_length` & `aperture`. 

More details can be viewed in [this post](https://pathtracing.home.blog/depth-of-field/).

![](https://pathtracinghome.files.wordpress.com/2018/12/dof1.png?w=1089)

<table>
    <tr>
        <th>Depth of Field (Aperture=0.3)</th>
    </tr>
    <tr>
        <th><img src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/depth_of_field.png?raw=true" class="responsive"/></th>
    </tr>
</table>

<video controls>
  <source src="https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/realtime_depth_of_field.mp4?raw=true" type="video/mp4">
</video>


### How to Run

Due to time limit, this code is not refactored to be production-ready. Many features are controlled by macros or even hardcoded, which is indeed irrational. More refactoring job might be accomplished in future.

To run this program for now, users need to adjust code.
Most changes should be done within [`pathtrace.cu`](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/src/pathtrace.cu).

```cpp
...
// Default settings
#define USE_FIRST_BOUNCE_CACHE 0
#define USE_SORT_BY_MATERIAL 0
#define MIS 1 // Multiple Importance Sampling
#define RUSSIAN_ROULETTE 0
#if RUSSIAN_ROULETTE
	#define RR_THRESHOLD 0.7f
#endif
#define USE_ENV_MAP 1
#define USE_BVH 1
#define TONE_MAPPING 0
...
// Change loaded scene here
static Scene * hst_scene = new Scene("..\\scenes\\pathtracer_robots_demo.glb");
...

```

Most settings should be self-explainable.

Another part where users can change settings is in [`bvh.h`](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/src/bvh.h)

```cpp
...
#define BVH_NAIVE 1
#define BVH_SAH 0
...
```

Here, we can determine whether BVH will use a naive split or a surface area heuristic function. 

### Other Result
- [Run Video](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/realtime.mp4)
- [Depth of Field Video](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/realtime_depth_of_field.mp4)



### Future (If possible)

#### Cuda Side
- [ ] More cuda optimization
    - [ ] Bank conflict
    - [ ] Loop unroll
        - [ ] Light sample loop (if multiple light rays) 
    - [ ] Higher parallelism (Use streams?)
- [ ] Tile-based raytracing
    - [ ] Potentially, it should increase the rendering speed, as it will maximize locallity within one pixel/tile. No more realtime camera movement though.

#### Render Side
- [ ] Adaptive Sampling
- [ ] Mipmap
- [ ] ReSTIR
- [ ] Refractive
- [ ] True B**S**DF (Add some subsurface scattering if possible?)
- [ ] Volume Rendering ~~(Ready for NeRF)~~

---

**Below are the development history & some bloppers of this program. If you are not intersted, there is no need to keep reading.**



### History

- [x] Load mesh within arbitrary scene
    - [x] Triangle
    - [x] Integrate `tinygltf`
    - [x] Scene Node Tree

- [ ] Core
    - [x] G Buffer
    - [x] Russian Roulette
    - [x] Sort by material

- [ ] More BSDF
    - [x] Diffuse
    - [x] Emissive
    - [x] Microfacet
    - [ ] Reflective
    - [ ] Refractive
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
    - [x] Monte carlo sampling
    - [x] Importance sampling
    - [x] Direct Lighting
    - [x] Multiple Importance Sampling

- [ ] Camera
    - [x] Jitter
    - [x] Field of depth
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
    ![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/accessor_data_types.png?raw=true)
    - Done with loading a scene with node tree!
            ![blender_reference](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/blender_reference.png?raw=true)
            ![rendered](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/first_scene.png?raw=true)
        > Can't tell how excited I am! Now my raytracer is open to most of the scenes!
        - Scene with parenting relationship
            ![with_parenting](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/scene_with_parenting.png?raw=true)

09.23-09.26
> Waste too much time on OOP. Eventually used C-style coding.

09.26
Finally, finish gltf loading and basic bsdf.

- A brief trial
    - Note that this difference might be due to different bsdf we are using right now. For convenience, we are using the most naive Diffuse BSDF, while Blender use a standard BSDF by default.
![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/first_trial_glb_scene.png?raw=true)


09.27

Naive BVH (probably done...)
Scene with 1k faces
- One bounce, normal shading
    - Without BVH: FPS 10.9
    ![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/normal_shading_without_bvh.png?raw=true)

    - With BVH: FPS 53.4
    ![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/normal_shading_with_bvh.png?raw=true)
    - 5 times faster
- Multiple bounces
    - Without BVH: FPS 7.6
        ![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/multiple_bounces_without_bvh.png?raw=true)
    - With BVH: FPS 22.8
        ![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/multiple_bounces_with_bvh.png?raw=true)


09.28

- SAH BVH(probably done...)

- Texture sampling
    - Try billboarding
    ![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/billboarding.png?raw=true)

09.29

- Texture mapping
    - Texture mapping test(only baseColor shading)
    ![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/texture_mapping_test.png?raw=true)
    - Bump mapping
        - Normals in world coordinate
    ![](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/bump_mapping_world_coord.png?raw=true)
        - Before bump mapping
    ![](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/bump_mapping_before_annotated.png?raw=true)
        - After bump mapping
    ![](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/bump_mapping_after_annotated.png?raw=true)
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

10.1-10.2
Try to refactor camera
- Failed. gltf seems to have a really ambiguous definition of camera.

10.3
- [ ] Denoising
    - [x] OpenImage Denoiser built
        - CPU only for now
        - [ ] Figure out how to build `oidn` for cuda
    - [ ] Integrate it into project


10.4-10.6

- Microfacet

10.7 

- Environment map

10.8

- Fix random number issue(Maybe try to generate a better random number array in future?)
    - Before
![Alt text](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/before_fixing_random_number_issue.png?raw=true)
    - After
![](https://github.com/SydianAndrewChen/Project3-CUDA-Path-Tracer/blob/main/img/after_fixing_random_number_issue.png?raw=true)

    **Please notice the fracture on rabbit head before fixing**

10.9
- MIS (Finally!)

- Russian Roulette 
    - Pro: Speed up by 60%
    - Con: Lower the converge speed


- Depth of field
    - Add a realtime slider to adjust