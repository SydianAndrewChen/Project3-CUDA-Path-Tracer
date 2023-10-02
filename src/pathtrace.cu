#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "depScene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "light.h"
#include "scene.h"
#include "bsdf.h"
#include "bvh.h"
#include "texture.h"
#include "camera.h"

//#include "utilities.cuh"

#define USE_FIRST_BOUNCE_CACHE 1
#define USE_SORT_BY_MATERIAL 0
#define ONE_BOUNCE_DIRECT_LIGHTINIG 0
#define USE_BVH 1

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		//color.x = glm::clamp((int)pix.x, 0, 255);
		//color.y = glm::clamp((int)pix.y, 0, 255);
		//color.z = glm::clamp((int)pix.z, 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static HostScene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Triangle* dev_triangles= nullptr;
//static Scene* pa = new Scene("..\\scenes\\pathtracer_dragon.glb");
static Scene * pa;
static BSDFStruct * dev_bsdfStructs = nullptr;
static BVHAccel * bvh = nullptr;
static BVHNode* dev_bvhNodes = nullptr;
static ShadeableIntersection * dev_first_iteration_first_bounce_cache_intersections = nullptr;
static Texture * dev_textures = nullptr;
static TextureInfo * dev_textureInfos = nullptr;
int textureSize = 0;

#if USE_SORT_BY_MATERIAL
thrust::device_ptr<int> dev_bsdfStructIDs = nullptr;
thrust::device_ptr<int> dev_bsdfStructIDs_copy = nullptr;
#endif
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

__global__ void initDeviceTextures(Texture & dev_texture, const TextureInfo & dev_textureInfo, unsigned char * data) {
	dev_texture.width = dev_textureInfo.width;
	dev_texture.height = dev_textureInfo.height;
	dev_texture.nrChannels = dev_textureInfo.nrChannels;
	printf("dev_texture.width: %d, dev_texture.height: %d, dev_texture.nrChannels: %d\n", dev_texture.width, dev_texture.height, dev_texture.nrChannels);
	dev_texture.data = data;
	printf("dev_texture.data[0]: %d\n", dev_texture.data[0]);

	auto test_sample = sampleTextureRGBA(dev_texture, glm::vec2(1.0f, 1.0f));
	printf("test_sample: %f, %f, %f, %f\n", test_sample.x, test_sample.y, test_sample.z, test_sample.w);
}

__global__ void initPrimitivesNormalTexture(Triangle* triangles, Texture* textures, int triangle_size) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < triangle_size){
		triangles[index].normalTexture = textures + triangles[index].normalTextureID;
	}
}

__global__ void initBSDFWithTextures(BSDFStruct* bsdfStructs, Texture* texture, int bsdfStructs_size) {
	for (size_t i = 0; i < bsdfStructs_size; i++)
	{
		if (bsdfStructs[i].baseColorTextureID != -1)
			bsdfStructs[i].baseColorTexture	= &(texture[bsdfStructs[i].baseColorTextureID]);
		if (bsdfStructs[i].metallicRoughnessTextureID != -1)
			bsdfStructs[i].metallicRoughnessTexture = &(texture[bsdfStructs[i].metallicRoughnessTextureID]);
		if (bsdfStructs[i].normalTextureID != -1)
			bsdfStructs[i].normalTexture = &(texture[bsdfStructs[i].normalTextureID]);
		printf("bsdfStructs[i].diffuseTextureID: %d\n", bsdfStructs[i].baseColorTextureID);
		printf("bsdfStructs[i].roughnessTextureID: %d\n", bsdfStructs[i].metallicRoughnessTextureID);

	}
}

void pathtraceInitBeforeMainLoop(Scene * pa) {

	// TODO: initialize any extra device memeory you need

	textureSize = pa->textures.size();
	auto textureInfos = pa->textures.data();
	cudaMalloc(&dev_textureInfos, textureSize * sizeof(TextureInfo));
	cudaMemcpy(dev_textureInfos, textureInfos, textureSize * sizeof(TextureInfo), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_textures, textureSize * sizeof(Texture));
	unsigned char* dev_texture_data = nullptr;
	for (int i = 0; i < textureSize; i++)
	{
		cudaMalloc(&dev_texture_data, textureInfos[i].width * textureInfos[i].height * textureInfos[i].nrChannels * sizeof(unsigned char));
		cudaMemcpy(dev_texture_data, textureInfos[i].data.data(), textureInfos[i].width * textureInfos[i].height * textureInfos[i].nrChannels * sizeof(unsigned char), cudaMemcpyHostToDevice);
		initDeviceTextures << <1, 1 >> > (dev_textures[i], dev_textureInfos[i], dev_texture_data);
	}


	checkCUDAError("initDeviceTextures");

	auto bsdfStructs = pa->bsdfStructs.data();
	cudaMalloc(&dev_bsdfStructs, pa->bsdfStructs.size() * sizeof(BSDFStruct));
	cudaMemcpy(dev_bsdfStructs, bsdfStructs, pa->bsdfStructs.size() * sizeof(BSDFStruct), cudaMemcpyHostToDevice);

	initBSDFWithTextures << <1, 1 >> > (dev_bsdfStructs, dev_textures, pa->bsdfStructs.size());
	checkCUDAError("initBSDFWithTextures");


	bvh = new BVHAccel();
	bvh->initBVH(pa->triangles);
	auto triangles = bvh->orderedPrims.data();
	cudaMalloc(&dev_triangles, bvh->orderedPrims.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, triangles, bvh->orderedPrims.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvhNodes, bvh->nodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvhNodes, bvh->nodes.data(), bvh->nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	int triangle_size = bvh->orderedPrims.size();
	int blockSize = 256;
	dim3 initNormalTextureBlock((triangle_size + blockSize - 1) / blockSize);

	initPrimitivesNormalTexture << <initNormalTextureBlock, blockSize >> > (dev_triangles, dev_textures, triangle_size);

}

void pathtraceInit(HostScene* scene, Scene * _pa) {
	pa = _pa;
	hst_scene = scene;

	const DepCamera& cam = hst_scene->state.depCamera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_first_iteration_first_bounce_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_iteration_first_bounce_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if USE_SORT_BY_MATERIAL
	int* dev_bsdfStructIDs_ptr;
	int * dev_bsdfStructIDs_copy_ptr;
	cudaMalloc(&(dev_bsdfStructIDs_ptr), pixelcount * sizeof(int));
	cudaMemset(dev_bsdfStructIDs_ptr, 0, pixelcount * sizeof(int));
	cudaMalloc(&dev_bsdfStructIDs_copy_ptr, pixelcount * sizeof(int));
	cudaMemset(dev_bsdfStructIDs_copy_ptr, 0, pixelcount * sizeof(int));
	dev_bsdfStructIDs = thrust::device_ptr<int>(dev_bsdfStructIDs_ptr);
	dev_bsdfStructIDs_copy = thrust::device_ptr<int>(dev_bsdfStructIDs_copy_ptr);
#endif // USE_SORT_BY_MATERIAL
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	//cudaFree(dev_triangles);
	//cudaFree(dev_bsdfStructs);
	//delete bvh;
	cudaFree(dev_first_iteration_first_bounce_cache_intersections);
#if USE_SORT_BY_MATERIAL
	cudaFree(dev_bsdfStructIDs.get());
	cudaFree(dev_bsdfStructIDs_copy.get());
#endif // SORT_BY_MATERIAL
	checkCUDAError("pathtraceFree");
}

void pathtraceFreeAfterMainLoop(Scene * pa) {
	cudaFree(dev_triangles);
	cudaFree(dev_bsdfStructs);
	cudaFree(dev_textureInfos);

	/* TODO: Also need to free the texture that dev_texture point to*/
	cudaFree(dev_textures);
	delete bvh;
	delete pa;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(DepCamera cam, Camera camera, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < camera.resolution.x && y < camera.resolution.y) {
		int index = x + (y * camera.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = camera.position;
		segment.color = glm::vec3(0.0f, 0.0f, 0.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::random::uniform_real_distribution<float> u01(0, 1);
		glm::vec2 jitter(u01(rng) - 0.5f, u01(rng) - 0.5f);
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(camera.view
			- camera.right * camera.pixelLength.x * ((float)x + jitter[0] - (float)camera.resolution.x * 0.5f)
			- camera.up * camera.pixelLength.y * ((float)y + jitter[1] - (float)camera.resolution.y * 0.5f)
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.constantTerm = glm::vec3(1.0f, 1.0f, 1.0f);
	}

	//if (x < cam.resolution.x && y < cam.resolution.y) {
	//	int index = x + (y * cam.resolution.x);
	//	PathSegment& segment = pathSegments[index];

	//	segment.ray.origin = cam.position;
	//	segment.color = glm::vec3(0.0f, 0.0f, 0.0f);

	//	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	//	thrust::random::uniform_real_distribution<float> u01(0,1);
	//	glm::vec2 jitter(u01(rng)-0.5f, u01(rng)-0.5f);
	//	// TODO: implement antialiasing by jittering the ray
	//	segment.ray.direction = glm::normalize(cam.view
	//		- cam.right * cam.pixelLength.x * ((float)x + jitter[0] - (float)cam.resolution.x * 0.5f)
	//		- cam.up * cam.pixelLength.y * ((float)y + jitter[1] - (float)cam.resolution.y * 0.5f)
	//	);

	//	segment.pixelIndex = index;
	//	segment.remainingBounces = traceDepth;
	//	segment.constantTerm = glm::vec3(1.0f, 1.0f, 1.0f);
	//}
}

//TODO: Change to BVH in future!
__device__ bool intersectCore(BVHNode * nodes, int bvhNodes_size, Triangle * triangles, int triangles_size, Ray & ray, ShadeableIntersection& intersection) {
#if USE_BVH
	intersectBVH(nodes, bvhNodes_size, triangles, ray, intersection);
#else
	for (int i = 0; i < triangles_size; i++)
	{
		intersectTriangle(triangles[i], ray, &intersection);
	}
#endif
	return intersection.t > EPSILON;
}


__global__ void intersect(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Triangle* triangles
	, int triangles_size
	, ShadeableIntersection* intersections
	, BVHNode * bvhNodes
	, int bvhNodes_size
#if USE_SORT_BY_MATERIAL
	, thrust::device_ptr<int> bsdfStructIDs
	, thrust::device_ptr<int> bsdfStructIDs_copy
#endif
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		Ray & ray = pathSegments[path_index].ray;
		ray.min_t = 0.0;
		ray.max_t = FLT_MAX;
		intersections[path_index].t = -1;
		//intersectBVH(bvhNodes, bvhNodes_size, triangles, ray, intersections[path_index]);
		intersectCore(bvhNodes, bvhNodes_size, triangles, triangles_size, pathSegments[path_index].ray, intersections[path_index]);
#if USE_SORT_BY_MATERIAL
		bsdfStructIDs[path_index] = intersections[path_index].materialId;
		bsdfStructIDs_copy[path_index] = intersections[path_index].materialId;
#endif // SORT_BY_MATERIAL

	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeBSDF(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, BSDFStruct * bsdfStructs
	, Triangle * triangles
	, int triangles_size
	, BVHNode * bvhNodes
	, int bvhNodes_size
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& pathSegment = pathSegments[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			BSDFStruct & bsdfStruct = bsdfStructs[intersection.materialId];
			//pathSegment.color = glm::vec3(intersection.surfaceNormal);
			//pathSegment.color = glm::vec3(intersection.uv.x, intersection.uv.y, 0.0f);
			////printf("intersection.uv: %f, %f\n", intersection.uv.x, intersection.uv.y);
			//return;
			// If the material indicates that the object was a light, "light" the ray
			if (bsdfStruct.bsdfType == BSDFType::EMISSIVE) {
				pathSegments[idx].remainingBounces = 0;
				pathSegment.color += (bsdfStruct.emissiveFactor * bsdfStruct.strength * pathSegment.constantTerm);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				glm::mat3x3 o2w;
				make_coord_space(o2w, intersection.surfaceNormal);
				glm::mat3x3 w2o(glm::transpose(o2w));
				if (bsdfStructs[intersection.materialId].normalTextureID != -1) {
					glm::vec3 sampledNormal = sampleTextureRGB(*(bsdfStructs[intersection.materialId].normalTexture), intersection.uv) * 2.0f - 1.0f;
					intersection.surfaceNormal = o2w * sampledNormal;
					make_coord_space(o2w, intersection.surfaceNormal);
					w2o = glm::transpose(o2w);
				}
				//return;
				glm::vec3 intersect = pathSegment.ray.at(intersection.t);
				glm::vec3 L_direct;
				float pdf;
				glm::vec3 wo = w2o * -pathSegment.ray.direction;
				glm::vec3 wi;
#if  ONE_BOUNCE_DIRECT_LIGHTINIG
				/* Estimate one bounce direct lighting */
				int n_sample = 10;
				int n_valid_sample = 0;
				for (size_t i = 0; i < n_sample; i++)
				{
					sample_f(bsdfStruct, wo, wi, &pdf, rng);
					float cosineTerm = abs(wi.z);
					Ray one_bounce_ray;
					one_bounce_ray.direction = o2w * wi;
					one_bounce_ray.origin = intersect;
					one_bounce_ray.min_t = EPSILON;
					one_bounce_ray.max_t = FLT_MAX;
					ShadeableIntersection oneBounceIntersection;
					oneBounceIntersection.t = -1.0f;

					if (intersectCore(bvhNodes, bvhNodes_size, triangles, triangles_size, one_bounce_ray, oneBounceIntersection)){
						auto oneBounceBSDF = bsdfStructs[oneBounceIntersection.materialId];
						if (oneBounceBSDF.bsdfType == BSDFType::EMISSIVE) {
							n_valid_sample++;
							auto Li = oneBounceBSDF.emissiveFactor * oneBounceBSDF.strength;

							// TODO: Figure out why is it still not so similar to the reference rendered image?
							//		May be checkout the light propogation chapter of pbrt to find out how we use direct light estimation?
							L_direct += Li * bsdf * cosineTerm;
						}
					}
				}

				float one_over_n = 1.0f / (n_valid_sample + 1);
				pathSegment.color += (L_direct * pathSegment.constantTerm * one_over_n);
#endif //  ONE_BOUNCE_DIRECT_LIGHTINIG
				glm::vec3 bsdf = sample_f(bsdfStruct, wo, wi, &pdf, rng, intersection.uv);
				float cosineTerm = abs(wi.z);
#if ONE_BOUNCE_DIRECT_LIGHTINIG
				pathSegment.constantTerm *= (bsdf * cosineTerm * one_over_n / pdf);
#else
				pathSegment.constantTerm *= (bsdf * cosineTerm / pdf);
#endif			
				pathSegment.ray.origin = intersect;
				pathSegment.ray.direction = o2w * wi;
				pathSegment.remainingBounces--;
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void sampleScreen(glm::vec3 * dev_image, Texture & texture, int imageWidth, int imageHeight) {
	for (size_t x = 0; x < imageWidth; x++)
	{
		for (size_t y = 0; y < imageHeight; y++)
		{
			auto sampleResult = sampleTextureRGBA(texture, glm::vec2(float(x) / imageWidth, float(y) / imageHeight));
			dev_image[x + y * imageWidth] = glm::vec3(sampleResult.x, sampleResult.y, sampleResult.z);
		}
	}
}

struct HasHit{
    __host__ __device__ bool operator()(const PathSegment & path) const {
        return path.remainingBounces != 0;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	checkCUDAError("before generate camera ray");

	const int traceDepth = hst_scene->state.traceDepth;
	const DepCamera& cam = hst_scene->state.depCamera;
	Camera& camera = pa->camera;
	camera.fov = cam.fov;
	camera.pixelLength = cam.pixelLength;
	//camera.position = cam.position;
	//camera.view = cam.view;
	//camera.up = cam.up;
	//camera.right = cam.right;
	camera.resolution = cam.resolution;

	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, camera, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {


		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if USE_FIRST_BOUNCE_CACHE
		if (iter == 1) {
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


			intersect << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_triangles
				, pa->triangles.size()
				, dev_intersections
				, dev_bvhNodes
				, bvh->nodes.size()
#if USE_SORT_BY_MATERIAL
				, dev_bsdfStructIDs
				, dev_bsdfStructIDs_copy
#endif
				);
			if (depth == 0) {
				cudaMemcpy(dev_first_iteration_first_bounce_cache_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			//hasFirstBounceCache = true;
		}
		else {
			if (depth == 0) {
				cudaMemcpy(dev_intersections, dev_first_iteration_first_bounce_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				intersect << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_triangles
					, pa->triangles.size()
					, dev_intersections
					, dev_bvhNodes
					, bvh->nodes.size()
#if USE_SORT_BY_MATERIAL
					, dev_bsdfStructIDs
					, dev_bsdfStructIDs_copy
#endif
					);
			}
		}
#else
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


		intersect << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_triangles
			, pa->triangles.size()
			, dev_intersections
			, dev_bvhNodes
			, bvh->nodes.size()
#if USE_SORT_BY_MATERIAL
			, dev_bsdfStructIDs
			, dev_bsdfStructIDs_copy
#endif
			);
#endif

#if USE_SORT_BY_MATERIAL
		thrust::sort_by_key(thrust::device, dev_bsdfStructIDs, dev_bsdfStructIDs + num_paths, dev_paths);
		thrust::sort_by_key(thrust::device, dev_bsdfStructIDs_copy, dev_bsdfStructIDs_copy + num_paths, dev_intersections);
#endif // USE_SORT_BY_MATERIAL

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		shadeBSDF << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_bsdfStructs,
			dev_triangles,
			pa->triangles.size(),
			dev_bvhNodes,
			bvh->nodes.size()
			);
		cudaDeviceSynchronize();

		checkCUDAError("shadeMaterial failed\n");
		// Use dev_paths for compaction result
		// check if compacted dev_intersections has zero element, if it is, then iteraionComplete should be true.
		// If no, then we might be able to store the pixel that has not finished dev_intersection to largely decrease the number 
		// of pixels that are required to continue raytracing.
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, HasHit());
		num_paths = dev_path_end - dev_paths;

		iterationComplete = (num_paths == 0); // TODO: should be based off stream compaction results.
		//iterationComplete = (true);
		
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	// dim3 numBlocksPixels = (num_paths + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////
	//int imageWidth = hst_scene->state.camera.resolution.x;
	//int imageHeight = hst_scene->state.camera.resolution.y;
	//sampleScreen << <1, 1 >> > (dev_image, dev_textures[0], imageWidth, imageHeight);

	// Test texture sampling
	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);



	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
