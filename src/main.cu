#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "engine/cam.cuh"
#include "prim/sphere.cuh"
#include "prim/vec.cuh"
#include "settings.h"

// globals
// TODO: some (if not all) of these should be passed as a parameter.
uint32_t *resMat;
uint32_t *ans;
size_t pitch;
int contextSize;
int lightSize;
sphere **context;
vec **lights;
cam *camera;
// to make it faster to change the camera, a host copy is also kept
cam *cameraHost;

__global__
void renderer(uint32_t *resMat, size_t pitch, cam *camera, sphere **context, int contextSize, vec **lights, int lightSize) {
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < HEIGHT; i += blockDim.y * gridDim.y) {
        uint32_t *resMatRow = (uint32_t *) ((char *) resMat + i * pitch);
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < WIDTH; j += blockDim.x * gridDim.x) {
            float x = (float) 2 * j / WIDTH - 1;
            float y = (float) 2 * i / HEIGHT - 1;

            // make ray
            // origin
            float rOrigin1, rOrigin2, rOrigin3;
            rOrigin1 = camera->origin->value[0];
            rOrigin2 = camera->origin->value[1];
            rOrigin3 = camera->origin->value[2];
            // diraction
            float rDiraction1, rDiraction2, rDiraction3;
            // calculate diraction
            float rRightScale1 = camera->right->value[0] * camera->width * x;
            float rRightScale2 = camera->right->value[1] * camera->width * x;
            float rRightScale3 = camera->right->value[2] * camera->width * x;
            float rUpScale1 = camera->up->value[0] * camera->height * y;
            float rUpScale2 = camera->up->value[1] * camera->height * y;
            float rUpScale3 = camera->up->value[2] * camera->height * y;
            float rSumUp1 = rUpScale1 + rRightScale1;
            float rSumUp2 = rUpScale2 + rRightScale2;
            float rSumUp3 = rUpScale3 + rRightScale3;
            rDiraction1 = camera->forward->value[0] + rSumUp1;
            rDiraction2 = camera->forward->value[1] + rSumUp2;
            rDiraction3 = camera->forward->value[2] + rSumUp3;
            // normalize
            float rDiractionLength = vecLengthCuda(rDiraction1, rDiraction2, rDiraction3);
            rDiraction1 /= rDiractionLength;
            rDiraction2 /= rDiractionLength;
            rDiraction3 /= rDiractionLength;

            // next, cast the ray to get the color
            float closestIntersection = INFINITY;
            uint32_t closestIntersectionC = 0;

            for (int i = 0; i < contextSize; i++) {
                // check for closest intersection
                float intersectionPoint = sphereGetIntersectionCuda(context[i], rOrigin1, rOrigin2, rOrigin3, rDiraction1, rDiraction2, rDiraction3);

                // if its the closest intersection, update
                if (intersectionPoint < closestIntersection) {
                    closestIntersection = intersectionPoint;
                    closestIntersectionC = context[i]->color;
                }
            }

            // return the closest intersection color
            resMatRow[j] = closestIntersectionC;
        }
    }
}

void RTInit() {
    // TODO: CUDA DLL's sanity check
    // define result matricies
    ans = (uint32_t *) malloc(WIDTH * HEIGHT * sizeof(uint32_t));
    cudaMallocPitch(&resMat, &pitch, WIDTH * sizeof(uint32_t), HEIGHT);

    // define context
    contextSize = 2;
    sphere **contextHost = (sphere **) malloc(contextSize * sizeof(sphere *));
    
    vec *spherePos1 = buildVec(3);
    spherePos1->value[0] = 2.0;
    spherePos1->value[1] = 0;
    spherePos1->value[2] = 0;
    vec *spherePos2 = buildVec(3);
    spherePos2->value[0] = -0.7;
    spherePos2->value[1] = -0.1f;
    spherePos2->value[2] = 0;
    sphere *sphereHost1 = buildSphere(1.0f, 0x0050aaff, spherePos1);
    sphere *sphereHost2 = buildSphere(0.3f, 0x0000ff00, spherePos2);
    contextHost[0] = buildSphereCudaCopy(sphereHost1);
    contextHost[1] = buildSphereCudaCopy(sphereHost2);

    cudaMalloc(&context, contextSize * sizeof(sphere *));
    cudaMemcpy(context, contextHost, contextSize * sizeof(sphere *), cudaMemcpyHostToDevice);

    freeVec(spherePos1);
    freeVec(spherePos2);
    free(sphereHost1);
    free(sphereHost2);
    free(contextHost);

    // define lighting
    lightSize = 1;
    vec **lightsHost = (vec **) malloc(lightSize * sizeof(vec *));

    vec *lightPos1 = buildVec(3);
    lightPos1->value[0] = 0.0f;
    lightPos1->value[1] = 2.0f;
    lightPos1->value[2] = 0.0f;
    lightsHost[0] = buildVecCudaCopy(lightPos1);

    cudaMalloc(&lights, lightSize * sizeof(vec *));
    cudaMemcpy(lights, lightsHost, lightSize * sizeof(vec *), cudaMemcpyHostToDevice);

    freeVec(lightPos1);
    free(lightsHost);

    // define camera
    vec *upguide = buildVec(3);
    upguide->value[0] = 0;
    upguide->value[1] = 1;
    upguide->value[2] = 0;
    vec *target = buildVec(3);
    target->value[0] = 1.0f;
    target->value[1] = 0.0f;
    target->value[2] = 0.0f;
    vec *origin = buildVec(3);
    origin->value[0] = -2.0f;
    origin->value[1] = 0;
    origin->value[2] = 0;
    cameraHost = buildCam(FOV, screenRatio, upguide, target, origin);
    camera = buildCamCudaCopy(cameraHost);
}

void RTCleanup() {
    // free the rest of the variables used constently.
    cudaFree(resMat);
    free(ans);
    freeCamCudaCopy(camera);
    freeCam(cameraHost);

    // free context
    // make a pointer of all device pointers in host
    sphere **contextHost = (sphere **) malloc(contextSize * sizeof(sphere));
    cudaMemcpy(contextHost, context, contextSize * sizeof(sphere), cudaMemcpyDeviceToHost);
    // free each pointer in array
    for (int i = 0; i < contextSize; i++)
        freeSphereCudaCopy(contextHost[i]);
    // free the rest of the structs
    cudaFree(context);
    free(contextHost);
}

void RTTranslateCamera(float x, float y, float z) {
    camTranslate(cameraHost, x, y, z);

    // update device camera
    // can be optimized, instead of creating a new instance, simply update the existing one (if its faster)
    freeCamCudaCopy(camera);
    camera = buildCamCudaCopy(cameraHost);
}

void RTRotateCamera(float yaw, float pitch) {
    camRotate(cameraHost, yaw, pitch);

    // update device camera
    // can be optimized, instead of creating a new instance, simply update the existing one (if its faster)
    freeCamCudaCopy(camera);
    camera = buildCamCudaCopy(cameraHost);
}

uint32_t* RTEntryPoint() {
    // print some device infromation.
    // TODO: failing these runtime API calls would cause the program to crash - not the best way of testing for
    // compatible devices, but one nonetheless. see RTInit for a better solution.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Found compatible device: %s with compute-capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);

    // call renderer
    renderer<<<(int) (ceilf(WIDTH * HEIGHT / 512)), 512>>>(resMat, pitch, camera, context, contextSize, lights, lightSize);

    // get back color matrix
    // TODO: this should be double - buffer swapped, so that the GPU controller can be threaded.
    cudaMemcpy2D(ans, WIDTH * sizeof(uint32_t), resMat, pitch, WIDTH * sizeof(uint32_t), HEIGHT, cudaMemcpyDeviceToHost);

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (i % WIDTH == 0)
            printf("\n");
        printf("%d, ", ans[i]);
    }

    return ans;
}