#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "engine/cam.cuh"
#include "prim/sphere.cuh"
#include "prim/vec.cuh"

// constants
#define M_PI 3.14159265358979323846

const int WIDTH = 1980/5, HEIGHT = 1080/5;
const float screenRatio = WIDTH / HEIGHT;
const float fov = (float) 25 * (M_PI / 180);

__global__
void renderer(uint32_t *resMat, size_t pitch, cam *camera, sphere **context, int contextSize) {
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

int main() {
    // print some device infromation.
    // failing these runtime API calls would cause the program to crash - not the best way of testing for
    // compatible devices, but one nonetheless.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Found compatible device: %s with compute-capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);

    // define result matrix
    uint32_t *resMat;
    uint32_t *ans = (uint32_t *) malloc(WIDTH * HEIGHT * sizeof(uint32_t));

    size_t pitch;
    cudaMallocPitch(&resMat, &pitch, WIDTH * sizeof(uint32_t), HEIGHT);

    // define context
    int contextSize = 2;
    sphere **contextHost = (sphere **) malloc(contextSize * sizeof(sphere *));
    sphere **context;
    
    vec *spherePos1 = buildVec(3);
    spherePos1->value[0] = 1.2;
    spherePos1->value[1] = 0;
    spherePos1->value[2] = 0;
    vec *spherePos2 = buildVec(3);
    spherePos2->value[0] = -0.7;
    spherePos2->value[1] = -0.1f;
    spherePos2->value[2] = 0;
    sphere *sphereHost1 = buildSphere(0.6f, 4, spherePos1);
    sphere *sphereHost2 = buildSphere(0.3f, 5, spherePos2);
    contextHost[0] = buildSphereCudaCopy(sphereHost1);
    contextHost[1] = buildSphereCudaCopy(sphereHost2);

    cudaMalloc(&context, contextSize * sizeof(sphere *));
    cudaMemcpy(context, contextHost, contextSize * sizeof(sphere *), cudaMemcpyHostToDevice);

    free(spherePos1);
    free(spherePos2);
    free(sphereHost1);
    free(sphereHost2);
    free(contextHost);

    // define camera
    // this process can be automized, but it will be a part of the heigher-language
    // calls later on.
    vec *upguide = buildVec(3);
    upguide->value[0] = 0;
    upguide->value[1] = 1;
    upguide->value[2] = 0;
    vec *target = buildVec(3);
    target->value[0] = 1.0f;
    target->value[1] = 1.0f;
    target->value[2] = 1.0f;
    vec *origin = buildVec(3);
    origin->value[0] = 0;
    origin->value[1] = 0;
    origin->value[2] = 0;
    cam *cameraHost = buildCam(fov, screenRatio, upguide, target, origin);
    cam *camera = buildCamCudaCopy(cameraHost);

    free(upguide);
    free(target);
    free(origin);
    free(cameraHost);

    // call renderer
    renderer<<<(int) (ceilf(WIDTH * HEIGHT / 512)), 512>>>(resMat, pitch, camera, context, contextSize);

    // get back color matrix
    cudaMemcpy2D(ans, WIDTH * sizeof(uint32_t), resMat, pitch, WIDTH * sizeof(uint32_t), HEIGHT, cudaMemcpyDeviceToHost);

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (i % WIDTH == 0)
            printf("\n");
        printf("%d, ", ans[i]);
    }

    // free rendering context intilaization
    freeSphereCudaCopy(context);
    freeCamCudaCopy(camera);
    
    // free results
    cudaFree(resMat);
    free(ans);

    return 0;
}