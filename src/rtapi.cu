#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "engine/cam.cuh"
#include "engine/color.cuh"
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
            float intersectingSphereX = 0;
            float intersectingSphereY = 0;
            float intersectingSphereZ = 0;
            float intersectingSphereR = 0;
            float intersectingSphereReflectivity = 0;
            uint32_t closestIntersectionC = 0;

            for (int i = 0; i < contextSize; i++) {
                // check for closest intersection
                float intersectionPoint = sphereGetIntersectionCuda(context[i], rOrigin1, rOrigin2, rOrigin3, rDiraction1, rDiraction2, rDiraction3);

                // if its the closest intersection, update
                if (intersectionPoint < closestIntersection) {
                    closestIntersection = intersectionPoint;
                    closestIntersectionC = context[i]->color;
                    intersectingSphereX = context[i]->pos->value[0];
                    intersectingSphereY = context[i]->pos->value[1];
                    intersectingSphereZ = context[i]->pos->value[2];
                    intersectingSphereR = context[i]->radius;
                    intersectingSphereReflectivity = context[i]->reflectivity;
                }
            }
            
            // once we have the closest intersection, we apply lighting, if there is an intersection
            if (closestIntersection != INFINITY) {
                for (int i = 0; i < lightSize; i++) {
                    // diffused lighting
                    // first we need the point of intersection in 3D space
                    float intersectionAt3DSpaceX = rOrigin1 + rDiraction1 * closestIntersection;
                    float intersectionAt3DSpaceY = rOrigin2 + rDiraction2 * closestIntersection;
                    float intersectionAt3DSpaceZ = rOrigin3 + rDiraction3 * closestIntersection;

                    // next, get the intersectiong point in regards to the light source
                    float intersectionPointToLight1 = lights[i]->value[0] - intersectionAt3DSpaceX;
                    float intersectionPointToLight2 = lights[i]->value[1] - intersectionAt3DSpaceY;
                    float intersectionPointToLight3 = lights[i]->value[2] - intersectionAt3DSpaceZ;
                    // normalize to make sure length stays consistant
                    float intersectionPointToLightL = vecLengthCuda(intersectionPointToLight1, intersectionPointToLight2, intersectionPointToLight3);
                    intersectionPointToLight1 /= intersectionPointToLightL;
                    intersectionPointToLight2 /= intersectionPointToLightL;
                    intersectionPointToLight3 /= intersectionPointToLightL;

                    // now we need the normal. the normal of a sphere at any point is the point itself
                    // but this must be localized and adjusted for radius
                    float sphereNormal1 = (intersectionAt3DSpaceX - intersectingSphereX) / intersectingSphereR;
                    float sphereNormal2 = (intersectionAt3DSpaceY - intersectingSphereY) / intersectingSphereR;
                    float sphereNormal3 = (intersectionAt3DSpaceZ - intersectingSphereZ) / intersectingSphereR;
                    // normalize 
                    float sphereNormalL = vecLengthCuda(sphereNormal1, sphereNormal2, sphereNormal3);
                    sphereNormal1 /= sphereNormalL;
                    sphereNormal2 /= sphereNormalL;
                    sphereNormal3 /= sphereNormalL;
                    
                    // TODO: check if anything is obscuring the light
                    // calculate diffused lighting
                    float diffused = intersectionPointToLight1 * sphereNormal1 + intersectionPointToLight2 * sphereNormal2 + intersectionPointToLight3 * sphereNormal3;
                    // clamp diffused lighting
                    diffused = max(AMBIENTILLUMINATION, min(1.0f, diffused));
                    // apply diffused lighting to color
                    closestIntersectionC = colorMult(closestIntersectionC, diffused);

                    // specular lighting
                    // get realtive camera direction to the point of intersection
                    float cameraDiraction1 = rOrigin1 - intersectionAt3DSpaceX;
                    float cameraDiraction2 = rOrigin2 - intersectionAt3DSpaceY;
                    float cameraDiraction3 = rOrigin3 - intersectionAt3DSpaceZ;
                    // normalize
                    float cameraDiractionL = vecLengthCuda(cameraDiraction1, cameraDiraction2, cameraDiraction3);
                    cameraDiraction1 /= cameraDiractionL;
                    cameraDiraction2 /= cameraDiractionL;
                    cameraDiraction3 /= cameraDiractionL;

                    // next, get the light source in regards to the intersectiong point
                    float lightToIntersectionPoint1 = intersectionAt3DSpaceX - lights[i]->value[0];
                    float lightToIntersectionPoint2 = intersectionAt3DSpaceY - lights[i]->value[1];
                    float lightToIntersectionPoint3 = intersectionAt3DSpaceZ - lights[i]->value[2];
                    // normalize to make sure length stays consistant
                    float lightToIntersectionPointL = vecLengthCuda(lightToIntersectionPoint1, lightToIntersectionPoint2, lightToIntersectionPoint3);
                    lightToIntersectionPoint1 /= lightToIntersectionPointL;
                    lightToIntersectionPoint2 /= lightToIntersectionPointL;
                    lightToIntersectionPoint3 /= lightToIntersectionPointL;
                    
                    // calculate reflection vector
                    float lightToIntersectionPointDotNorm = lightToIntersectionPoint1 * sphereNormal1 + lightToIntersectionPoint2 * sphereNormal2 + lightToIntersectionPoint3 * sphereNormal3;

                    float reflectionVector1 = lightToIntersectionPoint1 - 2 * sphereNormal1 * lightToIntersectionPointDotNorm;
                    float reflectionVector2 = lightToIntersectionPoint2 - 2 * sphereNormal2 * lightToIntersectionPointDotNorm;
                    float reflectionVector3 = lightToIntersectionPoint3 - 2 * sphereNormal3 * lightToIntersectionPointDotNorm;

                    // calculate specular lighting
                    float reflectionDotCameraDir = reflectionVector1 * cameraDiraction1 + reflectionVector2 * cameraDiraction2 + reflectionVector3 * cameraDiraction3;
                    float specularScalar = max(0.0f, min(1.0f, reflectionDotCameraDir));
                    float specular = specularScalar * specularScalar * intersectingSphereReflectivity;
                    // apply specular lighting to color
                    closestIntersectionC = colorAdd(closestIntersectionC, (uint32_t) (specular * 255));
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
    contextSize = 4;
    sphere **contextHost = (sphere **) malloc(contextSize * sizeof(sphere *));
    
    vec *spherePos1 = buildVec(3);
    spherePos1->value[0] = 2.0f;
    spherePos1->value[1] = 0;
    spherePos1->value[2] = 0;
    vec *spherePos2 = buildVec(3);
    spherePos2->value[0] = -0.7f;
    spherePos2->value[1] = -0.1f;
    spherePos2->value[2] = 0;
    vec *spherePos3 = buildVec(3);
    spherePos3->value[0] = 2.0f;
    spherePos3->value[1] = 0;
    spherePos3->value[2] = 1.0f;
    vec *spherePos4 = buildVec(3);
    spherePos4->value[0] = 0;
    spherePos4->value[1] = 1.1f;
    spherePos4->value[2] = -2.0f;
    sphere *sphereHost1 = buildSphere(1.0f, 1.0f, buildColor(0, 0, 255), spherePos1);
    sphere *sphereHost2 = buildSphere(0.3f, 1.0f, buildColor(0, 255, 0), spherePos2);
    sphere *sphereHost3 = buildSphere(0.6f, 1.0f, buildColor(255, 0, 0), spherePos3);
    sphere *sphereHost4 = buildSphere(0.5f, 1.0f, buildColor(127, 127, 127), spherePos4);
    contextHost[0] = buildSphereCudaCopy(sphereHost1);
    contextHost[1] = buildSphereCudaCopy(sphereHost2);
    contextHost[2] = buildSphereCudaCopy(sphereHost3);
    contextHost[3] = buildSphereCudaCopy(sphereHost4);

    cudaMalloc(&context, contextSize * sizeof(sphere *));
    cudaMemcpy(context, contextHost, contextSize * sizeof(sphere *), cudaMemcpyHostToDevice);

    freeVec(spherePos1);
    freeVec(spherePos2);
    freeVec(spherePos3);
    freeVec(spherePos4);
    free(sphereHost1);
    free(sphereHost2);
    free(sphereHost3);
    free(sphereHost4);
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
    renderer<<<(int) (ceilf(WIDTH * HEIGHT / 1024)), 1024>>>(resMat, pitch, camera, context, contextSize, lights, lightSize);

    // get back color matrix
    // profiling has shown this is a major bottleneck, responsible for a couple of ms of delay.
    // drawing directly from the GPU would be preferrable, but would require CUDA writing to a
    // texture that is later displayed using openGL. 
    cudaMemcpy2D(ans, WIDTH * sizeof(uint32_t), resMat, pitch, WIDTH * sizeof(uint32_t), HEIGHT, cudaMemcpyDeviceToHost);
    
    return ans;
}
