#include <stdio.h>
#include <math.h>
#include "../prim/vec.cuh"

#ifndef CAM_H
#define CAM_H

typedef struct {
    float FOV, aspectRatio;
    float height, width;
    vec *upg;
    vec *target, *origin;
    vec *forward, *right, *up;
} cam;

/*
cam struct: represents a camera in virtual space.

functions:
cam* buildCam - host & device. basic camera constructor.
cam* buildCamCudaCopy - host. constructs and returns a pointer to a deep copy of a cam in device memory.
cam* camRotate - host. rotate the camera using pitch and yaw.
void camTranslate - host. translate camera position in space.
ray* makeRay - device. make a ray using xy coordinates.
void camUpdateSpecificationVectors - host. updates specification vectors.
*/

__host__
static void camUpdateSpecificationVectors(cam *cam) {
    cam->forward = vecNorm(vecSub(cam->target, cam->origin));
    cam->right = vecNorm(vec3Cross(cam->upg, cam->forward));
    cam->up = vec3Cross(cam->right, cam->forward);
}

__host__ 
cam* buildCam(float FOV, float screenRatio, vec *upguide, vec *target, vec *origin) {
    if (FOV <= 0) {
        printf("Camera FOV of less than or 0 is not allowed.");
        return NULL;
    } else if (screenRatio <= 0) {
        printf("Camera screen ratio of less than or 0 is not allowed.");
        return NULL;
    }

    cam *newCam;

    newCam = (cam *) malloc(sizeof(cam));

    newCam->FOV = FOV;
    newCam->aspectRatio = screenRatio;

    newCam->upg = vecNorm(upguide);
    newCam->height = tanf(FOV);
    newCam->width = newCam->height * screenRatio;

    newCam->target = target;
    newCam->origin = origin;

    camUpdateSpecificationVectors(newCam);
    
    return newCam;
}

__host__
cam* buildCamCudaCopy(cam *source) {
    
    cam *newCamHost = (cam *) malloc(sizeof(cam));
    cam *newCamDevice;

    // copy primitives
    newCamHost->FOV = source->FOV;
    newCamHost->aspectRatio = source->aspectRatio;
    newCamHost->width = source->width;
    newCamHost->height = source->height;

    // copy non-primitives
    cudaMalloc(&(newCamHost->upg), sizeof(vec));
    cudaMalloc(&(newCamHost->target), sizeof(vec));
    cudaMalloc(&(newCamHost->origin), sizeof(vec));
    cudaMalloc(&(newCamHost->forward), sizeof(vec));
    cudaMalloc(&(newCamHost->right), sizeof(vec));
    cudaMalloc(&(newCamHost->up), sizeof(vec));
    vec *sourceUpgCudaCopy = buildVecCudaCopy(source->upg);
    vec *sourceTargetCudaCopy = buildVecCudaCopy(source->target);
    vec *sourceOriginCudaCopy = buildVecCudaCopy(source->origin);
    vec *sourceForwardCudaCopy = buildVecCudaCopy(source->forward);
    vec *sourceRightCudaCopy = buildVecCudaCopy(source->right);
    vec *sourceUpCudaCopy = buildVecCudaCopy(source->up);
    cudaMemcpy(newCamHost->upg, sourceUpgCudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaMemcpy(newCamHost->target, sourceTargetCudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaMemcpy(newCamHost->origin, sourceOriginCudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaMemcpy(newCamHost->forward, sourceForwardCudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaMemcpy(newCamHost->right, sourceRightCudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaMemcpy(newCamHost->up, sourceUpCudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaFree(sourceUpgCudaCopy);
    cudaFree(sourceTargetCudaCopy);
    cudaFree(sourceOriginCudaCopy);
    cudaFree(sourceForwardCudaCopy);
    cudaFree(sourceRightCudaCopy);
    cudaFree(sourceUpCudaCopy);
    
    // link pointers
    cudaMalloc(&newCamDevice, sizeof(cam));
    cudaMemcpy(newCamDevice, newCamHost, sizeof(cam), cudaMemcpyHostToDevice);
    free(newCamHost);

    return newCamDevice;
}

__host__
void freeCamCudaCopy(cam *cam) {
    cudaFree(cam->upg);
    cudaFree(cam->target);
    cudaFree(cam->origin);
    cudaFree(cam->forward);
    cudaFree(cam->right);
    cudaFree(cam->up);
    cudaFree(cam);
}

__host__
void camRotate(cam *cam, float yaw, float pitch) {
    // rotate the cameras target according to movement offsets in the screen.
    // first, translate the 2D coordinates of the screen offsets to 3D coordinates relative to camera;
    vec *mouseTranslation3D = vecAdd(vecMulScalar(cam->right, yaw), vecMulScalar(cam->up, pitch));

    // we need the direction vector local to the cameras origin instead of the world origin point, to make the
    // calculation simpler
    vec *localDiraction = vecSub(cam->target, cam->origin);

    // the relative new target will be the same distance from the last target to the new target, except it has
    // been moved by the mouse translation.
    vec *relativeNewTarget = vecAdd(mouseTranslation3D, cam->forward);

    // make the relation between the lengths the same
    vec *potentialTarget = vecAdd(cam->origin, vecMulScalar(relativeNewTarget, (float) vecLength(localDiraction) / vecLength(relativeNewTarget)));

    // if the pitch is too big, the forward vector may move behind the cameras origin. This will cause the camera to
    // flip, making to forward vector point the other way around. After which, if the player continues rotating down
    // the forward vector will once again end up behind the origin. To solve this, limit the camera's new target to
    // never be too out of the cameras origin.
    float diff = cam->origin->value[1] - potentialTarget->value[1];
    if (diff > -7 && diff < 7)
        potentialTarget = cam->target;

    // once the new target is guaranteed to be in range, apply the changes.
    cam->target = potentialTarget;

    // lastly, since we changed the target, we must update our guiding vectors.
    camUpdateSpecificationVectors(cam);
}

__host__
void camTranslate(cam *cam, vec *translationVector) {
    vec *movementVector;
    movementVector = vecMulScalar(cam->forward, translationVector->value[0]);
    movementVector = vecAdd(movementVector, vecMulScalar(cam->up, translationVector->value[1]));
    movementVector = vecAdd(movementVector, vecMulScalar(cam->right, translationVector->value[2]));

    cam->origin = vecAdd(cam->origin, movementVector);
    cam->target = vecAdd(cam->target, movementVector);
    free(movementVector);

    camUpdateSpecificationVectors(cam);
}

#endif