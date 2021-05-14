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
    freeVec(cam->forward);
    freeVec(cam->right);
    freeVec(cam->up);

    cam->forward = vecNorm(vecSub(cam->target, cam->origin));
    cam->right = vec3Cross(cam->upg, cam->forward);
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

    // these are not nesecery, but its just so it could be freed later when camUpdateSpecificationVectors is called
    newCam->forward = buildVec(3);
    newCam->right = buildVec(3);
    newCam->up = buildVec(3);

    camUpdateSpecificationVectors(newCam);
    
    return newCam;
}

void freeCam(cam *source) {
    freeVec(source->target);
    freeVec(source->origin);
    freeVec(source->upg);
    freeVec(source->forward);
    freeVec(source->right);
    freeVec(source->up);
    free(source);
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
void freeCamCudaCopy(cam *camera) {
    // make a pointer of all device pointers in host
    cam *tempHostCam = (cam *) malloc(sizeof(cam));
    cudaMemcpy(tempHostCam, camera, sizeof(cam), cudaMemcpyDeviceToHost);

    // free them
    cudaFree(tempHostCam->upg);
    cudaFree(tempHostCam->target);
    cudaFree(tempHostCam->origin);
    cudaFree(tempHostCam->forward);
    cudaFree(tempHostCam->right);
    cudaFree(tempHostCam->up);
    
    // free struct itself
    free(tempHostCam);
    cudaFree(camera);
}

__host__
void camRotate(cam *cam, float yaw, float pitch) {
    // rotate the cameras target according to movement offsets in the screen.
    // first, translate the 2D coordinates of the screen offsets to 3D coordinates relative to camera;
    vec *rightScale = vecMulScalar(cam->right, yaw);
    vec *upScale = vecMulScalar(cam->up, pitch);
    vec *mouseTranslation3D = vecAdd(rightScale, upScale);

    freeVec(rightScale);
    freeVec(upScale);

    // we need the direction vector local to the cameras origin instead of the world origin point, to make the
    // calculation simpler
    vec *localDiraction = vecSub(cam->target, cam->origin);

    // the relative new target will be the same distance from the last target to the new target, except it has
    // been moved by the mouse translation.
    vec *relativeNewTarget = vecAdd(mouseTranslation3D, cam->forward);

    freeVec(mouseTranslation3D);

    // make the relation between the lengths the same
    // TODO: memory leak.
    vec *potentialTarget = vecAdd(cam->origin, vecMulScalar(relativeNewTarget, (float) vecLength(localDiraction) / vecLength(relativeNewTarget)));

    freeVec(localDiraction);
    freeVec(relativeNewTarget);
    // if the pitch is too big, the forward vector may move behind the cameras origin. This will cause the camera to
    // flip, making to forward vector point the other way around. After which, if the player continues rotating down
    // the forward vector will once again end up behind the origin. To solve this, limit the camera's new target to
    // never be too out of the cameras origin.
    // TODO: memory leak. this whole part is not even nessecery. its better to just use a counter to make sure
    // yaw never goes above (or below negative) 180 degrees.
    /*float diff = cam->origin->value[1] - potentialTarget->value[1];
    if (diff > -7 && diff < 7) {
        potentialTarget = cam->target;
    }*/

    // once the new target is guaranteed to be in range, apply the changes.
    cam->target = potentialTarget;

    // lastly, since we changed the target, we must update our guiding vectors.
    camUpdateSpecificationVectors(cam);
}

__host__
void camTranslate(cam *cam, float x, float y, float z) {
    vec *movementVector;
    vec *movementVector1 = vecMulScalar(cam->forward, x);
    vec *movementVector2 = vecMulScalar(cam->up, y);
    vec *movementVector3 = vecMulScalar(cam->right, z);
    vec *movementVectorSum = vecAdd(movementVector1, movementVector2);
    movementVector = vecAdd(movementVectorSum, movementVector3);
    freeVec(movementVector1);
    freeVec(movementVector2);
    freeVec(movementVector3);
    freeVec(movementVectorSum);

    // save the old vectors to free them later
    vec *pOrigin = cam->origin, *pTarget = cam->target;
    
    // update
    cam->origin = vecAdd(cam->origin, movementVector);
    cam->target = vecAdd(cam->target, movementVector);
    
    // free
    freeVec(movementVector);
    freeVec(pOrigin);
    freeVec(pTarget);

    camUpdateSpecificationVectors(cam);
}

#endif
