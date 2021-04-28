#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "vec.cuh"
#include "types.h"

#ifndef SPHERE_H
#define SPHERE_H

/*
sphere struct: represents a sphere in space.
functions:
sphere* buildSphere - host & device. basic sphere constructor.
sphere* buildSphereCudaCopy - host. constructs and returns a pointer to a deep copy of a sphere in device memory.
float sphereGetIntersection - device. returns point of intersection with a ray.
vec* sphereGetNormal - device. returns sphere normal at a point.
*/

__host__
sphere* buildSphere(float radius, float reflectivity, uint32_t color, vec *pos) {
    if (radius <= 0) {
        printf("Sphere radius of less than or 0 is not allowed.");
        return NULL;
    }

    sphere *newSphere;

    newSphere = (sphere *) malloc(sizeof(sphere));

    newSphere->radius = radius;
    newSphere->reflectivity = reflectivity;
    newSphere->pos = pos;
    newSphere->color = color;

    return newSphere;
}

void freeSphere(sphere *source) {
    freeVec(source->pos);
    free(source);
}

__host__
sphere* buildSphereCudaCopy(sphere *source) {

    sphere *newSphereHost = (sphere *) malloc(sizeof(sphere));
    sphere *newSphereDevice;

    // copy primitives
    newSphereHost->radius = source->radius;
    newSphereHost->reflectivity = source->reflectivity;
    newSphereHost->color = source->color;

    // copy non-primitives
    // TODO: this needs to be tested, to make sure there is no segf
    cudaMalloc(&(newSphereHost->pos), sizeof(vec));
    vec *sourcePosCudaCopy = buildVecCudaCopy(source->pos);
    cudaMemcpy(newSphereHost->pos, sourcePosCudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaFree(sourcePosCudaCopy);

    //link pointers
    cudaMalloc(&newSphereDevice, sizeof(sphere));
    cudaMemcpy(newSphereDevice, newSphereHost, sizeof(sphere), cudaMemcpyHostToDevice);
    free(newSphereHost);

    return newSphereDevice;
}

__host__
void freeSphereCudaCopy(sphere *sphereDevice) {
    // make a pointer of all device pointers in host
    sphere *tempHostSphere = (sphere *) malloc(sizeof(sphere));
    cudaMemcpy(tempHostSphere, sphereDevice, sizeof(sphere), cudaMemcpyDeviceToHost);

    // free them
    cudaFree(tempHostSphere->pos);
    
    // free structs itself
    cudaFree(sphereDevice);
    free(tempHostSphere);
}

__device__
float sphereGetIntersectionCuda(sphere *sphere, float rOrg1, float rOrg2, float rOrg3, float rDir1, float rDir2, float rDir3) {
    /* for finding a ray and sphere intersection, the derived formula is a quadratic formula, since the ray can
    intersect at two, one, or zero points. were only interested in the closest one;
    the sphere is also localized, to cancel out the translation factors and make the formula a little easier to compute.
    
    the formula is:
    a = length(ray dir)^2
    b = 2 * ray pos DOT ray dir
    c = length(ray pos)^2 - rad^2
    */
   // translate ray to point of origin
   float localOrigin1 = rOrg1 - sphere->pos->value[0];
   float localOrigin2 = rOrg2 - sphere->pos->value[1];
   float localOrigin3 = rOrg3 - sphere->pos->value[2];

   // using the formula, get a, b, and c
   float len = vecLengthCuda(rDir1, rDir2, rDir3);
   float a = len * len;
   float b = 2 * (localOrigin1 * rDir1 + localOrigin2 * rDir2 + localOrigin3 * rDir3);
   len = vecLengthCuda(localOrigin1, localOrigin2, localOrigin3);
   float c = len * len - (sphere->radius * sphere->radius);

   // to find if an intersection even exists, we just need the discriminant; since its in the square root, if its
   // negative the function is undefined, and thus there are no common points between the ray and the sphere.
   float discriminant = (b * b) - (4 * a * c);

   // if the discriminate is negative, it has no square root and thus no solution. in that case, there is no intersection
   if (discriminant < 0)
        return INFINITY;

    discriminant = sqrtf(discriminant);

    // there is at least one point of intersection. find it
    float p1 = (-b - discriminant) / (2 * a);
    float p2 = (-b + discriminant) / (2 * a);

    // were only interested in the closest point of intersection, therefore, we only need to return the smallest one.
    // were also not interested in intersections behind the origin of the ray, meaning no negative results
    if (p1 > 0.001f && p1 < p2)
        return p1;
    else if (p2 > 0.001f)
        return p2;
    return INFINITY;
}

#endif