#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "vec.cuh"

#ifndef POLYGON_CUH
#define POLYGON_CUH

typedef struct {
    vec *vert1, *vert2, *vert3;
    uint32_t color;
    float reflectivity;
} poly;

__host__
poly* buildPoly(vec *vert1, vec *vert2, vec *vert3, uint32_t color, float reflectivity) {
    poly *newPoly;

    newPoly = (poly *) malloc(sizeof(poly));

    newPoly->vert1 = vert1;
    newPoly->vert2 = vert2;
    newPoly->vert3 = vert3;
    newPoly->color = color;
    newPoly->reflectivity = reflectivity;

    return newPoly;
}

void freePoly(poly *source) {
    freeVec(source->vert1);
    freeVec(source->vert2);
    freeVec(source->vert3);
    free(source);
}

__host__
poly* buildPolyCudaCopy(poly *source) {
    poly *newPolyHost = (poly *) malloc(sizeof(poly));
    poly *newPolyDevice;

    // copy primitives
    newPolyHost->color = source->color;
    newPolyHost->reflectivity = source->reflectivity;

    // copy non-primitives
    cudaMalloc(&(newPolyHost->vert1), sizeof(vec));
    cudaMalloc(&(newPolyHost->vert2), sizeof(vec));
    cudaMalloc(&(newPolyHost->vert3), sizeof(vec));
    vec *sourceVert1CudaCopy = buildVecCudaCopy(source->vert1);
    vec *sourceVert2CudaCopy = buildVecCudaCopy(source->vert2);
    vec *sourceVert3CudaCopy = buildVecCudaCopy(source->vert3);
    cudaMemcpy(newPolyHost->vert1, sourceVert1CudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaMemcpy(newPolyHost->vert2, sourceVert2CudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaMemcpy(newPolyHost->vert3, sourceVert3CudaCopy, sizeof(vec), cudaMemcpyHostToDevice);
    cudaFree(sourceVert1CudaCopy);
    cudaFree(sourceVert2CudaCopy);
    cudaFree(sourceVert3CudaCopy);

    // link pointers
    cudaMalloc(&newPolyDevice, sizeof(poly));
    cudaMemcpy(newPolyDevice, newPolyHost, sizeof(poly), cudaMemcpyHostToDevice);
    free(newPolyHost);

    return newPolyDevice;
}

__host__
void freePolyCudaCopy(poly *polyDevice) {
    // make a pointer of all device pointers in host
    poly *tempHostPoly = (poly *) malloc(sizeof(poly));
    cudaMemcpy(tempHostPoly, polyDevice, sizeof(poly), cudaMemcpyDeviceToHost);

    // free them
    cudaFree(tempHostPoly->vert1);
    cudaFree(tempHostPoly->vert2);
    cudaFree(tempHostPoly->vert3);
    
    // free structs itself
    cudaFree(polyDevice);
    free(tempHostPoly);
}

__device__
float polyGetIntersectionCuda(poly *poly, float rOrg1, float rOrg2, float rOrg3, float rDir1, float rDir2, float rDir3) {
    float edge11 = poly->vert2->value[0] - poly->vert1->value[0];
    float edge12 = poly->vert2->value[1] - poly->vert1->value[1];
    float edge13 = poly->vert2->value[2] - poly->vert1->value[2];
    float edge21 = poly->vert3->value[0] - poly->vert1->value[0];
    float edge22 = poly->vert3->value[1] - poly->vert1->value[1];
    float edge23 = poly->vert3->value[2] - poly->vert1->value[2];
    // rDir cross edge2
    float h1 = rDir2 * edge23 - rDir3 * edge22;
    float h2 = rDir3 * edge21 - rDir1 * edge23;
    float h3 = rDir1 * edge22 - rDir2 * edge21;
    // edge1 dot h
    float a = edge11 * h1 + edge12 * h2 + edge13 * h3;

    if (a == 0.001)
        return INFINITY;
    
    float s1 = rOrg1 - poly->vert1->value[0];
    float s2 = rOrg2 - poly->vert1->value[1];
    float s3 = rOrg3 - poly->vert1->value[2];
    float f = 1 / a, u = f * (s1 * h1 + s2 * h2 + s3 * h3);

    if (u < 0.0f || u > 1.0f)
        return INFINITY;
    
    // s cross edge1
    float q1 = s2 * edge13 - s3 * edge12;
    float q2 = s3 * edge11 - s1 * edge13;
    float q3 = s1 * edge12 - s2 * edge11;
    float v = f * (rDir1 * q1 + rDir2 * q2 + rDir3 * q3);

    if (v < 0.0f || u + v > 1.0f)
        return INFINITY;
    
    float t = f * (edge21 * q1 + edge22 * q2 + edge23 * q3);
    if (t > 0.0001f)
        return t;
    return INFINITY;
}

#endif