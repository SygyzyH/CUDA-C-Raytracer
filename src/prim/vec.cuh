#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef VEC_H
#define VEC_H

typedef struct {
    float *value;
    int size;
} vec;

/*
vec struct: represents a vector of any size.

functions:
buildVec - host & device. basic vector constructor.
buildVecCudaCopy - host. constructs and returns a pointer to a deep copy of a vector in device memory.
printVec - host. prints a vector structure.
float vecLength - host & device. returns vector length.
vec* vecNorm - host & device. normalizes a vector.
vec* vecAdd - host & device. adds two vectors.
vec* vecSub - host & device. subtracts two vectors.
vec* vecMulScalar - host & device. scalaric multiplication of a vector and a scalar.
vec* vecDivScalar - host & device. scalaric division of a vector and a scalar.
float vecDot - host & device. dot product of two vectors.
vec* vec3Cross - host & device. cross product of two vectors of size 3 only.
*/

//////////
// Utility
//////////
__device__ __host__
vec* buildVec(int size) {
    // common code to both archs
    if (size < 1) {
        printf("Vector size of less than 1 is not allowed.\n");
        return NULL;
    }
    
    vec *newVec;

    newVec = (vec *) malloc(sizeof(vec));
    newVec->value = (float *) malloc(size * sizeof(float));

    newVec->size = size;
    return newVec;
}

__host__
vec* buildVecCudaCopy(vec *source) {
    
    /*
    Since unified memomry (or managed memory, as NVIDIA calls it) is not implemented here, there is a
    need to pass the original vector struct to GPU memory. In my opinion, the most elegant and readable
    way to do that would be translating a copy from the CPU (a more efficiant way to do it would be to
    seperate the constructor to a __host__ and a __device__ implementation, but since there is no function
    overloading, this would be very confusing to program with). the plan to convert the object goes as
    follows, for any struct:

    - Construct a pointer in CPU memory of a dummy struct
    - Construct a pointer in GPU memory of the final struct (this will be the returned struct)
    - Construct a pointer in GPU memory to a deep copy of each element of the struct (not all elements
    need to be copied this way, only non-primitive objects and pointers).
    - Link the struct pointer in GPU to the struct pointer in CPU
    - Do a complete struct memcpy from the CPU struct to the final GPU struct

    By the end of the process, the struct should have been fully replecated in memory. Freeing the 
    original struct should be considered, since most uses of this function will only deal with the
    CPU copy tempereraly.
    */
    
    // length def is used for most of the process
    int len = source->size;

    // construct pointer to CPU copy
    vec *newVecHost = (vec *) malloc(sizeof(vec));
    // construct pointers to GPU copy
    vec *newVecDevice;
    
    // copy primitive elements to struct
    newVecHost->size = len;
    
    // copy non-primitive elements to struct
    // malloc GPU copies in GPU mem
    cudaMalloc(&(newVecHost->value), len * sizeof(float));
    // deep copy non-primitives from CPU to GPU
    cudaMemcpy(newVecHost->value, source->value, len * sizeof(float), cudaMemcpyHostToDevice);

    // link CPU and GPU pointer values
    cudaMalloc(&newVecDevice, sizeof(vec));
    cudaMemcpy(newVecDevice, newVecHost, sizeof(vec), cudaMemcpyHostToDevice);
    free(newVecHost);
    
    return newVecDevice;
}

__device__ __host__
void printVec(vec *vec) {
    int size = vec->size;
    if (size != 3) {
        printf("v[");
        for (int i = 0; i < size; i++) 
            if (i < size - 1)
                printf("%f, ", vec->value[i]);
            else
                printf("%f]\n", vec->value[i]);
    } else {
        printf("v[%f, %f, %f]\n", vec->value[0], vec->value[1], vec->value[2]);
    }
}

//////////////
// Mathematics
//////////////
__device__ __host__
float vecLength(vec *vec) {
    float length = 0;

    for (int i = 0; i < vec->size; i++) {
        float value = vec->value[i];
        length += value * value;
    }
    length = sqrtf(length);

    return length;
}

__device__
float vecLengthCuda(float a, float b, float c) {
    float length = 0;
    length += a * a;
    length += b * b;
    length += c * c;

    return sqrtf(length);
}

__device__ __host__
vec* vecNorm(vec *vec) {
    float length = vecLength(vec);

    for (int i = 0; i < vec->size; i++)
        vec->value[i] /= length;
    
    return vec;
}

__device__ __host__
vec* vecAdd(vec *vec1, vec *vec2) {
    if (vec1->size != vec2->size) {
        printf("Size mismatch: Vector1 is of size %d, while Vector2 is of size %d.\n", vec1->size, vec2->size);
        return NULL;
    }
    int size = vec1->size;

    vec *ans = buildVec(size);
    for (int i = 0; i < size; i++)
        ans->value[i] = vec1->value[i] + vec2->value[i];

    return ans;
}

__device__ __host__
vec* vecSub(vec *vec1, vec *vec2) {
    if (vec1->size != vec2->size) {
        printf("Size mismatch: Vector1 is of size %d, while Vector2 is of size %d.\n", vec1->size, vec2->size);
        return NULL;
    }
    int size = vec1->size;

    vec *ans = buildVec(size);
    for (int i = 0; i < size; i++)
        ans->value[i] = vec1->value[i] - vec2->value[i];

    return ans;
}

__device__ __host__
vec* vecMulScalar(vec *vec1, float scalar) {
    vec *ans = buildVec(vec1->size);
    for (int i = 0; i < ans->size; i++)
        ans->value[i] = vec1->value[i] * scalar;
    
    return ans;
}

__device__ __host__
vec* vecDivScalar(vec *vec1, float scalar) {
    vec *ans = buildVec(vec1->size);
    for (int i = 0; i < ans->size; i++)
        ans->value[i] = vec1->value[i] / scalar;
    
    return ans;
}

__device__ __host__
float vecDot(vec *vec1, vec *vec2) {
    if (vec1->size != vec2->size) {
        printf("Size mismatch: Vector1 is of size %d, while Vector2 is of size %d.\n", vec1->size, vec2->size);
        return -1;
    }
    int size = vec1->size;
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
        sum += vec1->value[i] * vec2->value[i];
    
    return sum;
}

__device__ __host__
vec* vec3Cross(vec *vec1, vec *vec2) {
    if (vec1->size != 3 || vec2->size != 3) {
        printf("Cross operation can currently only be done with vectors of size 3.");
        return NULL;
    }
    int size = vec1->size;

    vec *ans = buildVec(size);
    ans->value[0] = vec1->value[1] * vec2->value[2] - vec1->value[2] * vec2->value[1];
    ans->value[1] = vec1->value[2] * vec2->value[0] - vec1->value[0] * vec2->value[2];
    ans->value[2] = vec1->value[0] * vec2->value[1] - vec1->value[1] * vec2->value[0];

    return ans;
}

#endif