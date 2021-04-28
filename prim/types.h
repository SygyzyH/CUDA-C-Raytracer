#include <stdint.h>

#ifndef TYPES_H
#define TYPES_H

typedef struct {
    float *value;
    int size;
} vec;

typedef struct {
    vec *vert1, *vert2, *vert3;
    uint32_t color;
    float reflectivity;
} poly;

typedef struct {
    float radius;
    float reflectivity;
    uint32_t color;
    vec *pos;
} sphere;

#endif