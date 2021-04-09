#include <stdint.h>

#ifndef COLOR_H
#define COLOR_H

const int REDMASK = 0x000000ff;
const int GREENMASK = 0x0000ff00;
const int BLUEMASK = 0x00ff0000;

__device__ __host__
uint32_t buildColor(uint32_t R, uint32_t G, uint32_t B) {
    return (R << 0) + (G << 8) + (B << 16);
}

/*
lerp functions would only be used in the future, for texturing. however, i belive CUDA has a prebuilt alternative.
in any case, these are here.
*/
__device__
static float lerp(float v0, float v1, float t) {
    return (1 - t) * v0 + t * v1;
}

__device__
uint32_t colorLerp(uint32_t c1, uint32_t c2, float factor) {
    // deconstruct colors
    float c1R = (float) ((c1 & REDMASK) >> 0);
    float c1G = (float) ((c1 & GREENMASK) >> 8);
    float c1B = (float) ((c1 & BLUEMASK) >> 16);

    float c2R = (float) ((c2 & REDMASK) >> 0);
    float c2G = (float) ((c2 & GREENMASK) >> 8);
    float c2B = (float) ((c2 & BLUEMASK) >> 16);

    return buildColor((uint32_t) lerp(c1R, c2R, factor), (uint32_t) lerp(c1G, c2G, factor), (uint32_t) lerp(c1B, c2B, factor));
}

__device__
uint32_t colorAdd(uint32_t c, uint32_t factor) {
    uint32_t cR = (uint32_t) ((c & REDMASK) >> 0) + factor;
    uint32_t cG = (uint32_t) ((c & GREENMASK) >> 8) + factor;
    uint32_t cB = (uint32_t) ((c & BLUEMASK) >> 16) + factor;
    return buildColor(min(cR, 255), min(cG, 255), min(cB, 255));
}

__device__
uint32_t colorMult(uint32_t c, float scalar) {
    uint32_t cR = (uint32_t) (((c & REDMASK) >> 0) * scalar);
    uint32_t cG = (uint32_t) (((c & GREENMASK) >> 8) * scalar);
    uint32_t cB = (uint32_t) (((c & BLUEMASK) >> 16) * scalar);
    return buildColor(min(cR, 255), min(cG, 255), min(cB, 255));
}

#endif