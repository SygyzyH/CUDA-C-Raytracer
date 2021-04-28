#include <inttypes.h>
#include "prim/types.h"

#ifndef RTAPIFUNC_H
#define RTAPIFUNC_H

uint32_t* RTEntryPoint();
void RTInit(sphere **contextI, int contextLength, poly **meshI, int meshLength);
void RTCleanup();
void RTTranslateCamera(float x, float y, float z);
void RTRotateCamera(float yaw, float pitch);

#endif