#include <inttypes.h>
#ifndef RTAPIFUNC_H
#define RTAPIFUNC_H

uint32_t* RTEntryPoint();
void RTInit();
void RTCleanup();
void RTTranslateCamera(float x, float y, float z);

#endif