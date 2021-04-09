#include <inttypes.h>

#ifndef MANAGERAPIFUNC_h
#define MANAGERAPIFUNC_h

void ManagerInit();
void ManagerCleanup();
uint32_t* ManagerGetPixelData();
void ManagerFreePixelData();
void ManagerTranslateCamera(float x, float y, float z);
void ManagerRotateCamera(float yaw, float pitch);

#endif