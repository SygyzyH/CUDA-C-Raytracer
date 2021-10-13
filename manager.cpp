#include <inttypes.h>
#include <thread>
#include <mutex>
#include <chrono>
#include "RTAPIFunc.h"
#include "ManagerAPIFunc.h"
#include "objParser.h"

/* 
this is the secondary thread manager. the secondery thread will drive the main.cu API
with a double buffer to make the visual calculations constantly update, regrdless of
the current window driver.
*/

void threadDriver();
bool active = true, initialized = false;
int activeBuffer = 1;

uint32_t *buffer1, *buffer2;
std::mutex mutex;

std::thread RTThread(threadDriver);

void ManagerInit() {
    std::vector<poly *> mesh = parseObjectFiles(1, "objects/axis.obj");
    poly **nativeMesh = (poly **) malloc(mesh.size() * sizeof(poly *));
    for (int i = 0; i < mesh.size(); i++)
        nativeMesh[i] = mesh[i];

    sphere **nativeContext = (sphere **) malloc(sizeof(sphere *));
    nativeContext[0] = (sphere *) malloc(sizeof(sphere));
    vec *pos = (vec *) malloc(sizeof(vec));
    pos->size = 3;
    pos->value = (float *) malloc(3 * sizeof(float));
    pos->value[0] = 0.5f;
    pos->value[1] = -1;
    pos->value[2] = 0.5f;
    nativeContext[0]->pos = pos;
    nativeContext[0]->color = 0x007f7f00;
    nativeContext[0]->radius = 0.2;
    nativeContext[0]->reflectivity = 1;
    RTInit(nativeContext, 1, nativeMesh, mesh.size());
    RTThread.detach();
    buffer1 = RTEntryPoint();
    buffer2 = RTEntryPoint();
    initialized = true;
}

void ManagerCleanup() {
    active = false;
    RTCleanup();
    // free
}

uint32_t* ManagerGetPixelData() {
    mutex.lock();
    if (activeBuffer == 1)
        return buffer1;
    return buffer2;
}

void ManagerFreePixelData() {
    mutex.unlock();
}

void ManagerTranslateCamera(float x, float y, float z) {
    mutex.lock();
    RTTranslateCamera(x, y, z);
    mutex.unlock();
}

void ManagerRotateCamera(float yaw, float pitch) {
    mutex.lock();
    RTRotateCamera(yaw, pitch);
    mutex.unlock();
}

void threadDriver() {
    while (active) {
        if (initialized) {
            if (mutex.try_lock()) {
                if (activeBuffer == 1) {
                    buffer2 = RTEntryPoint();
                    activeBuffer = 2;
                } else {
                    buffer1 = RTEntryPoint();
                    activeBuffer = 1;
                }
                mutex.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
}
