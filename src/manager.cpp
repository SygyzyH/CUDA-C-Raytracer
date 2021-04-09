#include <inttypes.h>
#include <thread>
#include <mutex>
#include <chrono>
#include "RTAPIFunc.h"
#include "ManagerAPIFunc.h"

/* 
this is the secondary thread manager. the secondery thread will drive the main.cu API
with a double buffer to make the visual calculations constantly update, regrdless of
the current window driver.
*/
/*
TODO: this can still be improved. I belive the use of mutex is actually throtteling the application
because of the secondary thread being required to sleep to give the main thread a chance to do anything.
it could also be that this is the fastest view.cpp is able to achive using it's draw function.
*/

void threadDriver();
bool active = true, initialized = false;
int activeBuffer = 1;

uint32_t *buffer1, *buffer2;
std::mutex mutex;

std::thread RTThread(threadDriver);

void ManagerInit() {
    RTInit();
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