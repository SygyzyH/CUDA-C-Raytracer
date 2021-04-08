// handles key pressing
#include "settings.h"

#ifndef HANDLER_H
#define HANDLER_H

enum diraction { FORWARD = 0b00000001, BACKWORD = 0b00000010, LEFT = 0b00000100,
                 RIGHT = 0b00001000, UP = 0b00010000, DOWN = 0b00100000};

short activeKeys = 0x00;

void handleKeys(float *x, float *y, float *z) {
    *x = ((activeKeys & FORWARD) >> 0) * MOVESPEED + ((activeKeys & BACKWORD) >> 1) * -MOVESPEED;
    *y = ((activeKeys & UP) >> 4) * -0.01 + ((activeKeys & DOWN) >> 5) * 0.01;
    *z = ((activeKeys & RIGHT) >> 3) * MOVESPEED + ((activeKeys & LEFT) >> 2) * -MOVESPEED;
}

#endif