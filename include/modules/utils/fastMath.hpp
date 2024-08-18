#ifndef FAST_MATH
#define FAST_MATH

#include <cmath>
#include <iostream>

#ifndef M_PI_2
    #define M_PI_2 6.283185f
#endif

class FastMath {
public:
    static const int TABLE_SIZE = 360;
    static float cosTable[TABLE_SIZE];
    static float sinTable[TABLE_SIZE];

    static void init();

    static float cos(float angle);
    static float sin(float angle);

private:
    static float getValue(float angle, const float* table);
};

#endif