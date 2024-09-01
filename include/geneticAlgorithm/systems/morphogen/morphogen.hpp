#ifndef MORPHOGEN
#define MORPHOGEN

#include <map>
#include <vector>
#include <vector_types.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * A morphogen is a signalling chemical used in a morphogen cascade to describe the morphological development
 * of a creature. These are determined by hox genes.
 */
struct Morphogen {
    int type = 0;
    float angle;
    float startConcentration, endConcentration;
    float2 extra;

    Morphogen(int type, float angle, float start, float end, float2 extra={0,0})
        : type(type), angle(angle), startConcentration(start), endConcentration(end), extra(extra) {
    }

    __host__ __device__ float sample(float2 cellPos, float cellRadius,
                                     float distFraction, const float2& point) const;
};

#endif