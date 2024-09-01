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
    int id;
    int type = 0;
    float2 pos;
    float startConcentration, endConcentration;
    float3 extra;

    Morphogen(int id, float2 pos, float start, float end, float3 extra={0,0,0})
        : id(id), pos(pos), startConcentration(start), endConcentration(end), extra(extra) {
    }

    __host__ __device__ float sample(const float2& point);
};

#endif