#version 130

uniform sampler2D texture;
uniform vec2 resolution;
uniform float seed;

// Generation params
uniform vec4 colours[25];
uniform int numColours;
uniform float noiseWarp;
uniform float noiseFrequency;
uniform int noiseOctaves;
uniform bool smoothNoise;
uniform float time;
uniform vec2 offset;

varying vec2 vTexCoord;

const int PERMUTATION_SIZE = 256;
const int permutation[256] = int[](151,160,137,91,90,15,
                                       131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
                                       190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
                                       88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
                                       77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
                                       102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
                                       135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
                                       5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
                                       223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
                                       129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
                                       251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
                                       49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
                                       138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180);


float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float grad(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

float noise3D(float x, float y, float z) {
    float _x = floor(x);
    float _y = floor(y);
    float _z = floor(z);

    int ix = int(_x) & 255;
    int iy = int(_y) & 255;
    int iz = int(_z) & 255;

    float fx = x - _x;
    float fy = y - _y;
    float fz = z - _z;

    float u = fade(fx);
    float v = fade(fy);
    float w = fade(fz);

    int A  = (permutation[ix & 255] + iy) & 255;
    int B  = (permutation[(ix + 1) & 255] + iy) & 255;
    int AA = (permutation[A] + iz) & 255;
    int AB = (permutation[(A + 1) & 255] + iz) & 255;
    int BA = (permutation[B] + iz) & 255;
    int BB = (permutation[(B + 1) & 255] + iz) & 255;

    float p0 = grad(permutation[AA], fx, fy, fz);
    float p1 = grad(permutation[BA], fx - 1.0, fy, fz);
    float p2 = grad(permutation[AB], fx, fy - 1.0, fz);
    float p3 = grad(permutation[BB], fx - 1.0, fy - 1.0, fz);
    float p4 = grad(permutation[AA + 1], fx, fy, fz - 1.0);
    float p5 = grad(permutation[BA + 1], fx - 1.0, fy, fz - 1.0);
    float p6 = grad(permutation[AB + 1], fx, fy - 1.0, fz - 1.0);
    float p7 = grad(permutation[BB + 1], fx - 1.0, fy - 1.0, fz - 1.0);

    float q0 = mix(p0, p1, u);
    float q1 = mix(p2, p3, u);
    float q2 = mix(p4, p5, u);
    float q3 = mix(p6, p7, u);

    float r0 = mix(q0, q1, v);
    float r1 = mix(q2, q3, v);

    return mix(r0, r1, w);
}

vec2 gradientNoise2D(vec2 pos) {
    float x = noise3D(pos.x, pos.y, 0.0);
    float y = noise3D(pos.x + 5.2, pos.y + 1.3, 0.0);
    return vec2(x, y) * 2.0 - 1.0;
}

float warpedNoise(vec2 pos, float warpAmount) {
    vec2 offset = gradientNoise2D(pos) * warpAmount;
    return noise3D(pos.x + offset.x, pos.y + offset.y, time);
}

float fbm_warped(vec2 pos) {
    float sum = 0.0;
    float amplitude = 1.0;
    float frequency = noiseFrequency;
    float warpAmount = noiseWarp;

    for (int i = 0; i < noiseOctaves; i++) {
        sum += warpedNoise(pos * frequency, warpAmount) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
        warpAmount *= 0.5;
    }

    return sum;
}

void main() {
    float x = (gl_TexCoord[0].x * resolution.x + offset.x)/500 + seed*1000;
    float y = (gl_TexCoord[0].y * resolution.y + offset.y)/500;

    float noiseValue = fbm_warped(vec2(x, y));

    // Normalize the noise value to [0, 1] range
    noiseValue = clamp((noiseValue*2.0 + 1.0) * 0.5, 0.0, 1.0);

    // Find the appropriate color based on noise value
    vec4 color = colours[0];  // Default to first color


    if (smoothNoise) {
        // Calculate band size
        float bandSize = 1.0 / float(numColours - 1);

        for (int i = 0; i < numColours - 1; i++) {
            float lowerBound = bandSize * float(i);
            float upperBound = bandSize * float(i + 1);

            if (noiseValue >= lowerBound && (noiseValue <= upperBound || i == numColours - 2)) {
                float t = (noiseValue - lowerBound) / bandSize;
                color = mix(colours[i], colours[i + 1], t);
                break;
            }
        }
    } else {
        // Calculate band size
        float bandSize = 1.0 / float(numColours);

        // Find the appropriate color based on noise value
        for (int i = 1; i < numColours; i++) {
            if (noiseValue > bandSize * float(i)) {
                color = colours[i];
            }
        }
    }

    gl_FragColor = color;
}