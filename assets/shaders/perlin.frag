#version 120

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

// Helper function to get permutation value
int getPermutation(int index) {
    int i = int(mod(float(index), 256.0));
    
    // First 32 values
    if (i < 32) {
        if (i == 0) return 151; if (i == 1) return 160; if (i == 2) return 137; if (i == 3) return 91;
        if (i == 4) return 90; if (i == 5) return 15; if (i == 6) return 131; if (i == 7) return 13;
        if (i == 8) return 201; if (i == 9) return 95; if (i == 10) return 96; if (i == 11) return 53;
        if (i == 12) return 194; if (i == 13) return 233; if (i == 14) return 7; if (i == 15) return 225;
        if (i == 16) return 140; if (i == 17) return 36; if (i == 18) return 103; if (i == 19) return 30;
        if (i == 20) return 69; if (i == 21) return 142; if (i == 22) return 8; if (i == 23) return 99;
        if (i == 24) return 37; if (i == 25) return 240; if (i == 26) return 21; if (i == 27) return 10;
        if (i == 28) return 23; if (i == 29) return 190; if (i == 30) return 6; if (i == 31) return 148;
    }
    // Next 32 values (32-63)
    else if (i < 64) {
        if (i == 32) return 247; if (i == 33) return 120; if (i == 34) return 234; if (i == 35) return 75;
        if (i == 36) return 0; if (i == 37) return 26; if (i == 38) return 197; if (i == 39) return 62;
        if (i == 40) return 94; if (i == 41) return 252; if (i == 42) return 219; if (i == 43) return 203;
        if (i == 44) return 117; if (i == 45) return 35; if (i == 46) return 11; if (i == 47) return 32;
        if (i == 48) return 57; if (i == 49) return 177; if (i == 50) return 33; if (i == 51) return 88;
        if (i == 52) return 237; if (i == 53) return 149; if (i == 54) return 56; if (i == 55) return 87;
        if (i == 56) return 174; if (i == 57) return 20; if (i == 58) return 125; if (i == 59) return 136;
        if (i == 60) return 171; if (i == 61) return 168; if (i == 62) return 68; if (i == 63) return 175;
    }
    // Next 32 values (64-95)
    else if (i < 96) {
        if (i == 64) return 74; if (i == 65) return 165; if (i == 66) return 71; if (i == 67) return 134;
        if (i == 68) return 139; if (i == 69) return 48; if (i == 70) return 27; if (i == 71) return 166;
        if (i == 72) return 77; if (i == 73) return 146; if (i == 74) return 158; if (i == 75) return 231;
        if (i == 76) return 83; if (i == 77) return 111; if (i == 78) return 229; if (i == 79) return 122;
        if (i == 80) return 60; if (i == 81) return 211; if (i == 82) return 133; if (i == 83) return 230;
        if (i == 84) return 220; if (i == 85) return 105; if (i == 86) return 92; if (i == 87) return 41;
        if (i == 88) return 55; if (i == 89) return 46; if (i == 90) return 245; if (i == 91) return 40;
        if (i == 92) return 244; if (i == 93) return 102; if (i == 94) return 143; if (i == 95) return 54;
    }
    // Next 32 values (96-127)
    else if (i < 128) {
        if (i == 96) return 65; if (i == 97) return 25; if (i == 98) return 63; if (i == 99) return 161;
        if (i == 100) return 1; if (i == 101) return 216; if (i == 102) return 80; if (i == 103) return 73;
        if (i == 104) return 209; if (i == 105) return 76; if (i == 106) return 132; if (i == 107) return 187;
        if (i == 108) return 208; if (i == 109) return 89; if (i == 110) return 18; if (i == 111) return 169;
        if (i == 112) return 200; if (i == 113) return 196; if (i == 114) return 135; if (i == 115) return 130;
        if (i == 116) return 116; if (i == 117) return 188; if (i == 118) return 159; if (i == 119) return 86;
        if (i == 120) return 164; if (i == 121) return 100; if (i == 122) return 109; if (i == 123) return 198;
        if (i == 124) return 173; if (i == 125) return 186; if (i == 126) return 3; if (i == 127) return 64;
    }
    // Next 32 values (128-159)
    else if (i < 160) {
        if (i == 128) return 52; if (i == 129) return 217; if (i == 130) return 226; if (i == 131) return 250;
        if (i == 132) return 124; if (i == 133) return 123; if (i == 134) return 5; if (i == 135) return 202;
        if (i == 136) return 38; if (i == 137) return 147; if (i == 138) return 118; if (i == 139) return 126;
        if (i == 140) return 255; if (i == 141) return 82; if (i == 142) return 85; if (i == 143) return 212;
        if (i == 144) return 207; if (i == 145) return 206; if (i == 146) return 59; if (i == 147) return 227;
        if (i == 148) return 47; if (i == 149) return 16; if (i == 150) return 58; if (i == 151) return 17;
        if (i == 152) return 182; if (i == 153) return 189; if (i == 154) return 28; if (i == 155) return 42;
        if (i == 156) return 223; if (i == 157) return 183; if (i == 158) return 170; if (i == 159) return 213;
    }
    // Next 32 values (160-191)
    else if (i < 192) {
        if (i == 160) return 119; if (i == 161) return 248; if (i == 162) return 152; if (i == 163) return 2;
        if (i == 164) return 44; if (i == 165) return 154; if (i == 166) return 163; if (i == 167) return 70;
        if (i == 168) return 221; if (i == 169) return 153; if (i == 170) return 101; if (i == 171) return 155;
        if (i == 172) return 167; if (i == 173) return 43; if (i == 174) return 172; if (i == 175) return 9;
        if (i == 176) return 129; if (i == 177) return 22; if (i == 178) return 39; if (i == 179) return 253;
        if (i == 180) return 19; if (i == 181) return 98; if (i == 182) return 108; if (i == 183) return 110;
        if (i == 184) return 79; if (i == 185) return 113; if (i == 186) return 224; if (i == 187) return 232;
        if (i == 188) return 178; if (i == 189) return 185; if (i == 190) return 112; if (i == 191) return 104;
    }
    // Next 32 values (192-223)
    else if (i < 224) {
        if (i == 192) return 218; if (i == 193) return 246; if (i == 194) return 97; if (i == 195) return 228;
        if (i == 196) return 251; if (i == 197) return 34; if (i == 198) return 242; if (i == 199) return 193;
        if (i == 200) return 238; if (i == 201) return 210; if (i == 202) return 144; if (i == 203) return 12;
        if (i == 204) return 191; if (i == 205) return 179; if (i == 206) return 162; if (i == 207) return 241;
        if (i == 208) return 81; if (i == 209) return 51; if (i == 210) return 145; if (i == 211) return 235;
        if (i == 212) return 249; if (i == 213) return 14; if (i == 214) return 239; if (i == 215) return 107;
        if (i == 216) return 49; if (i == 217) return 192; if (i == 218) return 214; if (i == 219) return 31;
        if (i == 220) return 181; if (i == 221) return 199; if (i == 222) return 106; if (i == 223) return 157;
    }
    // Last 32 values (224-255)
    else {
        if (i == 224) return 184; if (i == 225) return 84; if (i == 226) return 204; if (i == 227) return 176;
        if (i == 228) return 115; if (i == 229) return 121; if (i == 230) return 50; if (i == 231) return 45;
        if (i == 232) return 127; if (i == 233) return 4; if (i == 234) return 150; if (i == 235) return 254;
        if (i == 236) return 138; if (i == 237) return 236; if (i == 238) return 205; if (i == 239) return 93;
        if (i == 240) return 222; if (i == 241) return 114; if (i == 242) return 67; if (i == 243) return 29;
        if (i == 244) return 24; if (i == 245) return 72; if (i == 246) return 243; if (i == 247) return 141;
        if (i == 248) return 128; if (i == 249) return 195; if (i == 250) return 78; if (i == 251) return 66;
        if (i == 252) return 215; if (i == 253) return 61; if (i == 254) return 156; if (i == 255) return 180;
    }
    return 0;
}


float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float grad(int hash, float x, float y, float z) {
    int h = int(mod(float(hash), 16.0));
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    float signU = mod(float(h), 2.0) == 0.0 ? 1.0 : -1.0;
    float signV = mod(float(h), 4.0) < 2.0 ? 1.0 : -1.0;
    return signU * u + signV * v;
}

float noise3D(float x, float y, float z) {
    float _x = floor(x);
    float _y = floor(y);
    float _z = floor(z);

    int ix = int(mod(_x, 256.0));
    int iy = int(mod(_y, 256.0));
    int iz = int(mod(_z, 256.0));

    float fx = x - _x;
    float fy = y - _y;
    float fz = z - _z;

    float u = fade(fx);
    float v = fade(fy);
    float w = fade(fz);

    int A  = int(mod(float(getPermutation(int(mod(float(ix), 256.0))) + iy), 256.0));
    int B  = int(mod(float(getPermutation(int(mod(float(ix + 1), 256.0))) + iy), 256.0));
    int AA = int(mod(float(getPermutation(A) + iz), 256.0));
    int AB = int(mod(float(getPermutation(int(mod(float(A + 1), 256.0))) + iz), 256.0));
    int BA = int(mod(float(getPermutation(B) + iz), 256.0));
    int BB = int(mod(float(getPermutation(int(mod(float(B + 1), 256.0))) + iz), 256.0));

    float p0 = grad(getPermutation(AA), fx, fy, fz);
    float p1 = grad(getPermutation(BA), fx - 1.0, fy, fz);
    float p2 = grad(getPermutation(AB), fx, fy - 1.0, fz);
    float p3 = grad(getPermutation(BB), fx - 1.0, fy - 1.0, fz);
    float p4 = grad(getPermutation(AA + 1), fx, fy, fz - 1.0);
    float p5 = grad(getPermutation(BA + 1), fx - 1.0, fy, fz - 1.0);
    float p6 = grad(getPermutation(AB + 1), fx, fy - 1.0, fz - 1.0);
    float p7 = grad(getPermutation(BB + 1), fx - 1.0, fy - 1.0, fz - 1.0);

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