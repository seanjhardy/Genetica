#version 130

uniform sampler2D texture;
uniform vec2 resolution;

// Generation params
uniform float time;

float twist = 0.5;
float waveFreq = 2.0;
float waveHeight = 2.0;
float baseSize = 2.0;

//source: https://iquilezles.org/articles/distfunctions2d
float sdBox( in vec2 uv, in vec2 boxSize )
{
    vec2 d = abs(uv) - boxSize;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float square(vec2 uv, vec2 center, float size)
{
    float pixelSize = fwidth(uv.y);
    float sdf = sdBox(uv - center, vec2(size - pixelSize * 0.5));
    return smoothstep(pixelSize, -pixelSize, sdf);
}

vec3 backboneColour(float depth) {
    return mix(vec3(0.1, 0.0, 0.0), vec3(0.3, 0.0, 0.0), depth);
}

vec3 baseColour(float depth) {
    vec3 colour = vec3(0.0, 0.3, 0.3);
    return mix(colour*0.2, colour, depth);
}

float depthToBlur(float depth)
{
    return mix(10.0, 10.0, depth);
}

void main() {
    vec2 uv = gl_FragCoord.xy;

    vec2 startUV = gl_FragCoord.xy / resolution;
    startUV.y -= 0.5;
    float angle = 0.1;
    startUV *= mat2(cos(angle), sin(angle), -sin(angle), cos(angle));

    vec2 baseUV = startUV;
    baseUV *= 20.0;
    float pixelSize = fwidth(baseUV.x);

    float uvPos = floor(baseUV.x) * twist;
    vec2 dnaUV = vec2(fract(baseUV.x) - 0.5, baseUV.y);

    float sinDNA = sin(uvPos + time);
    float cosDNA = cos(uvPos + time);
    float wave = sin(startUV.x * waveFreq)*waveHeight;

    float dnaPos1 = sinDNA * 2.0;
    float dnaPos2 = -sinDNA * 2.0;

    float dnaDepth1 = cosDNA * 0.5 + 0.5;
    float dnaDepth2 = -cosDNA * 0.5 + 0.5;

    float depthLine = mix(dnaDepth1, dnaDepth2, smoothstep(dnaPos1, dnaPos2, dnaUV.y));

    float lineBlur = depthToBlur(depthLine);
    float lineAlpha = mod(dnaUV.x + 0.3, 1.0) < 0.4 ? 1.0 : 0.0;
    lineAlpha *= 1.0 - step(abs(dnaPos1), abs(dnaUV.y - wave));

    float square1Alpha = square(dnaUV,
                                vec2(0.0, dnaPos1 + wave),
                                dnaDepth1 * 0.2 + 0.5);
    float square2Alpha = square(dnaUV,
                                vec2(0.0, dnaPos2+ wave),
                                dnaDepth2 * 0.2 + 0.5);

    vec3 colour = vec3(0.0, 0.0, 0.0);

    int baseIndex = int(floor(startUV.x * 20.0));

    if (dnaDepth1 > dnaDepth2)
    {
        colour = mix(colour, baseColour(depthLine), lineAlpha);
        colour = mix(colour, backboneColour(dnaDepth2), square2Alpha);
        colour = mix(colour, backboneColour(dnaDepth1), square1Alpha);
    }
    else
    {
        colour = mix(colour, baseColour(depthLine), lineAlpha);
        colour = mix(colour, backboneColour(dnaDepth1), square1Alpha);
        colour = mix(colour, backboneColour(dnaDepth2), square2Alpha);
    }

    if (colour == vec3(0.0, 0.0, 0.0)) {
        discard;
    }
    gl_FragColor = vec4(colour, 1.0);
}