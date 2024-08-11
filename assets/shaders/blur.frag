#version 120
uniform sampler2D texture;
uniform vec2 resolution;
uniform float radius;

void main() {
    vec2 texcoord = gl_TexCoord[0].st;
    vec4 color = vec4(0.0);
    float totalWeight = 0.0;

    for (float x = -radius; x <= radius; x += 1.0) {
        for (float y = -radius; y <= radius; y += 1.0) {
            float weight = exp(-(x * x + y * y) / (2.0 * radius * radius));
            vec2 offset = vec2(x, y) / resolution;
            color += texture2D(texture, texcoord + offset) * weight;
            totalWeight += weight;
        }
    }

    gl_FragColor = color / totalWeight;
}