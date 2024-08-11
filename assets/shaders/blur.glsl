#version 330 core
in vec2 TexCoord;
out vec4 color;

uniform sampler2D texture;
uniform float radius;
uniform vec2 resolution;

void main() {
    /*vec4 color = vec4(0.0);
    float totalWeight = 0.0;

    for (float x = -radius; x <= radius; x += 1.0) {
        for (float y = -radius; y <= radius; y += 1.0) {
            float weight = exp(-(x * x + y * y) / (2.0 * radius * radius));
            vec2 offset = vec2(x, y) / resolution;
            color += texture2D(texture, TexCoord + offset) * weight;
            totalWeight += weight;
        }
    }

    color /= totalWeight;
    //color = vec4(0.0, 1.0, 0.0, 0.5);
    color.a = float(int(color.a * 255)) / 255.0;
    color.r = float(int(color.r * 255)) / 255.0;
    color.g = float(int(color.g * 255)) / 255.0;
    color.b = float(int(color.b * 255)) / 255.0;
    FragColor = color;*/
    color = texture2D(texture, TexCoord);
}