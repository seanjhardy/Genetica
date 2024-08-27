#version 120

uniform sampler2D texture;
varying vec4 vColor;     // Received color from the vertex shader
varying vec2 vTexCoord;  // Received texture coordinates from the vertex shader

void main()
{
    // Sample the texture at the given coordinates
    vec4 textureColor = texture2D(texture, vTexCoord);

    vec3 normalizedTextureColor = textureColor.rgb * 2.0 - 1.0;
    vec3 finalColor = vColor.rgb + vColor.rgb * normalizedTextureColor * textureColor.a;
    gl_FragColor = vec4(finalColor, vColor.a);
}
