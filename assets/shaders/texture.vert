#version 120

varying vec4 vColor;    // Pass color to the fragment shader
varying vec2 vTexCoord; // Pass texture coordinates to the fragment shader

void main()
{
    // Pass the vertex position to the next stage
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

    // Pass the vertex color to the fragment shader
    vColor = gl_Color;

    // Pass the texture coordinates to the fragment shader
    vTexCoord = gl_MultiTexCoord0.xy;
}
