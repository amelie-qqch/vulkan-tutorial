#version 450

layout (location = 0) out vec3 fragColor;


// 5 vertices to do a losange
vec2 positions[4] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.0),
    vec2(0.0, 0.5),
    vec2(-0.5, 0.0)
);

// use 4 colors to do a losange
vec3 colors[4] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 1.0, 0.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex%4], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex%4];
}