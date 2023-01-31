#version 450

layout (location = 0) out vec3 fragColor;

int numVertices = 4;

// 5 vertices to do a losange
vec2 positions[4] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.0),
    vec2(0.0, 0.5),
    vec2(-0.5, 0.0)
);

// use 4 colors to do a losange
vec3 colors[4] = vec3[](
    vec3(0.968, 0.811, 0.274),
    vec3(0.901, 0.196, 0.141),
    vec3(0.968, 0.811, 0.274),
    vec3(0.901, 0.196, 0.141)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex % numVertices], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex % numVertices];
}