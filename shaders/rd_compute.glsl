#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba32f) restrict readonly uniform image2D img_prev;
layout(set = 0, binding = 1, rgba32f) restrict writeonly uniform image2D img_next;

layout(push_constant, std430) uniform Params {
    float r;
    float s;
    float dt;
    float laplace_divisor;
    float diff_factor;
    float noise_scale;
    float min_a;
    float max_a;
    float min_b;
    float max_b;
    int screen_width;
    int screen_height;
} params;

ivec2 wrap_coord(ivec2 coord, ivec2 size) {
    return ivec2((coord.x + size.x) % size.x, (coord.y + size.y) % size.y);
}

// Fixed: Corrected bitwise operations using uints
float hash(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    return float(v.x) * (1.0 / 4294967296.0);
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(params.screen_width, params.screen_height);
    
    if (coord.x >= size.x || coord.y >= size.y) return;
    
    vec4 center = imageLoad(img_prev, coord);
    float a = center.r;
    float b = center.g;
    
    ivec2 left   = wrap_coord(coord + ivec2(-1,  0), size);
    ivec2 right  = wrap_coord(coord + ivec2( 1,  0), size);
    ivec2 top    = wrap_coord(coord + ivec2( 0, -1), size);
    ivec2 bottom = wrap_coord(coord + ivec2( 0,  1), size);
    
    vec2 val_left   = imageLoad(img_prev, left).rg;
    vec2 val_right  = imageLoad(img_prev, right).rg;
    vec2 val_top    = imageLoad(img_prev, top).rg;
    vec2 val_bottom = imageLoad(img_prev, bottom).rg;
    
    float laplace_a = (val_left.x + val_right.x + val_top.x + val_bottom.x - 4.0 * a) / params.laplace_divisor;
    float laplace_b = (val_left.y + val_right.y + val_top.y + val_bottom.y - 4.0 * b) / params.laplace_divisor;
    
    float noise = (hash(uvec2(coord)) * 0.1) - 0.05;
    
    float ra = params.diff_factor * (16.0 - a * b);
    float rb = params.diff_factor * (a * b - b - 12.0 - noise);
    
    float new_a = a + params.dt * (ra + params.r * params.s * laplace_a);
    float new_b = b + params.dt * (rb + params.s * laplace_b);
    
    new_a = clamp(new_a, params.min_a, params.max_a);
    new_b = clamp(new_b, params.min_b, params.max_b);
    
    imageStore(img_next, coord, vec4(new_a, new_b, 0.0, 1.0));
}