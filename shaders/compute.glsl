#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba32f, set = 0, binding = 0) uniform readonly image2D input_tex;
layout(rgba32f, set = 0, binding = 1) uniform writeonly image2D output_tex;

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(input_tex);
	
	if (pos.x >= size.x || pos.y >= size.y) return;
	
	vec4 center = imageLoad(input_tex, pos);
	float a = center.r;
	float b = center.g;
	
	// Load neighbors with wrapping
	vec4 left   = imageLoad(input_tex, ivec2((pos.x - 1 + size.x) % size.x, pos.y));
	vec4 right  = imageLoad(input_tex, ivec2((pos.x + 1) % size.x, pos.y));
	vec4 top    = imageLoad(input_tex, ivec2(pos.x, (pos.y - 1 + size.y) % size.y));
	vec4 bottom = imageLoad(input_tex, ivec2(pos.x, (pos.y + 1) % size.y));
	
	// Laplacian
	float laplace_a = -1.0 * a + 0.25 * (left.r + right.r + top.r + bottom.r);
	float laplace_b = -1.0 * b + 0.25 * (left.g + right.g + top.g + bottom.g);
	
	// Gray-Scott
	float d_a = 1.0;
	float d_b = 0.5;
	float f = 0.055;
	float k = 0.062;
	float dt = 0.5;
	
	float reaction = a * b * b;
	float new_a = a + (d_a * laplace_a - reaction + f * (1.0 - a)) * dt;
	float new_b = b + (d_b * laplace_b + reaction - (k + f) * b) * dt;
	
	new_a = clamp(new_a, 0.0, 1.0);
	new new_b = clamp(new_b, 0.0, 1.0);
	
	imageStore(output_tex, pos, vec4(new_a, new_b, 0.0, 1.0));
}