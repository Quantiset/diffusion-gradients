
shader_type canvas_item;

// r = A concentration
// g = B concentration
uniform sampler2D prev : filter_nearest, repeat_enable;
uniform sampler2D noise_tex;

uniform float r = 1.0;
uniform float s = 1.0;

uniform float dt = 1.0;

uniform float noise_scale = 0.0;

uniform float min_a = 0.0;
uniform float max_a = 9999.9;
uniform float min_b = 0.0;
uniform float max_b = 9999.9;

uniform float laplace_divisor = 1.0;
uniform float diff_factor = 0.005;

vec2 wrap(vec2 uv) {
	return vec2(
		mod(uv.x+1.0, 1.0), 
		mod(uv.y+1.0, 1.0)
	);
}

float hash(vec2 uv) {
    uvec2 v = uvec2(uv * 12345.678); // Scale UV to integer space
    v = v * 1664525u + 1013904223u;
    
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    
    return float(v.x) * (1.0 / 4294967296.0);
}

void fragment() {
	vec2 texel = TEXTURE_PIXEL_SIZE;
	
	float a = texture(prev, UV).r;
	float b = texture(prev, UV).g;
	
	vec2 top    = texture(prev, wrap(UV + vec2(0.0, -texel.y))).rg;
	vec2 right  = texture(prev, wrap(UV + vec2(texel.x, 0.0))).rg;
	vec2 bottom = texture(prev, wrap(UV + vec2(0.0, texel.y))).rg;
	vec2 left   = texture(prev, wrap(UV + vec2(-texel.x, 0.0))).rg;
	
	vec2 topright    = texture(prev, wrap(UV + vec2(texel.x, -texel.y))).rg;
	vec2 topleft     = texture(prev, wrap(UV + vec2(-texel.x, -texel.y))).rg;
	vec2 bottomright = texture(prev, wrap(UV + vec2(texel.x, texel.y))).rg;
	vec2 bottomleft  = texture(prev, wrap(UV + vec2(-texel.x, texel.y))).rg;
	
	vec4 col = texture(prev, UV);
	
	float laplace_a = (
		+ 1.0 * (left[0] + top[0] + right[0] + bottom[0])
		- 4.0 * a
	) / laplace_divisor;
	float laplace_b = (
		+ 1.0 * (left[1] + top[1] + right[1] + bottom[1])
		- 4.0 * b
	) / laplace_divisor;
	
	float noise = (hash(UV) * 0.1) - 0.05;
	//noise *= noise_scale;
	//noise = texture(noise_tex, UV).r * 0.01 - 0.005;
	
	float ra = diff_factor*(16.0 - a*b);
	float rb = diff_factor*(a*b - b - 12.0 - noise);
	
	float new_a = a + dt * (ra + r*s*laplace_a);
	float new_b = b + dt * (rb + s*laplace_b);
	
	new_a = max(0.0, new_a);
	new_b = max(0.0, new_b);
	new_a = clamp(new_a, min_a, max_a);
	new_b = clamp(new_b, min_b, max_b);
	
	COLOR = vec4(new_a, new_b, 0., 1.0);
}
