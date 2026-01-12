// formula derived from:  https://www.karlsims.com/rd.html
shader_type canvas_item;

// r = A concentration
// g = B concentration
uniform sampler2D prev : filter_nearest, repeat_enable;

uniform float r = 1.0;
uniform float s = 1.0;

uniform float dt = 1.0;

uniform float noise_scale = 0.0;

uniform float min_a = 0.0;
uniform float max_a = 9999.9;
uniform float min_b = 0.0;
uniform float max_b = 9999.9;

uniform float laplace_divisor = 6.0;

vec2 wrap(vec2 uv) {
	return vec2(
		mod(uv.x+1.0, 1.0), 
		mod(uv.y+1.0, 1.0)
	);
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
		+ 4.0 * (left[0] + top[0] + right[0] + bottom[0])
		+ 1.0 * (topleft[0] + topright[0] + bottomleft[0] + bottomright[0])
		- 20.0 * a
	) / laplace_divisor;
	float laplace_b =(
		+ 4.0 * (left[1] + top[1] + right[1] + bottom[1])
		+ 1.0 * (topleft[1] + topright[1] + bottomleft[1] + bottomright[1])
		- 20.0 * b
	) / laplace_divisor;
	
	float noise = (fract(sin(dot(UV*2300.0, vec2(16.9898,78.233))) * 43758.5453) - 0.5) * 0.01;
	noise *= noise_scale;
	
	float new_a = a + dt * (16.0 - a*b + r*s*laplace_a);
	float new_b = b + dt * (a*b - b - 12.0 + s*laplace_b);
	
	new_a = clamp(new_a, min_a, max_a);
	new_b = clamp(new_b, min_b, max_b);
	
	COLOR = vec4(new_a, new_b, 0., 1.0);
}
