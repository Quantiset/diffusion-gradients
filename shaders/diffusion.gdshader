// formula derived from:  https://www.karlsims.com/rd.html
shader_type canvas_item;

// r = A concentration
// g = B concentration
uniform sampler2D prev : filter_nearest, repeat_enable;

uniform float d_a = 1.0;
uniform float d_b = 0.5;

uniform float f = 0.055;
uniform float k = 0.062;

uniform float dt = 1.0;

uniform float noise_scale = 0.0;

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
	
	float laplace_a =
		+ 0.20 * (left[0] + top[0] + right[0] + bottom[0])
		+ 0.05 * (topleft[0] + topright[0] + bottomleft[0] + bottomright[0])
		- 1.00 * a;

	float laplace_b =
		+ 0.20 * (left[1] + top[1] + right[1] + bottom[1])
		+ 0.05 * (topleft[1] + topright[1] + bottomleft[1] + bottomright[1])
		- 1.00 * b;
	
	float noise = (fract(sin(dot(UV*2300.0, vec2(16.9898,78.233))) * 43758.5453) - 0.5) * 0.001;
	noise *= noise_scale;
	
	float reaction = a * b * b;
	float new_a = a + (d_a * laplace_a - reaction + f * (1.0 - a)) * dt;
	float new_b = b + (d_b * laplace_b + reaction - (k + f + noise) * b) * dt;
	
	new_a = clamp(new_a, 0.0, 1.0);
	new_b = clamp(new_b, 0.0, 1.0);
	
	COLOR = vec4(new_a, new_b, 0., 1.0);
}
