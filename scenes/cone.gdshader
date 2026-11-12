shader_type spatial;

// Maps a flat 2D pattern onto a (truncated) cone without distortion.
// A cone is developable: unrolled it is a flat sector of angle 2*pi*sin(alpha),
// so each surface point gets plane coords (rho*cos(phi), rho*sin(phi)) with
// rho = distance to apex along the surface, phi = angle * sin(alpha).
// Spots keep the same size and shape everywhere; the number of spots around the
// cone grows with radius (as in Jahanbakhsh & Milinkovitch 2027, Example 6).
// With top_radius = 0.4692 (bottom 1, height 3.14) the sector is exactly 60 deg,
// which matches hexagonal symmetry. Use Python/cone_unroll.py to get apex_uv / sector_rot.

uniform bool unroll = true;          // false = old cylindrical (u = angle) mapping
uniform float height = 3.14;
uniform float bottom_radius = 1.0;
uniform float top_radius = 0.4692;
uniform float tile_world = 8.0;      // world units covered by one texture tile
uniform vec2 apex_uv = vec2(0.5);
uniform float sector_rot = 0.0;
uniform mat2 lattice_fix = mat2(1.0);  // undoes lattice strain (from cone_unroll.py)
uniform float seam_blend = 0.08;     // radians of angle cross-faded at the seam
uniform sampler2D tex : source_color, repeat_enable, filter_linear_mipmap;
varying vec3 vert;

void vertex() {
	vert = VERTEX;
}

// Gradients are passed explicitly: the angle wraps at the seam, so automatic
// derivatives would spike there and pick the blurriest mip level (a grey line).
vec3 sample_unrolled(float rho, float phi, vec2 drho, vec2 dphi) {
	float a = sector_rot + phi;
	vec2 dir = lattice_fix * vec2(cos(a), sin(a));
	vec2 tan_dir = lattice_fix * vec2(-sin(a), cos(a));
	vec2 uv = apex_uv + rho / tile_world * dir;
	vec2 ddx = (dir * drho.x + tan_dir * rho * dphi.x) / tile_world;
	vec2 ddy = (dir * drho.y + tan_dir * rho * dphi.y) / tile_world;
	return textureGrad(tex, uv, ddx, ddy).rgb;
}

float wrap_pi(float x) {
	return x - 2.0 * PI * round(x / (2.0 * PI));
}

void fragment() {
	vec3 p = vert;
	float angle = atan(p.x, p.z);              // -PI..PI, seam at +-PI
	float v = (p.y + height / 2.0) / height;   // 0 bottom .. 1 top

	if (!unroll) {
		ALBEDO = texture(tex, vec2(angle / (2.0 * PI) + 0.5, v)).rgb;
	} else {
		float sin_a = (bottom_radius - top_radius) / length(vec2(bottom_radius - top_radius, height));
		float rho = mix(bottom_radius, top_radius, v) / sin_a;
		float phi = angle * sin_a;
		vec2 drho = vec2(dFdx(rho), dFdy(rho));
		vec2 dphi = vec2(wrap_pi(dFdx(angle)), wrap_pi(dFdy(angle))) * sin_a;
		vec3 c = sample_unrolled(rho, phi, drho, dphi);
		// cross-fade with the other side of the seam so it is continuous
		float w = smoothstep(PI - seam_blend, PI, abs(angle));
		vec3 c_other = sample_unrolled(rho, phi - sign(angle) * 2.0 * PI * sin_a, drho, dphi);
		ALBEDO = mix(c, c_other, 0.5 * w);
	}
}
