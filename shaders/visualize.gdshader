shader_type canvas_item;

uniform float max_b = 1.;
uniform float min_b = 0.;

uniform sampler2D from;

void fragment() {
	float b = texture(from, UV).g;
	float s = (b - min_b) / (max_b - min_b);
	COLOR.rgb = vec3(s, s, s);
}

//void light() {
//	// Called for every pixel for every light affecting the CanvasItem.
//	// Uncomment to replace the default light processing function with this one.
//}
