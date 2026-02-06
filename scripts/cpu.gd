extends Node2D

# Constants from MATLAB
const N = 256
const S_BASE = 0.005
const DA = 0.25
const DB = 0.0625
const DT = 0.9

# Simulation Arrays (64-bit for precision)
var a: PackedFloat64Array
var b: PackedFloat64Array
var a2: PackedFloat64Array
var b2: PackedFloat64Array
var beta: PackedFloat64Array

@onready var sprite = $Sprite2D

func _ready():
	# Initialize arrays
	a.resize(N * N)
	b.resize(N * N)
	a2.resize(N * N)
	b2.resize(N * N)
	beta.resize(N * N)
	
	for i in range(N * N):
		a[i] = 4.0
		b[i] = 4.0
		a2[i] = 4.0
		b2[i] = 4.0
		beta[i] = 12.0 + (randf() * 0.1 - 0.05)
	
	sprite.texture = ImageTexture.create_from_image(Image.create(N, N, false, Image.FORMAT_RGB8))
	sprite.scale = Vector2(2, 2)

func _process(_delta):
	for step in range(10):
		simulate_step()
	update_visualization()

func simulate_step():
	var next_a = a.duplicate()
	var next_b = b.duplicate()
	var next_a2 = a2.duplicate()
	var next_b2 = b2.duplicate()
	
	var min_a = 100.0
	var max_a = -100.0
	for val in a:
		if val < min_a: min_a = val
		if val > max_a: max_a = val
	
	var range_a = max_a - min_a
	if range_a < 0.0001: range_a = 1.0 
	
	for y in range(N):
		for x in range(N):
			var i = y * N + x
			
			var l = y * N + posmod(x - 1, N)
			var r = y * N + posmod(x + 1, N)
			var u = posmod(y - 1, N) * N + x
			var d = posmod(y + 1, N) * N + x
			
			var lap_a = a[l] + a[r] + a[u] + a[d] - 4.0 * a[i]
			var lap_b = b[l] + b[r] + b[u] + b[d] - 4.0 * b[i]
			var lap_a2 = a2[l] + a2[r] + a2[u] + a2[d] - 4.0 * a2[i]
			var lap_b2 = b2[l] + b2[r] + b2[u] + b2[d] - 4.0 * b2[i]
			
			var delA = S_BASE * (16.0 - a[i] * b[i]) + DA * lap_a
			var delB = S_BASE * (a[i] * b[i] - b[i] - beta[i]) + DB * lap_b
			
			next_a[i] = max(0.0, a[i] + DT * delA)
			next_b[i] = max(0.0, b[i] + DT * delB)
			
			var ref = (a[i] - min_a) / range_a
			var s_coupled = S_BASE * (4.0 * ref)
			
			var delA2 = s_coupled * (16.0 - a2[i] * b2[i]) + DA * lap_a2
			var delB2 = s_coupled * (a2[i] * b2[i] - b2[i] - beta[i]) + DB * lap_b2
			
			next_a2[i] = max(0.0, a2[i] + DT * delA2)
			next_b2[i] = max(0.0, b2[i] + DT * delB2)

	a = next_a
	b = next_b
	a2 = next_a2
	b2 = next_b2

func update_visualization():
	var img = Image.create(N, N, false, Image.FORMAT_RGB8)
	
	var min_a = 100.0
	var max_a = -100.0
	for val in a:
		min_a = min(min_a, val)
		max_a = max(max_a, val)
	
	var range_a = max_a - min_a
	
	for y in range(N):
		for x in range(N):
			var val = a[y * N + x]
			var norm = 0.0
			if range_a > 0.0001:
				norm = (val - min_a) / range_a
			
			var r = 1.0 - norm
			var g = 1.0 - norm
			var blue = 1.0
			img.set_pixel(x, y, Color(r, g, blue))
			
	sprite.texture.update(img)
