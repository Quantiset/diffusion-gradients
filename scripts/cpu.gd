extends Node2D

@onready var sprite := $Sprite2D

# Simulation parameters
var d_a := 1.0
var d_b := 0.5
var f := 0.055
var k := 0.062
var dt := 1.0

# Grid data
var width := 100
var height := 100

var concentrations_a: PackedFloat32Array
var concentrations_b: PackedFloat32Array
var next_a: PackedFloat32Array
var next_b: PackedFloat32Array

var image: Image
var texture: ImageTexture

func _ready():
	# Initialize arrays
	var size := width * height
	concentrations_a.resize(size)
	concentrations_b.resize(size)
	next_a.resize(size)
	next_b.resize(size)
	
	# Fill with initial conditions: A=1, B=0
	for i in range(size):
		concentrations_a[i] = 1.0
		concentrations_b[i] = 0.0
	
	# Add seed of B chemical in center
	var center_x := width / 2
	var center_y := height / 2
	for x in range(center_x - 5, center_x + 5):
		for y in range(center_y - 5, center_y + 5):
			if x >= 0 and x < width and y >= 0 and y < height:
				var idx := y * width + x
				concentrations_b[idx] = 1.0
	
	# Create image and texture
	image = Image.create(width, height, false, Image.FORMAT_RGB8)
	texture = ImageTexture.create_from_image(image)
	sprite.texture = texture
	
	update_image()

func _process(_delta):
	print(Engine.get_frames_per_second())
	simulate_step()
	update_image()

func simulate_step():
	# Calculate next state for all cells
	for y in range(height):
		for x in range(width):
			var idx := y * width + x
			var a := concentrations_a[idx]
			var b := concentrations_b[idx]
			
						# Get neighbor indices (with wrapping)
						# Flattened indices for cardinal neighbors (with wrap)
			var left   := y * width + ((x - 1 + width) % width)
			var right  := y * width + ((x + 1) % width)
			var top    := ((y - 1 + height) % height) * width + x
			var bottom := ((y + 1) % height) * width + x

			# Flattened indices for diagonal neighbors (with wrap)
			var top_left     := ((y - 1 + height) % height) * width + ((x - 1 + width) % width)
			var top_right    := ((y - 1 + height) % height) * width + ((x + 1) % width)
			var bottom_left  := ((y + 1) % height) * width + ((x - 1 + width) % width)
			var bottom_right := ((y + 1) % height) * width + ((x + 1) % width)

			# Center index
			var center := y * width + x

			# 9-point Laplacian with weights: center=-1, cardinal=0.2, diagonal=0.05
			var laplace_a := -1.0 * concentrations_a[center] + \
							 0.2 * (concentrations_a[left] + concentrations_a[right] + concentrations_a[top] + concentrations_a[bottom]) + \
							 0.05 * (concentrations_a[top_left] + concentrations_a[top_right] + concentrations_a[bottom_left] + concentrations_a[bottom_right])

			var laplace_b := -1.0 * concentrations_b[center] + \
							 0.2 * (concentrations_b[left] + concentrations_b[right] + concentrations_b[top] + concentrations_b[bottom]) + \
							 0.05 * (concentrations_b[top_left] + concentrations_b[top_right] + concentrations_b[bottom_left] + concentrations_b[bottom_right])
			
			# Gray-Scott reaction-diffusion
			var reaction := a * b * b
			var new_a := a + (d_a * laplace_a - reaction + f * (1.0 - a)) * dt
			var new_b := b + (d_b * laplace_b + reaction - (k + f) * b) * dt
			
			# Clamp values
			next_a[idx] = clampf(new_a, 0.0, 1.0)
			next_b[idx] = clampf(new_b, 0.0, 1.0)
	
	# Swap buffers
	var temp_a = concentrations_a
	concentrations_a = next_a
	next_a = temp_a
	
	var temp_b = concentrations_b
	concentrations_b = next_b
	next_b = temp_b

func update_image():
	# Convert concentrations to pixel colors
	for y in range(height):
		for x in range(width):
			var idx := y * width + x
			var b_val := concentrations_b[idx]
			
			# Visualize B concentration
			var gray := int(b_val * 255.0)
			image.set_pixel(x, y, Color8(gray, gray, gray))
	
	# Update texture
	texture.update(image)
