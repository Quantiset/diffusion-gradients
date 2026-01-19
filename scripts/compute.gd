extends Node2D

var rd: RenderingDevice
var shader: RID
var pipeline: RID
var texture_a: RID
var texture_b: RID
var uniform_set_a: RID
var uniform_set_b: RID
var width := 1028
var height := 1028
var use_a := true

func _ready():
	rd = RenderingServer.create_local_rendering_device()
	
	# Load compute shader
	var shader_file := load("res://shaders/compute.glsl")
	var shader_spirv: RDShaderSPIRV = shader_file.get_spirv()
	shader = rd.shader_create_from_spirv(shader_spirv)
	pipeline = rd.compute_pipeline_create(shader)
	
	# Create textures
	var fmt := RDTextureFormat.new()
	fmt.width = width
	fmt.height = height
	fmt.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | \
					 RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT | \
					 RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
	
	# Initialize data
	var img := Image.create(width, height, false, Image.FORMAT_RGBAF)
	img.fill(Color(1, 0, 0, 1))
	for x in range(width/2 - 5, width/2 + 5):
		for y in range(height/2 - 5, height/2 + 5):
			img.set_pixel(x, y, Color(1, 1, 0, 1))
	
	var data := img.get_data()
	texture_a = rd.texture_create(fmt, RDTextureView.new(), [data])
	texture_b = rd.texture_create(fmt, RDTextureView.new(), [data])
	
	# Create uniform sets (bind textures to shader)
	_create_uniform_sets()

func _create_uniform_sets():
	# Uniform set A reads from texture_a, writes to texture_b
	var uniform_a := RDUniform.new()
	uniform_a.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	uniform_a.binding = 0
	uniform_a.add_id(texture_a)
	
	var uniform_b_write := RDUniform.new()
	uniform_b_write.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	uniform_b_write.binding = 1
	uniform_b_write.add_id(texture_b)
	
	uniform_set_a = rd.uniform_set_create([uniform_a, uniform_b_write], shader, 0)
	
	# Uniform set B reads from texture_b, writes to texture_a
	var uniform_b := RDUniform.new()
	uniform_b.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	uniform_b.binding = 0
	uniform_b.add_id(texture_b)
	
	var uniform_a_write := RDUniform.new()
	uniform_a_write.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	uniform_a_write.binding = 1
	uniform_a_write.add_id(texture_a)
	
	uniform_set_b = rd.uniform_set_create([uniform_b, uniform_a_write], shader, 0)

func _process(_delta):
	# Dispatch compute shader
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	
	var uniform_set = uniform_set_a if use_a else uniform_set_b
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	
	# Dispatch work groups (8x8 threads per group)
	rd.compute_list_dispatch(compute_list, ceili(width / 8.0), ceili(height / 8.0), 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	use_a = !use_a
	
	# Get result and display
	var output_texture = texture_b if use_a else texture_a
	var byte_data := rd.texture_get_data(output_texture, 0)
	var img := Image.create_from_data(width, height, false, Image.FORMAT_RGBAF, byte_data)
	
	$Sprite2D.texture = ImageTexture.create_from_image(img)
