extends Node2D

var screen_size := Vector2i(1024, 1024)

var rd: RenderingDevice
var shader: RID
var pipeline: RID
var uniform_set_a: RID
var uniform_set_b: RID
var texture_a: RID
var texture_b: RID
var texture_rd: Texture2DRD

var use_a := true

@onready var output_sprite: Sprite2D = $Sprite2D

@export var run_name := "f055_k062"
@export_range(0,100,0.00001) var r := 4.0
@export_range(0,100,0.00001) var s := 0.0625
@export_range(0,100,0.00001) var dt := 0.9
@export var laplace_divisor := 1.0
@export_range(0,100,0.00001) var diff_factor := 0.005
@export var min_a := 0.0
@export var max_a := 999.9
@export var min_b := 0.0
@export var max_b := 999.9
@export var noise_scale := 1.0
@export var time_until_refresh := 0.1
@export var gen_map := false
@export var gen_map_refresh_each_tick := true

var gl_sum := 0.0
var gl_iters := 0
var gl_max := 0.0
var gl_avg := 0.0
var gl_recent_max := 0.0
var gl_det := 0.0

# For click interaction
var current_image_data: Image

func _ready():
	Engine.max_fps = 200
	
	if gen_map:
		_run_parameter_map()
	else:
		_initialize_compute()
		_init_simulation_state()
		ready_ui()

func _initialize_compute():
	rd = RenderingServer.get_rendering_device()
	
	# 1. Load compute shader
	var shader_file := load("res://shaders/rd_compute.glsl")
	var shader_spirv = shader_file.get_spirv()
	shader = rd.shader_create_from_spirv(shader_spirv)
	pipeline = rd.compute_pipeline_create(shader)

	var tf := RDTextureFormat.new()
	tf.width = screen_size.x
	tf.height = screen_size.y
	tf.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	tf.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
	)
	
	texture_a = rd.texture_create(tf, RDTextureView.new())
	texture_b = rd.texture_create(tf, RDTextureView.new())
	
	var u_a_read := RDUniform.new()
	u_a_read.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_a_read.binding = 0
	u_a_read.add_id(texture_a)
	
	var u_b_write := RDUniform.new()
	u_b_write.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_b_write.binding = 1
	u_b_write.add_id(texture_b)
	
	uniform_set_a = rd.uniform_set_create([u_a_read, u_b_write], shader, 0)
	
	var u_b_read := RDUniform.new()
	u_b_read.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_b_read.binding = 0
	u_b_read.add_id(texture_b)
	
	var u_a_write := RDUniform.new()
	u_a_write.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_a_write.binding = 1
	u_a_write.add_id(texture_a)
	
	uniform_set_b = rd.uniform_set_create([u_b_read, u_a_write], shader, 0)
	
	texture_rd = Texture2DRD.new()
	output_sprite.texture = texture_rd
	output_sprite.centered = false 

func _init_simulation_state():
	var img := Image.create(screen_size.x, screen_size.y, false, Image.FORMAT_RGBAF)
	
	for x in range(screen_size.x):
		for y in range(screen_size.y):
			img.set_pixel(x, y, Color(4.0, 4.0, 0.0, 1.0))
	
	var data := img.get_data()
	rd.texture_update(texture_a, 0, data)
	
	var empty_data = PackedByteArray()
	empty_data.resize(screen_size.x * screen_size.y * 16) 
	rd.texture_update(texture_b, 0, empty_data)

func ready_ui():
	return
	get_node("%FSlider").value_changed.connect(_on_f_slider_changed)
	get_node("%KSlider").value_changed.connect(_on_k_slider_changed)
	_update_sliders()

func _on_f_slider_changed(val):
	r = val
	get_node("%FLabel").text = str(r)

func _on_k_slider_changed(val):
	s = val
	get_node("%KLabel").text = str(s)

func _update_sliders():
	if has_node("%FSlider"): get_node("%FSlider").value = r
	if has_node("%KSlider"): get_node("%KSlider").value = s
	if has_node("%FLabel"): get_node("%FLabel").text = str(r)
	if has_node("%KLabel"): get_node("%KLabel").text = str(s)

var time_acc := 0.0

func _process(delta):
	
	if Input.is_action_just_pressed("ui_space"):
		_save_image()
	
	_dispatch_compute()
	
	use_a = !use_a
	
	texture_rd.texture_rd_rid = texture_b if use_a else texture_a
	
	time_acc += delta
	if time_acc > time_until_refresh:
		_analyze_and_display()
		time_acc = 0.0

func _dispatch_compute():
	var push_constants := PackedByteArray()
	
	push_constants.append_array(PackedFloat32Array([r, s, dt, laplace_divisor]).to_byte_array())
	push_constants.append_array(PackedFloat32Array([diff_factor, noise_scale, min_a, max_a]).to_byte_array())
	push_constants.append_array(PackedFloat32Array([min_b, max_b]).to_byte_array())
	push_constants.append_array(PackedInt32Array([screen_size.x, screen_size.y]).to_byte_array())
	
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	
	var current_set := uniform_set_a if use_a else uniform_set_b
	rd.compute_list_bind_uniform_set(compute_list, current_set, 0)
	rd.compute_list_set_push_constant(compute_list, push_constants, push_constants.size())
	
	var groups_x := (screen_size.x + 15) / 16
	var groups_y := (screen_size.y + 15) / 16
	rd.compute_list_dispatch(compute_list, groups_x, groups_y, 1)
	rd.compute_list_end()
	
	#if not Engine.is_editor_hint():
		#rd.submit()
	#else:
		#pass

func _analyze_and_display():
	var current_tex := texture_b if use_a else texture_a
	
	var data: PackedByteArray
	if not Engine.is_editor_hint():
		rd.sync() 
		data = rd.texture_get_data(current_tex, 0)
	else:
		return
	
	var img := Image.create_from_data(screen_size.x, screen_size.y, false, Image.FORMAT_RGBAF, data)
	current_image_data = img
	
	var min_b_t := 9999999.0
	var max_b_t := 0.0
	var contrib := 0.0
	var pxs := 0
	
	var edge_offset := 410
	
	for x in range(edge_offset, screen_size.x - edge_offset):
		for y in range(edge_offset, screen_size.y - edge_offset):
			var col := img.get_pixel(x, y)
			min_b_t = min(min_b_t, col.g)
			max_b_t = max(max_b_t, col.g)
			contrib += col.g
			pxs += 1
	
	if pxs > 0:
		gl_avg = contrib / pxs
		gl_sum += max_b_t - min_b_t
		gl_iters += 1
		gl_max = max(gl_max, max_b_t)
		gl_recent_max = max_b_t
		
		print("Min: ", min_b_t, " Max: ", max_b_t, " Avg: ", gl_avg)
	
	if output_sprite.material is ShaderMaterial:
		output_sprite.material.set_shader_parameter("max_b", max_b_t)
		output_sprite.material.set_shader_parameter("min_b", min_b_t)

func _click_region(click_pos: Vector2):
	var radius := 7
	
	var current_tex := texture_b if use_a else texture_a
	var data := rd.texture_get_data(current_tex, 0)
	var img := Image.create_from_data(screen_size.x, screen_size.y, false, Image.FORMAT_RGBAF, data)
	
	for x in range(int(click_pos.x - radius), int(click_pos.x + radius)):
		for y in range(int(click_pos.y - radius), int(click_pos.y + radius)):
			if x >= 0 and x < screen_size.x and y >= 0 and y < screen_size.y:
				var col := img.get_pixel(x, y)
				col.g = 1.00 # Inject B
				img.set_pixel(x, y, col)
	
	rd.texture_update(current_tex, 0, img.get_data())

func _save_image():
	if current_image_data:
		var na := "r%s_s%s" % [str(r).replace("0.", "."), str(s).replace("0.", ".")]
		var time_str := Time.get_datetime_string_from_system().replace(":", "-")
		current_image_data.save_png("res://saves/%s_%s.png" % [na, time_str])

func _run_parameter_map():
	pass

func _exit_tree():
	if rd:
		rd.free_rid(uniform_set_a)
		rd.free_rid(uniform_set_b)
		rd.free_rid(pipeline)
		rd.free_rid(shader)
		rd.free_rid(texture_a)
		rd.free_rid(texture_b)
