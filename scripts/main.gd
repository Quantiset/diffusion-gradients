extends Node2D
#https://mgmalheiros.github.io/research/leopard/leopard-2020-preprint.pdf

var screen_size := Vector2i(1028, 1028)

var use_a := false

@onready var output_sprite : Sprite2D = $Sprite2D
@onready var viewport1 : SubViewport = $SubViewport1
@onready var viewport2 : SubViewport = $SubViewport2
@onready var viewports = [viewport1, viewport2]

@export var run_name := "f055_k062"
@export var r := 4.0
@export var s := 4.0
@export var dt := 0.005
@export var laplace_divisor := 4.0
@export var diff_factor := 0.005
@export var min_a := 0.0
@export var max_a := 999.9
@export var min_b := 0.0
@export var max_b := 999.9

@export var noise_scale := 1.0
@export var noise_gen_scale := 1.0

@export var time_until_refresh := 0.1
@export var use_adv_setup := true

@export var gen_map := false

@export var gen_fourier_transform := false

var r_vals := [1.0, 40.0, 0.5]
var s_vals := [1.0, 6.0, 0.5]

var gl_sum := 0.0
var gl_iters := 0
var gl_max := 0.0
var gl_recent_max := 0.0
var gl_det := 0.0
var things

func _ready():
	Engine.max_fps = 300
	
	if gen_map:
		
		var time_str = Time.get_datetime_string_from_system().replace(":", "-")
		var file := FileAccess.open("res://saves/" + time_str + ".txt", FileAccess.WRITE_READ)
		
		var r_t : float = r_vals[0]
		while r_t <  r_vals[1]:
			r_t += r_vals[2]
			var s_t : float = s_vals[0]
			while s_t < s_vals[1]:
				s_t += s_vals[2]
				
				r = r_t
				s = s_t
				
				gl_recent_max = 0
				gl_iters = 0
				gl_max = 0
				gl_sum = 0
				gl_det = 0
				
				print("===== iter started =====")
				print(r, " ", s)
				
				ready(false)
				await get_tree().create_timer(3.2).timeout
				
				print(r, " ", s)
				file.seek_end()
				file.store_string(str(r)+" "+str(s)+" "+str(gl_iters)+" "+str(gl_max)+" "+
								  str(gl_sum)+" "+str(gl_recent_max)+" "+str(gl_det)+"\n")
		file.close()
	else:
		ready()

func ready(update_bars := true):
	get_node("%FSlider").text = str(r)
	get_node("%KSlider").text = str(s)
	if update_bars:
		_on_f_slider_drag_ended(r)
		_on_k_slider_drag_ended(s)
	for viewport in viewports:
		viewport.size = screen_size
		viewport.get_node("Sprite2D").material.set_shader_parameter("r", r)
		viewport.get_node("Sprite2D").material.set_shader_parameter("s", s)
		viewport.get_node("Sprite2D").material.set_shader_parameter("dt", dt)
		viewport.get_node("Sprite2D").material.set_shader_parameter("laplace_divisor", laplace_divisor)
		viewport.get_node("Sprite2D").material.set_shader_parameter("diff_factor", diff_factor)
		viewport.get_node("Sprite2D").material.set_shader_parameter("min_a", min_a)
		viewport.get_node("Sprite2D").material.set_shader_parameter("max_a", max_a)
		viewport.get_node("Sprite2D").material.set_shader_parameter("min_b", min_b)
		viewport.get_node("Sprite2D").material.set_shader_parameter("max_b", max_b)
		viewport.get_node("Sprite2D").material.set_shader_parameter("noise_scale", noise_scale)
	
	var img := Image.create(screen_size.x, screen_size.y, false, Image.FORMAT_RGBAF)
	
	var noise : FastNoiseLite = $NoisePattern.texture.noise
	
	if use_adv_setup:
		
		for x in range(0, screen_size.x):
			for y in range(0, screen_size.y):
				var n := 0.0
				if int((float(y) / screen_size.y) * 20) == 10:
					n = 1.0
				#img.set_pixel(x, y, Color(4, 4+n*(1+randf()*0.5), 0, 1)) 
				img.set_pixel(x, y, Color(4, 4+randf()*0.01, 0, 1)) 
		
		
	else:
		
		for x in range(0, screen_size.x):
			for y in range(0, screen_size.y):
				var n = (noise.get_noise_2d(x, y) + 1) * 0.4
				img.set_pixel(x, y, Color( 4 , 4 + n , 0, 1)) 
		
		#var mid_size = 5
		#for x in range(screen_size.x/2 - mid_size, screen_size.x/2 + mid_size):
			#for y in range(screen_size.y/2 - mid_size, screen_size.y/2 + mid_size):
				#img.set_pixel(x, y, Color( 0 , 0 , 0, 1)) 
	
	var new_tex := ImageTexture.create_from_image(img)
	viewport1.get_node("Sprite2D").material.set_shader_parameter("prev", new_tex)
	viewport2.get_node("Sprite2D").material.set_shader_parameter("prev", new_tex)
	
	viewport1.canvas_item_default_texture_filter = viewport1.DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST
	viewport2.canvas_item_default_texture_filter = viewport2.DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST
	var tex: ViewportTexture = viewport1.get_texture()
	output_sprite.texture = tex
	
	await get_tree().create_timer(0.5).timeout
	
	update()

var time := 0.0
var frames := 0
func _process(delta):
	$CanvasLayer/Label.text = str(Engine.get_frames_per_second())
	frames += 1
	time += delta
	
	if Input.is_action_just_pressed("ui_click"):
		click()
		update()
	
	if Input.is_action_just_pressed("ui_space"):
		save()
	
	if time > time_until_refresh:
		display()
		time = 0.0

func display():
	var from = viewport1 if use_a else viewport2
	var tex = from.get_texture()
	var image = tex.get_image()
	var min_b_t = 9999999.0;
	var max_b_t = 0.0;
	
	var edge_offset = 410;
	var pxs := 0;
	var contrib := 0.0;
	
	for x in range(edge_offset, screen_size.x-edge_offset):
		for y in range(edge_offset, screen_size.y-edge_offset):
			var col = image.get_pixel(x, y)
			min_b_t = min(min_b_t, col.g)
			max_b_t = max(max_b_t, col.g)
			pxs += 1
			contrib += col.g;
	
	print("Min: ", min_b_t, " Max: ", max_b_t, " Avg: ", contrib/pxs)
	
	gl_sum += max_b_t - min_b_t
	gl_iters += 1
	gl_max = max(gl_max, max_b_t)
	gl_recent_max = max_b_t
	output_sprite.material.set_shader_parameter("max_b", max_b_t)
	output_sprite.material.set_shader_parameter("min_b", min_b_t)
	
	if gen_fourier_transform:
		for x in range(screen_size.x):
			for y in range(screen_size.y):
				var value := 0.0
				for i in range(screen_size.x):
					for j in range(screen_size.y):
						var angle: float = -2*PI*(x*i/screen_size.x + y*j/screen_size.y)
						var real = image.get_pixel(i, j).get_luminance() * sin(angle)
						var im = image.get_pixel(i, j).get_luminance() * cos(angle)
						value += real
				if randi() % 200 == 1:
					print(x, " ", y, " ", value)
				#image.set_pixel(x, y, Color(value, value, value))
	var imgs = ImageTexture.create_from_image(image)
	output_sprite.material.set_shader_parameter("from", imgs)

func update():
	proc()
	proc()

func proc():
	var from = viewport1 if use_a else viewport2
	var to   = viewport2 if use_a else viewport1
	
	to.get_node("Sprite2D").material.set_shader_parameter("prev", from.get_texture())
	
	use_a = !use_a

func click():
	var click_pos = get_global_mouse_position()
	var from = viewport1 if use_a else viewport2
	var to = viewport2 if use_a else viewport1
	var img := from.get_texture().get_image()
	
	var radius = 7
	for x in range(int(click_pos.x - radius), int(click_pos.x + radius)):
		for y in range(int(click_pos.y - radius), int(click_pos.y + radius)):
			if x >= 0 and x < screen_size.x and y >= 0 and y < screen_size.y:
				var col = img.get_pixel(x, y)
				col.g = 1.00
				img.set_pixel(x, y, col)
	
	var tex = ImageTexture.create_from_image(img)
	to.get_node("Sprite2D").material.set_shader_parameter("prev", tex)
	from.get_node("Sprite2D").material.set_shader_parameter("prev", tex)

func save():
	var na = "r" + str(r).trim_prefix("0.") + "_s" + str(s).trim_prefix("0.")
	$Sprite2D.texture.get_image().save_png("res://saves/"+na+".png")


func _on_f_slider_drag_ended(value_changed):
	var new_r := float(get_node("%FSlider").text)
	get_node("%FLabel").text = str(new_r)
	r = new_r
	for viewport in viewports: 
		viewport.get_node("Sprite2D").material.set_shader_parameter("r", r)

func _on_k_slider_drag_ended(value_changed):
	var new_s := float(get_node("%KSlider").text)
	get_node("%KLabel").text = str(new_s)
	s = new_s
	for viewport in viewports: 
		viewport.get_node("Sprite2D").material.set_shader_parameter("s", s)
