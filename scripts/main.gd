extends Node2D
#https://mgmalheiros.github.io/research/leopard/leopard-2020-preprint.pdf

var screen_size := Vector2i(1024, 1024)

var use_a := false

@onready var output_sprite : Sprite2D = $Sprite2D
@onready var viewport1 : SubViewport = $SubViewport1
@onready var viewport2 : SubViewport = $SubViewport2
@onready var viewports = [viewport1, viewport2]

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

@export var gen_fourier_transform := false

var r_vals := [5.0, 20.0, 0.5]
var s_vals := [1.0, 12.0, 0.5]

var gl_sum := 0.0
var gl_iters := 0
var gl_max := 0.0
var gl_avg := 0.0
var gl_recent_max := 0.0
var gl_det := 0.0

var image # temporary place to cache latest image frame

func _ready():
	
	Engine.max_fps = 200
	
	if gen_map:
		
		var time_str = Time.get_datetime_string_from_system().replace(":", "-")
		var file := FileAccess.open("res://saves/" + time_str + ".txt", FileAccess.WRITE_READ)
		
		var s_t : float = s_vals[0]
		while s_t <  s_vals[1]:
			s_t += s_vals[2]
			var regen_map := true
			var r_t : float = r_vals[0]
			while r_t < r_vals[1]:
				r_t += r_vals[2]
				
				r = r_t
				s = s_t
				
				gl_recent_max = 0
				gl_iters = 0
				gl_max = 0
				gl_sum = 0
				gl_det = 0
				gl_avg = 0
				
				print("===== iter started =====")
				print(r, " ", s)
				
				ready(false, regen_map or gen_map_refresh_each_tick)
				await get_tree().create_timer(1.7 + 2.0 * int(regen_map and gen_map_refresh_each_tick)).timeout
				
				print(r, " ", s)
				file.seek_end()
				file.store_string(str(r)+" "+str(s)+" "+str(gl_iters)+" "+str(gl_max)+" "+
								str(gl_sum)+" "+str(gl_recent_max)+" "+str(gl_det)+" "+
								str(gl_avg)+" "+str(Anisotropy.describe(image)) + "\n")
				regen_map = false
		file.close()
	else:
		ready()

func ready(update_bars := true, refresh_map := true):
	get_node("%FSlider").text = str(r)
	get_node("%KSlider").text = str(s)
	if update_bars:
		_on_f_slider_drag_ended(r)
		_on_k_slider_drag_ended(s)
	print(r, " ", s)
	
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
	
	if refresh_map:
		var img := Image.create(screen_size.x, screen_size.y, false, Image.FORMAT_RGBAF)
		
		var noise : FastNoiseLite = $NoisePattern.texture.noise
		var blue_noise: Image = preload("res://assets/obluenoise256.png").get_image()
		
		for x in range(0, screen_size.x):
			for y in range(0, screen_size.y):
				var n := 0.0
				if int((float(y) / screen_size.y) * 20) == 10:
					n = 1.0
				#img.set_pixel(x, y, Color(4, 4+n*(1+randf()*0.5), 0, 1)) 
				#img.set_pixel(x, y, Color(4, 4+blue_noise.get_pixel(x, y).r*1.0, 0, 1))
				img.set_pixel(x, y, Color(4, 4, 0, 1)) 
			
		
		var new_tex := ImageTexture.create_from_image(img)
		viewport1.get_node("Sprite2D").material.set_shader_parameter("prev", new_tex)
		viewport2.get_node("Sprite2D").material.set_shader_parameter("prev", new_tex)
		
		viewport1.canvas_item_default_texture_filter = viewport1.DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST
		viewport2.canvas_item_default_texture_filter = viewport2.DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST
		var tex: ViewportTexture = viewport1.get_texture()
		output_sprite.texture = tex
		
		await get_tree().create_timer(0.2).timeout
	
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
	image = tex.get_image()
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
			#print(0.005*(16.0-col.r*col.g), " ", 0.005*(col.r*col.g-col.g-12.0))
			pxs += 1
			contrib += col.g;
	
	print("Min: ", min_b_t, " Max: ", max_b_t, " Avg: ", contrib/pxs)
	
	gl_avg = contrib/pxs
	gl_sum += max_b_t - min_b_t
	gl_iters += 1
	gl_max = max(gl_max, max_b_t)
	gl_recent_max = max_b_t
	output_sprite.material.set_shader_parameter("max_b", max_b_t)
	output_sprite.material.set_shader_parameter("min_b", min_b_t)
	
	var imgs = ImageTexture.create_from_image(image)
	output_sprite.texture =  imgs

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
