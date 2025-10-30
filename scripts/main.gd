extends Node2D

var screen_size := Vector2i(1028, 578)

var use_a := false

@onready var output_sprite : Sprite2D = $Sprite2D
@onready var viewport1 : SubViewport = $SubViewport1
@onready var viewport2 : SubViewport = $SubViewport2
@onready var viewports = [viewport1, viewport2]

@export var run_name := "f055_k062"
@export var d_a := 1.0
@export var d_b := 0.5
@export var f := 0.055
@export var k := 0.062
@export var dt := 1.0
@export var noise_scale := 1.0
@export var noise_gen_scale := 1.0

@export var time_until_refresh := 0.1
@export var use_adv_setup := true


func _ready():
	get_node("%FSlider").value = f
	get_node("%KSlider").value = k
	_on_f_slider_drag_ended(f)
	_on_k_slider_drag_ended(k)
	for viewport in viewports:
		viewport.size = screen_size
		viewport.get_node("Sprite2D").material.set_shader_parameter("d_a", d_a)
		viewport.get_node("Sprite2D").material.set_shader_parameter("d_b", d_b)
		viewport.get_node("Sprite2D").material.set_shader_parameter("f", f)
		viewport.get_node("Sprite2D").material.set_shader_parameter("k", k)
		viewport.get_node("Sprite2D").material.set_shader_parameter("dt", dt)
		viewport.get_node("Sprite2D").material.set_shader_parameter("noise_scale", noise_scale)
	
	var img := Image.create(screen_size.x, screen_size.y, false, Image.FORMAT_RGBAF)
	
	if use_adv_setup:
		
		var b = f + sqrt(f * f - 4 * f * (f+k) * (f+k) ) / (2 * (f + k))
		
		for x in range(0, screen_size.x):
			for y in range(0, screen_size.y):
				img.set_pixel(x, y, Color( (k + f) / b , b , 0, 1)) 
		
	else:
		
		var noise : FastNoiseLite = $NoisePattern.texture.noise
		
		for x in range(0, screen_size.x):
			for y in range(0, screen_size.y):
				img.set_pixel(x, y, Color( 1 , noise.get_noise_2d(x, y) , 0, 1)) 
		
		var mid_size = 5
		for x in range(screen_size.x/2 - mid_size, screen_size.x/2 + mid_size):
			for y in range(screen_size.y/2 - mid_size, screen_size.y/2 + mid_size):
				img.set_pixel(x, y, Color( 1 , 1 , 0, 1)) 
	
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
	var min_b = 1.0;
	var max_b = 0.0;
	
	if use_adv_setup:
		var edge_offset = 50;
		for x in range(edge_offset, screen_size.x-edge_offset):
			for y in range(edge_offset, screen_size.y-edge_offset):
				var col = image.get_pixel(x, y)
				min_b = min(min_b, col.g)
				max_b = max(max_b, col.g)
		print(min_b, max_b)
		output_sprite.material.set_shader_parameter("max_b", max_b)
		output_sprite.material.set_shader_parameter("min_b", min_b)

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
	$Sprite2D.texture.get_image().save_png("res://saves/"+run_name+".png")


func _on_f_slider_drag_ended(value_changed):
	var new_f := float(get_node("%FSlider").value)
	get_node("%FLabel").text = str(new_f)
	f = new_f
	for viewport in viewports: 
		viewport.get_node("Sprite2D").material.set_shader_parameter("f", f)

func _on_k_slider_drag_ended(value_changed):
	var new_k := float(get_node("%KSlider").value)
	get_node("%KLabel").text = str(new_k)
	k = new_k
	for viewport in viewports: 
		viewport.get_node("Sprite2D").material.set_shader_parameter("k", k)
