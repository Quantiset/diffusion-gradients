extends SceneTree
# Renders a patterned cone to PNGs.
# godot --path . --resolution 1400x1400 -s scripts/render_cone.gd -- <texture.png> <out_prefix> \
#       <apex_u> <apex_v> <sector_rot> [tile_world] [unroll 1/0] [m00 m10 m01 m11 lattice_fix]
# Writes <out_prefix>_front.png, _seam.png (back side, where the sector edges meet) _angle.png and _close.png (seam close-up).

var views := {
	"front": Transform3D.IDENTITY.translated(Vector3(0, 1.1, 3.7)).looking_at(Vector3(0, -0.05, 0)),
	"seam": Transform3D.IDENTITY.translated(Vector3(0, 1.1, -3.7)).looking_at(Vector3(0, -0.05, 0)),
	"angle": Transform3D.IDENTITY.translated(Vector3(2.7, 0.9, 2.1)).looking_at(Vector3(0, -0.1, 0)),
	"close": Transform3D.IDENTITY.translated(Vector3(0.35, 0.5, -1.9)).looking_at(Vector3(0, 0.35, -0.6)),
}

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var tex_path: String = args[0]
	var out_prefix: String = args[1]

	var img := Image.load_from_file(tex_path)
	img.generate_mipmaps()
	var tex := ImageTexture.create_from_image(img)

	var mat := ShaderMaterial.new()
	mat.shader = load("res://scenes/cone.gdshader")
	mat.set_shader_parameter("tex", tex)
	mat.set_shader_parameter("apex_uv", Vector2(float(args[2]), float(args[3])))
	mat.set_shader_parameter("sector_rot", float(args[4]))
	if args.size() > 5:
		mat.set_shader_parameter("tile_world", float(args[5]))
	if args.size() > 6:
		mat.set_shader_parameter("unroll", args[6] == "1")
	if args.size() > 10:
		mat.set_shader_parameter("lattice_fix", Transform2D(Vector2(float(args[7]), float(args[8])), Vector2(float(args[9]), float(args[10])), Vector2.ZERO))

	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.4692
	mesh.bottom_radius = 1.0
	mesh.height = 3.14
	mesh.radial_segments = 256
	mesh.rings = 32
	mesh.cap_top = false
	mesh.cap_bottom = false
	mesh.material = mat

	var root3d := Node3D.new()
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	root3d.add_child(mi)

	var light := DirectionalLight3D.new()
	light.transform = Transform3D(Basis.from_euler(Vector3(-0.6, 0.5, 0.0)), Vector3.ZERO)
	light.light_energy = 0.8
	root3d.add_child(light)
	var light2 := DirectionalLight3D.new()
	light2.transform = Transform3D(Basis.from_euler(Vector3(-0.6, 0.5 + PI, 0.0)), Vector3.ZERO)
	light2.light_energy = 0.8
	root3d.add_child(light2)

	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.16, 0.17, 0.19)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(1, 1, 1)
	env.ambient_light_energy = 0.6
	var we := WorldEnvironment.new()
	we.environment = env
	root3d.add_child(we)

	var cam := Camera3D.new()
	cam.fov = 60
	root3d.add_child(cam)
	root.add_child(root3d)

	for name in views:
		cam.transform = views[name]
		cam.current = true
		for i in 6:
			await process_frame
		await RenderingServer.frame_post_draw
		var shot := root.get_texture().get_image()
		shot.save_png("%s_%s.png" % [out_prefix, name])
		print("saved %s_%s.png" % [out_prefix, name])
	quit()
