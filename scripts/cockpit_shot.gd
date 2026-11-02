# A picture of a car's cockpit view, as the game draws it: the real car (GeodriveCar), the camera at the driver's eyes.
#   godot --path <game> --script cockpit_shot.gd -- --body=1 --out=<png> [--eye=0.02] [--forward=0.04]
extends SceneTree

func _initialize() -> void:
	var body := 1
	var out := "cockpit.png"
	var eye := 0.0
	var forward := 0.0
	var look_down := 12.0
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--body="): body = int(arg.trim_prefix("--body="))
		elif arg.begins_with("--out="): out = arg.trim_prefix("--out=")
		elif arg.begins_with("--eye="): eye = float(arg.trim_prefix("--eye="))
		elif arg.begins_with("--forward="): forward = float(arg.trim_prefix("--forward="))
		elif arg.begins_with("--down="): look_down = float(arg.trim_prefix("--down="))
	var root := Node3D.new()
	get_root().add_child(root)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, 30, 0)
	root.add_child(sun)
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color(0.55, 0.7, 0.85)
	env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.environment.ambient_light_color = Color(0.6, 0.6, 0.62)
	root.add_child(env)
	var car: RigidBody3D = ClassDB.instantiate("GeodriveCar")
	car.set("body", body)
	car.freeze = true
	root.add_child(car)
	await process_frame
	car.set_view(1)
	await process_frame
	var cam := Camera3D.new()
	cam.fov = 80.0
	root.add_child(cam)
	var at: Vector3 = car.get_cockpit_camera() + Vector3(0, eye, -forward)
	cam.global_position = car.global_transform * at
	cam.global_rotation_degrees = Vector3(-look_down, 0, 0)
	cam.make_current()
	for i in 10:
		await process_frame
	await RenderingServer.frame_post_draw
	get_root().get_texture().get_image().save_png(out)
	print("saved ", out, " eyes at ", at, " car ", car.get("body"))
	quit()
