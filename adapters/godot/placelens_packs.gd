# PlaceLens packs in Open-Planet: buildings and props made from phone photos, standing where they really are.
#
# An autoload. Installed by adapters/godot/install.py, which copies this file into the game and adds one line to
# project.godot:  [autoload] PlaceLens="*res://placelens/placelens_packs.gd"
#
# Packs are folders made by `placelens pack` (manifest.json, format "placelens-pack", and .glb models), put in
# <saved dir>/placelens/<pack>/ or given with --placelens=<folder>. --placelens-go starts you at the first pack.
#
# Follows Open-Planet's rules: places by lat/lon (world.from_lat_lon), checks once a second like props.gd, one item
# at a time, models read on a worker thread, and moved with the world when its origin shifts.
#
# F7 makes a new one in the game, from your photos and notes (placelens_maker.gd); it's added here when it's done.
extends Node

const FORMAT := "placelens-pack"

var main: Node = null
var world: Node3D = null
var holder: Node3D = null
var entries: Array = []  # {item, dir, node, loading}
var removed: PackedInt64Array = PackedInt64Array()  # OSM buildings this took away, given back on exit
var go_to := false
var _check := 0.0


func _ready() -> void:
	var args := OS.get_cmdline_user_args() + OS.get_cmdline_args()
	go_to = args.has("--placelens-go")
	get_tree().node_added.connect(_on_node_added)
	var cfg := ConfigFile.new()
	if cfg.load(SETTINGS) == OK:
		car_wheel = str(cfg.get_value("car", "steering_wheel", ""))
		if car_wheel != "" and not FileAccess.file_exists(car_wheel):
			car_wheel = ""	# (its model's gone: deleted)
	for dir in _pack_dirs(args):
		_read(dir)
	print("PlaceLens: %d item(s) in %d pack(s)" % [entries.size(), _pack_dirs(args).size()])
	var here: String = get_script().resource_path.get_base_dir()
	maker = load(here.path_join("placelens_maker.gd")).new()
	maker.packs = self
	add_child(maker)
	inventory = load(here.path_join("placelens_inventory.gd")).new()	# I: everything you can put down
	inventory.packs = self
	inventory.maker = maker
	add_child(inventory)


var maker: Node	# placelens_maker.gd: F7, photos to a pack
var inventory: Node	# placelens_inventory.gd: I, My things and Make from photos


# A pack made while playing: its items put up like the rest (an item already here from it is put up again); its
# props marked "New" in the inventory.
func add_pack(dir: String) -> void:
	for e in entries.filter(func(e): return e["dir"] == dir):
		if e["node"] != null:
			e["node"].queue_free()
		entries.erase(e)
	_offered_ids.clear()
	_read(dir)
	if inventory != null:
		for id in _offered_ids:
			inventory.made(id)


# Pack folders: the ones named on the command line, and every folder in <saved dir>/placelens with a manifest.
func _pack_dirs(args: PackedStringArray) -> PackedStringArray:
	var roots := PackedStringArray()
	for arg in args:
		if arg.begins_with("--placelens="):
			roots.append(arg.trim_prefix("--placelens="))
	roots.append(GameSettings.saved_dir().path_join("placelens"))
	var dirs := PackedStringArray()
	for root in roots:
		if FileAccess.file_exists(root.path_join("manifest.json")):
			dirs.append(root)
			continue
		var d := DirAccess.open(root)
		if d == null:
			continue
		for sub in d.get_directories():
			if FileAccess.file_exists(root.path_join(sub).path_join("manifest.json")):
				dirs.append(root.path_join(sub))
	return dirs


# Packs dropped in while playing (PlaceLens Studio's "Send to game"): looked for every few seconds and put up like
# one made with F7, "New" in My things.
var _looked := 0
var _tried := {}	# pack folders already read (or tried: one that isn't a pack isn't tried again)
func _new_packs() -> void:
	_looked += 1
	if _looked % 3 != 0:	# (every 3 s)
		return
	for e in entries:
		_tried[e["dir"]] = true
	for dir in _pack_dirs(OS.get_cmdline_user_args() + OS.get_cmdline_args()):
		if not _tried.has(dir):
			_tried[dir] = true
			print("PlaceLens: new pack %s" % dir)
			add_pack(dir)


func _read(dir: String) -> void:
	var manifest = JSON.parse_string(FileAccess.get_file_as_string(dir.path_join("manifest.json")))
	if typeof(manifest) != TYPE_DICTIONARY or manifest.get("format") != FORMAT:
		push_warning("PlaceLens: %s isn't a PlaceLens pack" % dir)
		return
	for item in manifest.get("items", []):
		entries.append({"item": item, "dir": dir, "node": null, "loading": false})
		if item.get("kind") in ["prop", "building"]:	# (a building: on the map where it is, and a copy to place)
			_offer_prop(item, dir)


# --- PlaceLens props in the game's own prop menu (build mode, My things) ---
# Open-Planet's props are made of simple shapes (a "look" of parts). Each PlaceLens prop is offered as one: a prop
# file in <saved dir>/packs/placelens/props/ (so it's there next time too) whose look is a plain box its real size
# (which gives it its solid, and its sitting on the ground), and wherever the game puts one up (the see-through one
# being placed, or one on the map) the box's look is swapped for the real model.

const PROP_PREFIX := "placelens_"
var _models := {}	# glb path -> a model read once, copied for each prop shown
var _offered: Array = []	# prop entries not yet given to the game's prop keeper (it's made with the world)
var _offered_ids := PackedStringArray()	# the last pack's prop ids (for "New" in the inventory)


func _offer_prop(item: Dictionary, dir: String) -> void:
	var id := PROP_PREFIX + (dir.get_file() + "_" + str(item["id"])).to_lower().validate_filename() \
		.replace(" ", "_").replace(".", "_").replace("-", "_")
	var size: Array = item.get("size_m", [1.0, 1.0, 1.0])	# wide, deep, tall
	var wide := float(size[0])
	var deep := float(size[1]) if size.size() > 2 else wide
	var tall := float(size[2]) if size.size() > 2 else float(size[1])
	var label = item.get("name")
	if item.get("kind") == "building":	# (its size from its footprint; the roof on top of its walls' height)
		var span := _footprint_span(item.get("footprint", []))
		wide = span.x
		deep = span.y
		tall = float(item.get("height_m", 3.0)) * 1.4
		if label == null or str(label) == "":
			label = "Building from photos"
	elif label == null:
		label = item["id"]
	var entry := {"id": id, "name": str(label).capitalize() + " (PlaceLens)",
		"placelens_model": dir.path_join(str(item["model"])),
		"look": {"parts": [{"shape": "box", "at": [0, tall / 2.0, 0], "size": [wide, tall, deep],
			"colour": "#8a8a8a", "finish": "matte"}]}}
	var folder := GameSettings.saved_dir().path_join("packs/placelens")
	DirAccess.make_dir_recursive_absolute(folder.path_join("props"))
	if not FileAccess.file_exists(folder.path_join("pack.json")):
		_write(folder.path_join("pack.json"), {"name": "PlaceLens",
			"about": "Things made from photos with PlaceLens (F7), real size; in build mode under My things."})
	_write(folder.path_join("props").path_join(id + ".json"), entry)
	_offered.append(entry)
	_offered_ids.append(id)
	_give_props()


# --- your own steering wheel in the car ---
# A thing made from photos as the car's steering wheel: hung on the car's own "SteeringWheel" part (which the car
# turns with the steering, in the cockpit view), whose own mesh is hidden. Remembered (user://placelens.cfg).

const WHEEL_SIZE := 0.36	# m across: what it's scaled to fit, if it's far off a real wheel's size (they're 0.28-0.38)
const SETTINGS := "user://placelens.cfg"
var car_wheel := ""	# the model (glb path) in the car, "" for the car's own


func set_car_wheel(model: String) -> void:
	car_wheel = model
	var cfg := ConfigFile.new()
	cfg.load(SETTINGS)
	cfg.set_value("car", "steering_wheel", model)
	cfg.save(SETTINGS)
	_apply_car_wheel()


func _apply_car_wheel() -> void:
	var car = main.get("car") if main != null else null
	if car == null:
		return
	var part := car.get_node_or_null("SteeringWheel") as MeshInstance3D
	if part == null:
		return
	var ours: Node = car.get_node_or_null("PlaceLensWheel")
	if ours != null and ours.get_meta("model", "") == car_wheel:
		return
	if ours != null:
		ours.queue_free()
		_wheel = null
	if not part.has_meta("own_mesh"):
		part.set_meta("own_mesh", part.mesh)
	if car_wheel == "":
		part.mesh = part.get_meta("own_mesh")
		return
	var model := _model(car_wheel)
	if model == null:
		part.mesh = part.get_meta("own_mesh")
		return
	# Facing the driver (the part's +Z, its turning axis) and centred on the hub. Made lying flat (front up, +Y):
	# stood up first; made standing (front +Z): as it is.
	var holder := Node3D.new()
	holder.name = "PlaceLensWheel"
	holder.set_meta("model", car_wheel)
	holder.add_child(model)
	var box := _bounds(model)
	var flat := box.size.y < 0.5 * minf(box.size.x, box.size.z)
	var turn := Basis(Vector3.RIGHT, PI / 2.0) if flat else Basis()
	var across := maxf(box.size.x, box.size.z if flat else box.size.y)
	var scale := 1.0 if across > 0.2 and across < 0.5 else WHEEL_SIZE / maxf(across, 0.01)
	model.transform = Transform3D(turn.scaled(Vector3.ONE * scale), -(turn * box.get_center()) * scale)
	part.mesh = null	# (the car's own wheel gone; the part, which the car turns, kept: ours follows it)
	car.add_child(holder)
	_wheel = holder
	_wheel_part = part


# Each frame: ours where the car's wheel is, turned as far as the steering turns it, but its face tilted to look at
# the driver's eyes (the car's column suits an average driver: the player's are set by their height, 6'4" here, and
# from up there the car's tilt laid the wheel flat, its top in the dashboard), and brought a little towards them.
var _wheel: Node3D = null
var _wheel_part: Node3D = null
const WHEEL_TOWARDS := 0.02	# m towards the driver from the hub: the car's column ends just behind it (10 cm towards the dashboard put the column's end in front of the wheel)


func _follow_wheel() -> void:
	if not (is_instance_valid(_wheel) and is_instance_valid(_wheel_part)):
		_wheel = null
		return
	_wheel.visible = _wheel_part.visible	# (shown in the cockpit view, as the car shows its own)
	if not _wheel.visible:
		return
	var car := _wheel.get_parent() as Node3D
	var t: Transform3D = _wheel_part.transform	# (the car's: tilt x turn about the column, at the hub)
	var camera := get_viewport().get_camera_3d()
	var eyes: Vector3 = car.global_transform.affine_inverse() * camera.global_position if camera != null \
		else t.origin + Vector3(0, 0.3, 0.5)
	var look := eyes - t.origin
	var tilt := atan2(-look.y, look.z)	# (about X: the face, +Z, towards the eyes)
	var column: Basis = Basis(Vector3.RIGHT, _column_pitch(t.basis)).inverse() * t.basis
	var turn := atan2(column.x.y, column.x.x)
	var face := Basis(Vector3.RIGHT, tilt)
	_wheel.transform = Transform3D(face * Basis(Vector3(0, 0, 1), turn), t.origin + face.z * WHEEL_TOWARDS)


# The car's column tilt, out of its wheel's turned basis (tilt about X, then turn about Z: its Z axis is the column's)
static func _column_pitch(b: Basis) -> float:
	return atan2(-b.z.y, b.z.z)


static func _bounds(node: Node) -> AABB:
	var box := AABB()
	var first := true
	for mesh in node.find_children("*", "MeshInstance3D", true, false):
		var part: AABB = mesh.transform * mesh.get_aabb()
		box = part if first else box.merge(part)
		first = false
	return box


# Delete a thing made from photos (its prop id): from the game's props (and every copy placed), from the map, and from
# its pack (its model too; the pack itself once it's empty), so it doesn't come back next time.
func delete_thing(id: String) -> void:
	var folder := GameSettings.saved_dir().path_join("packs/placelens/props")
	var keeper := _prop_keeper()
	var kind: Dictionary = keeper.kinds.get(id, {}) if keeper != null else {}
	if kind.is_empty():
		var parsed = JSON.parse_string(FileAccess.get_file_as_string(folder.path_join(id + ".json")))
		kind = parsed if parsed is Dictionary else {}
	var model := str(kind.get("placelens_model", ""))
	if model != "" and model == car_wheel:	# (the car gets its own wheel back)
		set_car_wheel("")
	DirAccess.remove_absolute(folder.path_join(id + ".json"))
	if keeper != null:
		for index in keeper._shown.keys():
			if keeper.placed[index].get("id") == id:
				keeper._shown[index].queue_free()
				keeper._shown.erase(index)
		keeper.kinds.erase(id)
		if keeper.chosen == id:
			keeper.stop()
	if model == "":
		return
	var dir := model.get_base_dir()
	for e in entries.filter(func(e): return e["dir"] == dir and str(e["item"].get("model")) == model.get_file()):
		if e["node"] != null:
			e["node"].queue_free()
		if e["item"].get("kind") == "building":	# (the map's own box back where it stood)
			var osm := str(e["item"].get("replaces", ""))
			var still := entries.any(func(o): return o != e and str(o["item"].get("replaces", "")) == osm)
			if world != null and not still and osm.begins_with("way/") and removed.has(int(osm.trim_prefix("way/"))):
				var kept := PackedInt64Array()
				for b in world.get_removed_buildings():
					if b != int(osm.trim_prefix("way/")):
						kept.append(b)
				world.set_removed_buildings(kept)
				removed.remove_at(removed.find(int(osm.trim_prefix("way/"))))
		entries.erase(e)
	var path := dir.path_join("manifest.json")
	var manifest = JSON.parse_string(FileAccess.get_file_as_string(path))
	if manifest is Dictionary:
		manifest["items"] = manifest.get("items", []).filter(func(i): return str(i.get("model")) != model.get_file())
		DirAccess.remove_absolute(model)
		if manifest["items"].is_empty():
			for file in DirAccess.get_files_at(dir):
				DirAccess.remove_absolute(dir.path_join(file))
			DirAccess.remove_absolute(dir)
		else:
			_write(path, manifest)
	print("PlaceLens: deleted %s" % id)


# A footprint's width east-west and depth north-south in metres ([lon, lat] corners)
static func _footprint_span(corners: Array) -> Vector2:
	if corners.is_empty():
		return Vector2(10, 10)
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for c in corners:
		var p := Vector2(float(c[0]), float(c[1]))
		lo = Vector2(minf(lo.x, p.x), minf(lo.y, p.y))
		hi = Vector2(maxf(hi.x, p.x), maxf(hi.y, p.y))
	var m_per_lon := 111320.0 * cos(deg_to_rad((lo.y + hi.y) / 2.0))
	return Vector2((hi.x - lo.x) * m_per_lon, (hi.y - lo.y) * 110540.0)


func _write(path: String, data: Dictionary) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(data, "\t"))
		f.close()


# The game's prop keeper (main.builder.props), once the game has one; older games have none.
func _prop_keeper() -> Node:
	var scene := get_tree().current_scene if is_inside_tree() else null
	var builder = scene.get("builder") if scene != null else null
	var keeper = builder.get("props") if builder != null else null
	return keeper if keeper != null and keeper.has_method("add_kind") else null


func _give_props() -> void:
	var keeper := _prop_keeper()
	if keeper == null:
		return
	for entry in _offered:
		keeper.add_kind(entry)
	_offered.clear()
	for arg in OS.get_cmdline_user_args():	# --placelens-wheel-test=<glb>: put it in the car, say if it's there
		if arg.begins_with("--placelens-wheel-test="):
			car_wheel = arg.trim_prefix("--placelens-wheel-test=")	# (not remembered: a test)
			for wait in 300:	# (the car comes a little after the world)
				if main.get("car") != null:
					break
				await get_tree().process_frame
			_apply_car_wheel()
			var part = main.car.get_node_or_null("SteeringWheel") if main.get("car") != null else null
			print("PlaceLens: wheel test: car %s, its wheel part %s, ours on it: %s, own mesh hidden: %s" % [
				main.get("car") != null, part != null, part != null and part.has_node("PlaceLensWheel"),
				part != null and part.mesh == null])
			get_tree().quit()
			return
	for arg in OS.get_cmdline_user_args():	# --placelens-delete=<id>: as the inventory's Delete (tests, tidying)
		if arg.begins_with("--placelens-delete="):
			delete_thing(arg.trim_prefix("--placelens-delete="))
			get_tree().quit()
			return
	if OS.get_cmdline_user_args().has("--placelens-inventory-test"):	# (a check: open it, hold the first of ours)
		inventory.show_it(true, 0)
		await get_tree().process_frame
		var cards: Array = inventory._page.find_children("*", "Button", true, false)
		var ours: Array = keeper.kinds.keys().filter(func(k): return String(k).begins_with(PROP_PREFIX))
		print("PlaceLens: inventory test: %d thing(s) shown, %d from photos" % [cards.size(), ours.size()])
		if not ours.is_empty():
			inventory._hold(ours[0])
			print("PlaceLens: inventory test: holding %s, build mode %s" % [keeper.chosen,
				"on" if main.builder.active else "OFF"])
		get_tree().quit()
		return
	if OS.get_cmdline_user_args().has("--placelens-prop-test"):	# (a check: each of ours put up as the game would)
		for id in keeper.kinds.keys().filter(func(k): return String(k).begins_with(PROP_PREFIX)):
			for which in ["see-through", "placed"]:	# (two at once, as in the game: the second gets renamed)
				var shown: Node3D = keeper.make(keeper.kinds[id], which == "placed")
				if which == "placed":	# (as the game's _show: its place in the list, then in)
					keeper.placed.append({"id": id})
					shown.set_meta("placed", keeper.placed.size() - 1)
				keeper.add_child(shown)
				if which == "placed":
					keeper.placed.pop_back()
				var real := shown.get_children().any(func(c): return c is Node3D and not c is GeometryInstance3D \
					and not c is StaticBody3D)
				print("PlaceLens: prop test %s (%s, named %s): %s" % [id, which, shown.name,
					"real model shown" if real else "STILL A BOX"])
		get_tree().quit()


# A prop of ours put up by the game: its box's look hidden, the real model put in its place.
func _on_node_added(node: Node) -> void:
	# Which prop: a placed one carries its place in the keeper's list ("placed" meta: Godot renames a node whose name
	# is taken, so the one put down while the see-through one is out comes in as "@Node3D@123"); the see-through one
	# goes by its name, "Prop_<id>".
	if not node is Node3D:
		return
	var keeper := node.get_parent()
	var kinds = keeper.get("kinds") if keeper != null else null
	if not kinds is Dictionary:
		return
	var id := String(node.name).trim_prefix("Prop_")
	if node.has_meta("placed"):
		var placed = keeper.get("placed")
		var index := int(node.get_meta("placed"))
		if placed is Array and index >= 0 and index < placed.size() and placed[index] is Dictionary:
			id = String(placed[index].get("id", ""))
	if not (id.begins_with(PROP_PREFIX) and kinds.has(id)):
		return
	var model := _model(str(kinds[id].get("placelens_model", "")))
	if model == null:
		return
	for child in node.get_children():
		if child is GeometryInstance3D:	# (the box's look; its solid stays)
			child.visible = false
	node.add_child(model)


func _model(path: String) -> Node3D:
	if not _models.has(path):
		var doc := GLTFDocument.new()
		var state := GLTFState.new()
		_models[path] = _coloured(doc.generate_scene(state)) if doc.append_from_file(path, state) == OK else null
		if _models[path] == null:
			push_warning("PlaceLens: couldn't read %s" % path)
	var model: Node3D = _models[path]
	return model.duplicate() if model != null else null


# Each second, once the world is built: the next item whose spot is in the built area put up.
func _process(delta: float) -> void:
	if _wheel != null:
		_follow_wheel()
	if world == null and not _find_world():
		return
	if not _offered.is_empty():
		_give_props()
	if car_wheel != "":
		_apply_car_wheel()	# (cheap when it's on already; a new car gets it too)
	_check -= delta
	if _check > 0.0 or not world.is_ready():
		return
	_check = 1.0
	_new_packs()
	if go_to and not entries.is_empty() and main.has_method("teleport") and entries[0]["item"].get("lat") != null:
		go_to = false
		var first: Dictionary = entries[0]["item"]
		main.teleport(float(first["lat"]), float(first["lon"]), str(first.get("name", "PlaceLens")))
		return
	for e in entries:
		var item: Dictionary = e["item"]
		if item.get("lat") == null:	# (a prop with no GPS that nobody has put anywhere yet)
			continue
		var here: bool = world.contains_lat_lon(float(item["lat"]), float(item["lon"]))
		if e["node"] != null and not here:  # (a new area elsewhere: taken down, put up again on coming back)
			e["node"].queue_free()
			e["node"] = null
		elif e["node"] == null and not e["loading"] and here:
			_load(e)
			return


func _find_world() -> bool:
	var scene := get_tree().current_scene
	if scene == null or scene.get("world") == null:
		return false
	main = scene
	world = scene.world
	holder = Node3D.new()
	holder.name = "PlaceLens"
	scene.add_child(holder)
	world.origin_shifted.connect(_on_origin_shifted)
	world.tree_exiting.connect(_give_back)  # (while the world still exists: by the autoload's own exit it may not)
	return true


# The model read off the main thread (the camshaft rule), then put up on it.
func _load(e: Dictionary) -> void:
	e["loading"] = true
	var path: String = str(e["dir"]).path_join(str(e["item"]["model"]))
	WorkerThreadPool.add_task(func():
		var doc := GLTFDocument.new()
		var state := GLTFState.new()
		var node: Node3D = null
		if doc.append_from_file(path, state) == OK:
			node = _coloured(doc.generate_scene(state))
		_put_up.call_deferred(e, node))


# A carved object's colours are in its mesh (glTF COLOR_0, straight from the photos' pixels), but Godot's glTF loader
# doesn't turn them on: without this every carved object (a teddy, a chair) shows plain white (found 2 Nov).
static func _coloured(node: Node3D) -> Node3D:
	if node == null:
		return null
	for mi in node.find_children("*", "MeshInstance3D", true, false):
		var mesh: Mesh = mi.mesh
		for i in mesh.get_surface_count():
			if mesh.surface_get_format(i) & Mesh.ARRAY_FORMAT_COLOR == 0:
				continue
			var mat := mesh.surface_get_material(i) as BaseMaterial3D
			if mat != null:
				mat.vertex_color_use_as_albedo = true
				mat.vertex_color_is_srgb = true	# (written as the photos' pixel values)
	return node


func _put_up(e: Dictionary, node: Node3D) -> void:
	e["loading"] = false
	if node == null:
		push_warning("PlaceLens: couldn't read %s" % e["item"]["model"])
		e["node"] = Node3D.new()  # (not tried again)
		return
	var item: Dictionary = e["item"]
	var at: Vector3 = world.from_lat_lon(float(item["lat"]), float(item["lon"]))
	var y: float = world.ground_height(at)
	if item["kind"] == "building":
		y = _lowest_ground(item, y)
		_replace(str(item.get("replaces", "")))
	_solid(node)
	holder.add_child(node)
	node.global_transform = Transform3D(Basis(Vector3.UP, deg_to_rad(float(item.get("yaw_deg", 0.0)))),
		Vector3(at.x, y, at.z))
	e["node"] = node
	print("PlaceLens: %s %s put up" % [item["kind"], item["name"] if item.get("name") else item["id"]])


# A building stands on the lowest ground under it: its walls reach down there (PlaceLens builds them that way, as
# Open-Planet does its own).
func _lowest_ground(item: Dictionary, y: float) -> float:
	for corner in item.get("footprint", []):
		var p: Vector3 = world.from_lat_lon(float(corner[1]), float(corner[0]))
		y = minf(y, world.ground_height(p))
	return y


# Open-Planet's own box for this OSM building taken away, drawn and solid.
func _replace(osm_id: String) -> void:
	if not osm_id.begins_with("way/"):
		return
	var id := int(osm_id.trim_prefix("way/"))
	if not world.get_removed_buildings().has(id):
		world.remove_building(id)
		removed.append(id)


# Solid to drive into and walk on: a collision shape from each mesh.
func _solid(node: Node) -> void:
	if node is MeshInstance3D:
		node.create_trimesh_collision()
	for child in node.get_children():
		_solid(child)


func _on_origin_shifted(offset: Vector3) -> void:
	for e in entries:
		if e["node"] != null:
			e["node"].global_position += offset


# The OSM buildings this took away given back, so the player's own saved list doesn't keep them. Done as the world
# leaves the scene, while it can still be asked; after that, nothing here touches it.
func _give_back() -> void:
	if world != null and not removed.is_empty():
		var kept := PackedInt64Array()
		for id in world.get_removed_buildings():
			if not removed.has(id):
				kept.append(id)
		world.set_removed_buildings(kept)
	removed.clear()
	world = null
	holder = null
	for e in entries:
		e["node"] = null
