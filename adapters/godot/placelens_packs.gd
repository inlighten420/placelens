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
	for dir in _pack_dirs(args):
		_read(dir)
	print("PlaceLens: %d item(s) in %d pack(s)" % [entries.size(), _pack_dirs(args).size()])
	var maker: Node = load(get_script().resource_path.get_base_dir().path_join("placelens_maker.gd")).new()
	maker.packs = self
	add_child(maker)


# A pack made while playing: its items put up like the rest (an item already here from it is put up again).
func add_pack(dir: String) -> void:
	for e in entries.filter(func(e): return e["dir"] == dir):
		if e["node"] != null:
			e["node"].queue_free()
		entries.erase(e)
	_read(dir)


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


func _read(dir: String) -> void:
	var manifest = JSON.parse_string(FileAccess.get_file_as_string(dir.path_join("manifest.json")))
	if typeof(manifest) != TYPE_DICTIONARY or manifest.get("format") != FORMAT:
		push_warning("PlaceLens: %s isn't a PlaceLens pack" % dir)
		return
	for item in manifest.get("items", []):
		entries.append({"item": item, "dir": dir, "node": null, "loading": false})


# Each second, once the world is built: the next item whose spot is in the built area put up.
func _process(delta: float) -> void:
	if world == null and not _find_world():
		return
	_check -= delta
	if _check > 0.0 or not world.is_ready():
		return
	_check = 1.0
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
			node = doc.generate_scene(state)
		_put_up.call_deferred(e, node))


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
