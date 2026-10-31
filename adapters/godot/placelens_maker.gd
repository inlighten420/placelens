# PlaceLens in the game: F7, pick your photos, say what they can't show, and it's made and put in the world.
#
# The panel: the photos (the system's own file picker, several at once), notes for the AI ("the other side of the
# building is the same", "about 20 m wide", "it's round"), where to put it (where the photos were taken, or where
# you're looking), and whether paid Meta calls are allowed (SAM 3.1 and Muse Image run on the Meta API; Muse Spark
# Open, which runs the agent, on free credits). The game stands still while it's open, as it does for a pause menu:
# the game reads its keys every frame, so typing would walk you off.
#
# Make starts PlaceLens's survey agent in the background (placelens -C <repo> survey <job> ...) and the game goes on.
# Its steps show along the top as it writes them (progress.jsonl, one JSON line a step); when it's done the pack it
# made is read in like any other, and what it said and asks is shown here.
#
# Where PlaceLens is: placelens.cfg next to this script (install.py writes it), or --placelens-home=<repo> and
# --placelens-python=<python.exe> on the command line.
extends CanvasLayer

const OPEN_KEY := KEY_F7
const AIM_REACH := 40.0	# m: "where you're looking" is the first thing hit within this...
const AIM_FALLBACK := 6.0	# m: ...else the ground this far ahead
const SPREAD := 2.0	# m between several props put at one spot
const POLL := 0.5	# s between reads of the job's progress
const SHOWN_FOR := 20.0	# s a finished job's line stays up
const STEPS := {	# what each of the agent's tools is doing, for the line along the top
	"list_photos": "looking through the photos",
	"describe": "Muse is looking at a photo",
	"pick": "finding which building a photo shows",
	"build": "building it: SAM 3.1 outlines, Muse measures (a few minutes)",
	"prop": "making the object: SAM 3.1 cuts it out, Muse sizes it (a few minutes)",
	"pack": "packing it for the game",
	"ask_person": "it has a question for you (F7)",
}

var packs: Node	# placelens_packs.gd: puts the finished pack up
var python := ""
var home := ""
var job := {}	# the one running: {name, dir, pid, progress, read, ended, spot}
var last := {}	# the last one's end: {summary, questions, pack, error, items}

var _panel: PanelContainer
var _photos := PackedStringArray()
var _photo_list: Label
var _notes: TextEdit
var _where: OptionButton
var _paid: CheckBox
var _make: Button
var _result: Label
var _go: Button
var _line: Label
var _dialog: FileDialog
var _was_mouse := Input.MOUSE_MODE_CAPTURED
var _was_paused := false
var _poll := 0.0
var _line_left := 0.0


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS	# (works while the game stands still for it)
	layer = 60
	_find_placelens()
	_make_panel()
	_line = Label.new()
	_line.add_theme_font_size_override("font_size", 18)
	_line.add_theme_color_override("font_outline_color", Color.BLACK)
	_line.add_theme_constant_override("outline_size", 6)
	_line.set_anchors_and_offsets_preset(Control.PRESET_CENTER_TOP, Control.PRESET_MODE_MINSIZE, 12)
	_line.grow_horizontal = Control.GROW_DIRECTION_BOTH
	_line.visible = false
	add_child(_line)


func _find_placelens() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(get_script().resource_path.get_base_dir().path_join("placelens.cfg")) == OK:
		python = str(cfg.get_value("placelens", "python", ""))
		home = str(cfg.get_value("placelens", "home", ""))
	for arg in OS.get_cmdline_user_args() + OS.get_cmdline_args():
		if arg.begins_with("--placelens-home="):
			home = arg.trim_prefix("--placelens-home=")
		elif arg.begins_with("--placelens-python="):
			python = arg.trim_prefix("--placelens-python=")


# --- the panel ---

func _make_panel() -> void:
	_panel = PanelContainer.new()
	_panel.visible = false
	_panel.custom_minimum_size = Vector2(560, 0)
	_panel.set_anchors_and_offsets_preset(Control.PRESET_CENTER, Control.PRESET_MODE_MINSIZE)
	_panel.grow_horizontal = Control.GROW_DIRECTION_BOTH
	_panel.grow_vertical = Control.GROW_DIRECTION_BOTH
	add_child(_panel)
	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 16)
	_panel.add_child(margin)
	var rows := VBoxContainer.new()
	rows.add_theme_constant_override("separation", 10)
	margin.add_child(rows)

	var title := Label.new()
	title.text = "PlaceLens: make something real from your photos"
	title.add_theme_font_size_override("font_size", 20)
	rows.add_child(title)

	var pick := Button.new()
	pick.text = "Choose photos..."
	pick.pressed.connect(_choose_photos)
	rows.add_child(pick)
	_photo_list = _wrapped("No photos yet. A building from the street, all round an object, or one good photo.")
	rows.add_child(_photo_list)

	rows.add_child(_wrapped("What should the AI know that the photos can't show?"))
	_notes = TextEdit.new()
	_notes.placeholder_text = "e.g. The other side of the building looks the same, and it's about 20 m wide.\n" \
		+ "Or: It's a round water tank, 1.8 m tall."
	_notes.custom_minimum_size = Vector2(0, 96)
	_notes.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	rows.add_child(_notes)

	var where_row := HBoxContainer.new()
	where_row.add_child(_wrapped("Put objects"))
	_where = OptionButton.new()
	_where.add_item("where the photos were taken")
	_where.add_item("where I'm looking")
	where_row.add_child(_where)
	rows.add_child(where_row)
	rows.add_child(_wrapped("(Buildings always go on their real spot, in place of the map's box.)", 13))

	_paid = CheckBox.new()
	_paid.text = "Allow paid Meta calls (SAM 3.1, Muse Image): needed for new photos"
	rows.add_child(_paid)

	var buttons := HBoxContainer.new()
	buttons.alignment = BoxContainer.ALIGNMENT_END
	var close := Button.new()
	close.text = "Close (F7)"
	close.pressed.connect(func(): _show(false))
	_make = Button.new()
	_make.text = "Make it"
	_make.pressed.connect(_start)
	buttons.add_child(close)
	buttons.add_child(_make)
	rows.add_child(buttons)

	rows.add_child(HSeparator.new())
	_result = _wrapped("", 14)
	rows.add_child(_result)
	_go = Button.new()
	_go.text = "Go there"
	_go.visible = false
	_go.pressed.connect(_go_there)
	rows.add_child(_go)

	_dialog = FileDialog.new()
	_dialog.use_native_dialog = true
	_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILES
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_dialog.filters = PackedStringArray(["*.jpg, *.jpeg, *.png, *.heic, *.heif ; Photos"])
	_dialog.title = "Photos for PlaceLens"
	_dialog.files_selected.connect(_photos_chosen)
	add_child(_dialog)


func _wrapped(text: String, size := 15) -> Label:
	var label := Label.new()
	label.text = text
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.add_theme_font_size_override("font_size", size)
	return label


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey and event.pressed and not event.echo):
		return
	if event.keycode == OPEN_KEY or (event.keycode == KEY_ESCAPE and _panel.visible):
		_show(not _panel.visible)
		get_viewport().set_input_as_handled()


func _show(on: bool) -> void:
	if on == _panel.visible:
		return
	if on:
		_was_mouse = Input.mouse_mode
		_was_paused = get_tree().paused
		var net = packs.main.get("net") if packs.main != null else null
		get_tree().paused = not (net != null and net.is_online())	# (as the game's menu: online, the game goes on)
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		_refresh()
	else:
		get_tree().paused = _was_paused
		Input.mouse_mode = _was_mouse
	_panel.visible = on


func _refresh() -> void:
	var running := not job.is_empty()
	_make.disabled = running
	_make.text = "Making one... (F7 to watch)" if running else "Make it"
	if python == "" or home == "":
		_result.text = "PlaceLens isn't set up here: run  python adapters/godot/install.py <this game's Godot folder>  " \
			+ "from the PlaceLens folder."
		_make.disabled = true
	elif running:
		_result.text = "Working on %s: %s" % [job["name"], _line.text]
	elif not last.is_empty():
		_result.text = _told(last)
	_go.visible = not running and not last.get("items", []).is_empty()


func _choose_photos() -> void:
	_dialog.popup_centered_ratio(0.6)


func _photos_chosen(paths: PackedStringArray) -> void:
	_photos = paths
	var names := PackedStringArray()
	for p in paths:
		names.append(p.get_file())
	_photo_list.text = "%d photo(s): %s" % [paths.size(), ", ".join(names)]


# --- making one ---

func _start() -> void:
	if _photos.is_empty():
		_result.text = "Choose a photo or more first."
		return
	var name := "game-" + Time.get_datetime_string_from_system().replace("-", "").replace(":", "").replace("T", "-")
	var dir := home.path_join("out").path_join("game").path_join(name)
	DirAccess.make_dir_recursive_absolute(dir)
	for p in _photos:
		DirAccess.copy_absolute(p, dir.path_join(p.get_file()))
	var progress := dir.path_join("progress.jsonl")
	var saved := ProjectSettings.globalize_path(GameSettings.saved_dir().path_join("placelens"))
	var args := PackedStringArray(["-m", "placelens.cli", "-C", home, "survey", dir, "--progress", progress,
		"--pack-to", saved])
	var notes := _notes.text.strip_edges()
	if notes != "":
		var f := FileAccess.open(dir.path_join("notes.txt"), FileAccess.WRITE)
		f.store_string(notes)
		f.close()
		args.append_array(["--notes-file", dir.path_join("notes.txt")])
	OS.set_environment("PLACELENS_META_OFF", "0" if _paid.button_pressed else "1")	# (the child inherits it)
	var pid := OS.create_process(python, args)
	if pid <= 0:
		_result.text = "Couldn't start PlaceLens (%s). Check placelens.cfg." % python
		return
	job = {"name": name, "dir": dir, "pid": pid, "progress": progress, "read": 0, "ended": false,
		"spot": _aim_spot() if _where.selected == 1 else {}}
	last = {}
	_say("PlaceLens: starting the agent with %d photo(s)" % _photos.size())
	_show(false)


# Where you're looking: the first thing hit within reach, else the ground a little ahead; and which way back to you,
# so what's put there faces you.
func _aim_spot() -> Dictionary:
	var camera := get_viewport().get_camera_3d()
	if camera == null or packs.world == null:
		return {}
	var from := camera.global_position
	var ahead := -camera.global_transform.basis.z
	var query := PhysicsRayQueryParameters3D.create(from, from + ahead * AIM_REACH)
	if packs.main.get("walker") != null:
		query.exclude = [packs.main.walker.get_rid()]
	var hit := camera.get_world_3d().direct_space_state.intersect_ray(query)
	var flat := Vector3(ahead.x, 0.0, ahead.z).normalized()
	var at: Vector3 = hit["position"] if not hit.is_empty() else from + flat * AIM_FALLBACK
	var lat_lon: PackedFloat64Array = packs.world.to_lat_lon(at)
	return {"lat": lat_lon[0], "lon": lat_lon[1], "yaw_deg": rad_to_deg(atan2(-flat.x, -flat.z)),
		"across": Vector2(flat.z, -flat.x)}


func _process(delta: float) -> void:
	if _line_left > 0.0:
		_line_left -= delta
		if _line_left <= 0.0:
			_line.visible = false
	if job.is_empty():
		return
	_poll -= delta
	if _poll > 0.0:
		return
	_poll = POLL
	var ended: bool = job["ended"]
	job["ended"] = not OS.is_process_running(job["pid"])	# (checked before reading: its last line is written first)
	var lines := FileAccess.get_file_as_string(job["progress"]).split("\n", false)
	while not job.is_empty() and job["read"] < lines.size():
		var event = JSON.parse_string(lines[job["read"]])
		job["read"] += 1
		if event is Dictionary:
			_heard(event)
	if not job.is_empty() and ended:
		_finish({"error": "PlaceLens stopped without finishing; its log is in out/survey/%s/" % job["name"]})


func _heard(event: Dictionary) -> void:
	if event.get("done", false):
		_finish(event)
	elif event.has("ok"):
		if not event["ok"]:
			_say("PlaceLens: %s didn't work (%s); the agent carries on" % [event["tool"], event.get("error", "")])
	else:
		var what: String = STEPS.get(event["tool"], event["tool"])
		_say("PlaceLens: " + what + (" (%s)" % event["about"] if event.get("about", "") != "" else ""))


func _finish(event: Dictionary) -> void:
	var name: String = job["name"]
	var spot: Dictionary = job["spot"]
	job = {}
	last = {"summary": str(event.get("summary", "")), "questions": event.get("questions", []),
		"error": str(event.get("error", "")), "items": []}
	var pack = event.get("pack")
	if pack is String and pack != "":
		last["items"] = _place(pack, spot)
		packs.add_pack(pack)
	if last["error"] != "":
		_say("PlaceLens couldn't make it: %s" % last["error"], SHOWN_FOR)
	elif last["items"].is_empty():
		_say("PlaceLens made nothing from %s (F7 for why)" % name, SHOWN_FOR)
	else:
		var names := PackedStringArray(last["items"].map(func(i): return str(i.get("name", i["id"]))))
		_say("PlaceLens made %s (F7 for what it said)" % ", ".join(names), SHOWN_FOR)
	if _panel.visible:
		_refresh()


# The new pack's props put where you were looking (or there anyway, if their photo had no GPS), facing you, and the
# pack's manifest changed to match so they stay there. Returns its items, for Go there.
func _place(pack: String, spot: Dictionary) -> Array:
	var path := pack.path_join("manifest.json")
	var manifest = JSON.parse_string(FileAccess.get_file_as_string(path))
	if typeof(manifest) != TYPE_DICTIONARY:
		return []
	var items: Array = manifest.get("items", [])
	var moved := 0
	for item in items:
		if item.get("kind") != "prop" or (spot.is_empty() and item.get("lat") != null):
			continue
		if spot.is_empty():	# (no GPS, and you didn't say where: where you are now)
			spot = _aim_spot()
			if spot.is_empty():
				continue
		var across: Vector2 = spot["across"] * SPREAD * (moved - (items.size() - 1) / 2.0)
		var at: Vector3 = packs.world.from_lat_lon(float(spot["lat"]), float(spot["lon"])) + Vector3(across.x, 0, across.y)
		var lat_lon: PackedFloat64Array = packs.world.to_lat_lon(at)
		item["lat"] = lat_lon[0]
		item["lon"] = lat_lon[1]
		item["yaw_deg"] = spot["yaw_deg"]
		item["placed"] = "where the player looked"
		moved += 1
	if moved:
		var f := FileAccess.open(path, FileAccess.WRITE)
		f.store_string(JSON.stringify(manifest, " "))
		f.close()
	return items.filter(func(i): return i.get("lat") != null)


func _told(end: Dictionary) -> String:
	var text := ""
	if end.get("error", "") != "":
		text += "It didn't finish: %s\n" % end["error"]
	if end.get("summary", "") != "":
		text += str(end["summary"]).strip_edges() + "\n"
	for q in end.get("questions", []):
		text += "Question: %s\n" % q
	return text.strip_edges()


func _go_there() -> void:
	var first: Dictionary = last["items"][0]
	_show(false)
	if packs.main.has_method("teleport"):
		packs.main.teleport(float(first["lat"]), float(first["lon"]), str(first.get("name", "PlaceLens")))


# The line along the top: up for `seconds`, or (0) till the next one
func _say(text: String, seconds := 0.0) -> void:
	_line.text = text
	_line.visible = true
	_line_left = seconds
