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

const Brand := preload("placelens_brand.gd")

const OPEN_KEY := KEY_F7
const AIM_REACH := 40.0	# m: "where you're looking" is the first thing hit within this...
const AIM_FALLBACK := 6.0	# m: ...else the ground this far ahead
const SPREAD := 2.0	# m between several props put at one spot
const POLL := 0.5	# s between reads of the job's progress
const SHOWN_FOR := 20.0	# s a finished job's line stays up
const STEPS := {	# what each of the agent's tools is doing, for the line along the top
	"list_photos": "looking through the photos",
	"describe": "Muse is looking at the photos",
	"pick": "finding which building a photo shows",
	"build": "building it: SAM 3.1 outlines, Muse measures (a few minutes)",
	"prop": "making the object: SAM 3.1 cuts it out, Muse sizes it (a few minutes)",
	"pack": "packing it for the game",
	"ask_person": "it has a question for you (F7)",
}
const KINDS := ["", "building", "object"]	# the "What is it?" choices, as survey --kind takes them
const SPINNER := ["|", "/", "-", "\\"]
const CARD_WIDTH := 330.0	# px: the progress card at the left edge
const LOG_KEPT := 200	# lines of the make's log kept for the panel
const CARD_LOG := 4	# of them, the last few shown on the card
const TOAST_FOR := 7.0	# s a "ready" notification stays up

var packs: Node	# placelens_packs.gd: puts the finished pack up
var python := ""
var home := ""
var job := {}	# the one running: {name, dir, pid, progress, read, ended, spot, started, step, doing, detail, done, total}
var last := {}	# the last one's end: {summary, questions, pack, error, items}

var _panel: PanelContainer
var _photos := PackedStringArray()
var _photo_list: Label
var _notes: TextEdit
var _where: OptionButton
var _kind: OptionButton
var _paid: CheckBox
var _bar: ProgressBar
var _make: Button
var _cancel_button: Button
var _log_box: RichTextLabel
var _log_lines := PackedStringArray()	# the running (or last) make's log
var _toasts: VBoxContainer	# the "ready" notifications, top left
var _result: Label
var _go: Button
var _line: Label
var _card: PanelContainer
var _dialog: FileDialog
var photos_dir := ""	# where the photo picker opens: placelens.cfg "photos", else the folder photos last came from
var _was_mouse := Input.MOUSE_MODE_CAPTURED
var _was_paused := false
var _poll := 0.0
var _line_left := 0.0


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS	# (works while the game stands still for it)
	layer = 60
	_find_placelens()
	_make_panel()
	add_child(Brand.badge())	# top right, always: this is PlaceLens, an add-on, not Open-Planet itself
	# A small card at the left edge (clear of the street name along the top): what it's doing, how long so far, and
	# a bar while a step counts photos.
	_card = PanelContainer.new()
	_card.add_theme_stylebox_override("panel", Brand.panel_look(0.78, 12))
	_card.custom_minimum_size = Vector2(CARD_WIDTH, 0)
	_card.set_anchors_and_offsets_preset(Control.PRESET_CENTER_LEFT, Control.PRESET_MODE_MINSIZE, 16)
	_card.grow_vertical = Control.GROW_DIRECTION_BOTH
	_card.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_card.visible = false
	add_child(_card)
	var stack := VBoxContainer.new()
	stack.add_theme_constant_override("separation", 8)
	_card.add_child(stack)
	_line = Label.new()
	_line.add_theme_font_size_override("font_size", 15)
	_line.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_line.custom_minimum_size = Vector2(CARD_WIDTH - 24, 0)
	stack.add_child(_line)
	_bar = ProgressBar.new()
	_bar.custom_minimum_size = Vector2(0, 8)
	_bar.show_percentage = false
	_bar.max_value = 1.0
	_bar.visible = false
	stack.add_child(_bar)
	_scripted.call_deferred()


# A make started from the command line, as if from the panel (for tests and demos, no clicking):
#   --placelens-make=<photo>[,<photo>...]  --placelens-kind=object|building  --placelens-paid  --placelens-quit
# (--placelens-quit: the game closes when it's done; the outcome is printed to the log either way)
var _quit_when_done := false


func _scripted() -> void:
	var make := ""
	var kind := ""
	var paid := false
	for arg in OS.get_cmdline_user_args() + OS.get_cmdline_args():
		if arg.begins_with("--placelens-make="):
			make = arg.trim_prefix("--placelens-make=")
		elif arg.begins_with("--placelens-kind="):
			kind = arg.trim_prefix("--placelens-kind=")
		elif arg == "--placelens-paid":
			paid = true
		elif arg == "--placelens-quit":
			_quit_when_done = true
	if make == "":
		return
	_photos_chosen(PackedStringArray(make.split(",", false)))
	_kind.selected = maxi(KINDS.find(kind), 0)
	_paid.button_pressed = paid
	print("PlaceLens: scripted make: %s (kind %s, paid %s)" % [_photo_list.text, kind if kind != "" else "AI decides", paid])
	_start()
	if job.is_empty():
		print("PlaceLens: scripted make didn't start: %s" % _result.text)
		if _quit_when_done:
			get_tree().quit(1)


# The game closing takes its running job with it (the agent and the command it's running), so a make left behind
# can't go on spending, nor drop a pack into a game that's gone.
func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST and not job.is_empty():	# (before the game quits, crash or not)
		_stop_job()


func _exit_tree() -> void:
	if not job.is_empty():
		_stop_job()
		print("PlaceLens: the game closed; stopped the make in progress (%s)" % job["name"])


# Stops the running job's agent and the command it's running (the whole process tree).
func _stop_job() -> void:
	if job.is_empty() or not OS.is_process_running(job["pid"]):
		return
	if OS.get_name() == "Windows":
		OS.execute("taskkill", PackedStringArray(["/PID", str(job["pid"]), "/T", "/F"]))
	else:
		OS.kill(job["pid"])


func _cancel() -> void:
	if job.is_empty():
		return
	_stop_job()
	_log("cancelled")
	_finish({"error": "cancelled (nothing was added; what it saved so far is in out/survey/%s/)" % job["name"]})


func _find_placelens() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(get_script().resource_path.get_base_dir().path_join("placelens.cfg")) == OK:
		python = str(cfg.get_value("placelens", "python", ""))
		home = str(cfg.get_value("placelens", "home", ""))
		photos_dir = str(cfg.get_value("placelens", "photos", ""))
	for arg in OS.get_cmdline_user_args() + OS.get_cmdline_args():
		if arg.begins_with("--placelens-home="):
			home = arg.trim_prefix("--placelens-home=")
		elif arg.begins_with("--placelens-python="):
			python = arg.trim_prefix("--placelens-python=")


# --- the panel ---

func _make_panel() -> void:
	_panel = PanelContainer.new()
	_panel.visible = false
	_panel.add_theme_stylebox_override("panel", Brand.panel_look(0.96, 0))
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

	rows.add_child(Brand.header("Make something real from your photos"))

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

	_kind = OptionButton.new()
	_kind.add_item("let the AI decide")
	_kind.add_item("a building")
	_kind.add_item("an object (a prop)")
	_kind.selected = int(_remembered("kind", 0))	# (as you chose last time)
	_kind.item_selected.connect(func(i: int): _remember("kind", i))
	rows.add_child(_row("What is it?", _kind))
	rows.add_child(_wrapped("(Saying so skips the AI's look at each photo: much quicker.)", 13))

	_where = OptionButton.new()
	_where.add_item("where the photos were taken")
	_where.add_item("where I'm looking")
	_where.add_item("nowhere: just add it to My things")
	_where.selected = int(_remembered("where", 0))	# (as you chose last time)
	_where.item_selected.connect(func(i: int): _remember("where", i))
	rows.add_child(_row("Put objects", _where))
	rows.add_child(_wrapped("(Buildings always go on their real spot, in place of the map's box.)", 13))

	_paid = CheckBox.new()
	_paid.text = "Allow paid Meta calls (SAM 3.1, Muse Image): needed for new photos"
	_paid.button_pressed = _remembered("paid", false)	# (off till you tick it; then it stays as you left it)
	_paid.toggled.connect(func(on: bool): _remember("paid", on))
	rows.add_child(_paid)

	var buttons := HBoxContainer.new()
	buttons.alignment = BoxContainer.ALIGNMENT_END
	var close := Button.new()
	close.text = "Close (F7)"
	close.pressed.connect(func(): _show(false))
	_make = Button.new()
	_make.text = "Make it"
	_make.pressed.connect(_start)
	_cancel_button = Button.new()
	_cancel_button.text = "Cancel the make"
	_cancel_button.visible = false
	_cancel_button.pressed.connect(_cancel)
	buttons.add_child(_cancel_button)
	buttons.add_child(close)
	buttons.add_child(_make)
	rows.add_child(buttons)

	rows.add_child(HSeparator.new())
	_result = _wrapped("", 14)
	rows.add_child(_result)
	_log_box = RichTextLabel.new()	# the live log: every step, with the time it came
	_log_box.custom_minimum_size = Vector2(0, 150)
	_log_box.scroll_following = true
	_log_box.selection_enabled = true
	_log_box.add_theme_font_size_override("normal_font_size", 13)
	_log_box.visible = false
	rows.add_child(_log_box)
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


# A label and its control side by side. The label doesn't wrap (in a row a wrapping label has no width, so it went
# one letter per line); the control takes the rest of the row.
func _row(text: String, control: Control) -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 10)
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 15)
	label.custom_minimum_size = Vector2(110, 0)
	row.add_child(label)
	control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(control)
	return row


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
		_result.text = "Working on %s:\n%s" % [job["name"], _line.text]
	elif not last.is_empty():
		_result.text = _told(last)
	_go.visible = not running and last.get("items", []).any(func(i): return i.get("lat") != null)
	_cancel_button.visible = running
	_log_box.visible = not _log_lines.is_empty()
	_log_box.text = "\n".join(_log_lines)


# A line in the make's log (the panel's log box, the card's last lines, and the game's own log).
func _log(text: String) -> void:
	var secs := 0
	if not job.is_empty():
		secs = (Time.get_ticks_msec() - int(job["started"])) / 1000
	var line := "[%d:%02d] %s" % [secs / 60, secs % 60, text]
	_log_lines.append(line)
	if _log_lines.size() > LOG_KEPT:
		_log_lines = _log_lines.slice(_log_lines.size() - LOG_KEPT)
	print("PlaceLens: " + line)
	if _panel.visible:
		_refresh()


func _choose_photos() -> void:
	var start := photos_dir if photos_dir != "" and DirAccess.dir_exists_absolute(photos_dir) \
		else str(_remembered("last_photos", ""))
	if start != "" and DirAccess.dir_exists_absolute(start):
		_dialog.current_dir = start
	_dialog.popup_centered_ratio(0.6)


func _photos_chosen(paths: PackedStringArray) -> void:
	_photos = paths
	if not paths.is_empty():
		_remember("last_photos", paths[0].get_base_dir())
	var names := PackedStringArray()
	for p in paths:
		names.append(p.get_file())
	_photo_list.text = "%d photo(s): %s" % [paths.size(), ", ".join(names)]


# --- making one ---

func _start() -> void:
	if _photos.is_empty():
		_result.text = "Choose a photo or more first."
		return
	if not _paid.button_pressed:	# (new photos always need SAM 3.1: ask now, not fail in a minute and a half)
		_ask_paid()
		return
	var name := "game-" + Time.get_datetime_string_from_system().replace("-", "").replace(":", "").replace("T", "-")
	var dir := home.path_join("out").path_join("game").path_join(name)
	DirAccess.make_dir_recursive_absolute(dir)
	for p in _photos:
		DirAccess.copy_absolute(p, dir.path_join(p.get_file()))
	var progress := dir.path_join("progress.jsonl")
	var saved := ProjectSettings.globalize_path(GameSettings.saved_dir().path_join("placelens"))
	var args := PackedStringArray(["-m", "placelens.cli", "-C", home, "survey", dir, "--progress", progress,
		"--pack-to", saved, "--parent-pid", str(OS.get_process_id())])	# (it stops itself if the game goes)
	var notes := _notes.text.strip_edges()
	if notes != "":
		var f := FileAccess.open(dir.path_join("notes.txt"), FileAccess.WRITE)
		f.store_string(notes)
		f.close()
		args.append_array(["--notes-file", dir.path_join("notes.txt")])
	if KINDS[_kind.selected] != "":
		args.append_array(["--kind", KINDS[_kind.selected]])
	OS.set_environment("PLACELENS_META_OFF", "0" if _paid.button_pressed else "1")	# (the child inherits it)
	var pid := OS.create_process(python, args)
	if pid <= 0:
		_result.text = "Couldn't start PlaceLens (%s). Check placelens.cfg." % python
		return
	job = {"name": name, "dir": dir, "pid": pid, "progress": progress, "read": 0, "ended": false,
		"spot": _aim_spot() if _where.selected == 1 else ({"nowhere": true} if _where.selected == 2 else {}), "started": Time.get_ticks_msec(), "step": 0,
		"doing": "starting the agent with %d photo(s)" % _photos.size(), "detail": "", "done": 0, "total": 0}
	last = {}
	_line_left = 0.0
	_log_lines = PackedStringArray()
	_log("started with %d photo(s)%s" % [_photos.size(),
		(", as " + _kind.get_item_text(_kind.selected)) if _kind.selected > 0 else ""])
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
			_card.visible = false
	if job.is_empty():
		return
	_show_progress()
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
	if event.get("done") is bool and event["done"]:	# (the end line only: true, never a count)
		_finish(event)
	elif event.has("detail"):	# a step inside a tool ("cutting out the bear in photo 3 of 9")
		job["detail"] = str(event["detail"])
		job["done"] = int(event.get("count", 0))
		job["total"] = int(event.get("total", 0))
		_log("  " + job["detail"])
	elif event.has("ok"):
		if not event["ok"]:
			job["detail"] = "%s didn't work (%s); the agent carries on" % [event["tool"], event.get("error", "")]
			job["total"] = 0
			_log("  " + job["detail"])
		else:
			_log("  %s done" % event["tool"])
	else:	# the agent starts a tool
		var what: String = STEPS.get(event["tool"], event["tool"])
		job["step"] += 1
		job["doing"] = what + (" (%s)" % event["about"] if event.get("about", "") != "" else "")
		job["detail"] = ""
		job["total"] = 0
		_log("Step %d: %s" % [job["step"], job["doing"]])


# The line along the top while it works: spinner, step, what it's doing, time so far; the bar when a step counts.
func _show_progress() -> void:
	var secs := (Time.get_ticks_msec() - int(job["started"])) / 1000
	var spin: String = SPINNER[(Time.get_ticks_msec() / 250) % SPINNER.size()]
	var text := "PlaceLens %s   %d:%02d\n" % [spin, secs / 60, secs % 60]
	var recent := _log_lines.slice(maxi(_log_lines.size() - CARD_LOG, 0))	# the live log's last lines
	text += "\n".join(recent) if not recent.is_empty() else str(job["doing"])
	text += "\n(F7: the whole log, or cancel)"
	_line.text = text
	_card.visible = true
	var total: int = job["total"]
	_bar.visible = total > 0
	if total > 0:
		_bar.value = float(job["done"]) / total


func _finish(event: Dictionary) -> void:
	var name: String = job["name"]
	_log("finished" if str(event.get("error", "")) == "" else "stopped: %s" % event["error"])
	var spot: Dictionary = job["spot"]
	var secs := (Time.get_ticks_msec() - int(job["started"])) / 1000
	job = {}
	_bar.visible = false
	last = {"summary": str(event.get("summary", "")), "questions": event.get("questions", []),
		"error": str(event.get("error", "")), "items": [], "cost": event.get("cost", {}),
		"seconds": int(event.get("seconds", secs))}
	var pack = event.get("pack")
	if pack is String and pack != "":
		last["items"] = _place(pack, spot)
		packs.add_pack(pack)
	if last["error"] != "":
		_say("PlaceLens couldn't make it: %s" % last["error"], SHOWN_FOR)
		if not last["error"].begins_with("cancelled"):
			_toast("Couldn't make it", _why_short(last["error"]), false)
	elif last["items"].is_empty():
		_say("PlaceLens made nothing from %s (F7 for why)" % name, SHOWN_FOR)
		_toast("Nothing was made", _why_short(last["summary"]), false)
	else:
		var names := PackedStringArray(last["items"].map(func(i): return str(i.get("name", i["id"]))))
		_say("PlaceLens made %s (F7 for what it said)" % ", ".join(names), SHOWN_FOR)
		for item in last["items"]:
			_toast("%s is ready" % str(item.get("name", item["id"])).capitalize(),
				"In your inventory: press I" if item.get("kind") == "prop"
				else "Up on the map where it really is, and a copy to place is in your inventory (I)")
	if _panel.visible:
		_refresh()
	print("PlaceLens: finished %s; placed: %s\n%s" % [name, ", ".join(PackedStringArray(last["items"].map(
		func(i): return "%s at %s, %s" % [i.get("name", i["id"]), i["lat"], i["lon"]]))), _told(last)])
	if _quit_when_done:
		get_tree().quit()


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
		# (a prop goes where you were looking only if you asked; one with no spot of its own, a small thing or a
		# photo without GPS, waits in your things for you to put down)
		if item.get("kind") != "prop" or spot.is_empty():
			continue
		if spot.get("nowhere", false):	# (you asked: not in the world, only in My things, for you to put down)
			item["lat"] = null
			item["lon"] = null
			item["placed"] = "in your things only"
			moved += 1
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
	return items	# (all of them: a small thing has no map spot, but it was made)


func _told(end: Dictionary) -> String:
	var text := ""
	if end.get("error", "") != "":
		text += "It didn't finish: %s\n" % end["error"]
	if end.get("summary", "") != "":
		text += str(end["summary"]).strip_edges() + "\n"
	for q in end.get("questions", []):
		text += "Question: %s\n" % q
	var cost = end.get("cost", {})
	if cost is Dictionary and not cost.is_empty():
		var usd := float(cost.get("meta_usd", 0))
		text += "\nCost: US$%s on the paid Meta API (%d calls: SAM 3.1, Muse Image). " \
			% [("%.2f" % usd) if usd >= 0.1 or usd == 0.0 else ("%.4f" % usd), int(cost.get("meta_calls", 0))] \
			+ "Muse Spark Open: %d calls, %s tokens, on the free Baseten credits." \
			% [int(cost.get("open_calls", 0)), _thousands(int(cost.get("open_tokens", 0)))]
	var secs := int(end.get("seconds", 0))
	if secs > 0:
		text += " Took %d min %02d s." % [secs / 60, secs % 60]
	return text.strip_edges()


func _thousands(n: int) -> String:
	var s := str(n)
	var out := ""
	while s.length() > 3:
		out = "," + s.right(3) + out
		s = s.left(s.length() - 3)
	return s + out


func _go_there() -> void:
	var first: Dictionary = last["items"].filter(func(i): return i.get("lat") != null)[0]
	_show(false)
	if packs.main.has_method("teleport"):
		packs.main.teleport(float(first["lat"]), float(first["lon"]), str(first.get("name", "PlaceLens")))


# A notification in the top left corner, as games do when something's ready: it slides in, stays a few seconds
# and fades. Several stack downwards.
# Make it, with paid calls off: they're needed (SAM 3.1 cuts the thing out of each new photo), so ask once; yes ticks
# the box (remembered) and makes it.
func _ask_paid() -> void:
	var ask := ConfirmationDialog.new()
	ask.title = "Allow paid Meta calls?"
	ask.dialog_text = "Making it needs Meta's SAM 3.1 to cut it out of your photos: about a quarter of a US cent " \
		+ "a photo (a walk round uses up to 6), from your Meta credits.\n\nAllow it? (Remembered: you won't be " \
		+ "asked again.)"
	ask.ok_button_text = "Allow and make it"
	ask.process_mode = Node.PROCESS_MODE_ALWAYS
	ask.confirmed.connect(func():
		_paid.button_pressed = true
		_start())
	ask.visibility_changed.connect(func(): if not ask.visible: ask.queue_free())
	add_child(ask)
	ask.popup_centered()


# Why it failed, in a few words for a notification (the whole story is under F7)
func _why_short(text: String) -> String:
	if text.contains("paid Meta call") or text.contains("PLACELENS_META_OFF"):
		return "Paid Meta calls are off: tick the box under F7 (once; it's remembered)"
	var first := text.strip_edges().split("\n")[0]
	return (first.left(90) + "...") if first.length() > 90 else first + "  (F7 for more)"


const SETTINGS := "user://placelens.cfg"	# what you chose last time (the paid box)


func _remembered(key: String, otherwise):
	var cfg := ConfigFile.new()
	return cfg.get_value("panel", key, otherwise) if cfg.load(SETTINGS) == OK else otherwise


func _remember(key: String, value) -> void:
	var cfg := ConfigFile.new()
	cfg.load(SETTINGS)
	cfg.set_value("panel", key, value)
	cfg.save(SETTINGS)


func _toast(title: String, line: String, good := true) -> void:
	if _toasts == null:
		_toasts = VBoxContainer.new()
		_toasts.set_anchors_and_offsets_preset(Control.PRESET_TOP_LEFT, Control.PRESET_MODE_MINSIZE, 16)
		_toasts.add_theme_constant_override("separation", 8)
		_toasts.mouse_filter = Control.MOUSE_FILTER_IGNORE
		add_child(_toasts)
	var card := PanelContainer.new()
	var look := StyleBoxFlat.new()
	look.bg_color = Color(0.1, 0.32, 0.18, 0.92) if good else Color(0.4, 0.1, 0.1, 0.92)
	look.set_corner_radius_all(8)
	look.set_content_margin_all(12)
	look.border_width_left = 4
	look.border_color = Color(0.45, 0.95, 0.55) if good else Color(1.0, 0.45, 0.4)
	card.add_theme_stylebox_override("panel", look)
	card.custom_minimum_size = Vector2(300, 0)
	card.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var text := VBoxContainer.new()
	var tag := Label.new()
	tag.text = Brand.NAME
	tag.add_theme_font_size_override("font_size", 12)
	tag.add_theme_color_override("font_color", Brand.ACCENT)
	text.add_child(tag)
	var top := Label.new()
	top.text = title
	top.add_theme_font_size_override("font_size", 17)
	var under := Label.new()
	under.text = line
	under.add_theme_font_size_override("font_size", 13)
	under.modulate = Color(1, 1, 1, 0.85)
	under.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	under.custom_minimum_size = Vector2(300, 0)
	text.add_child(top)
	text.add_child(under)
	card.add_child(text)
	_toasts.add_child(card)
	card.modulate.a = 0.0
	var tween := card.create_tween()
	tween.tween_property(card, "modulate:a", 1.0, 0.25)
	tween.tween_interval(TOAST_FOR)
	tween.tween_property(card, "modulate:a", 0.0, 0.6)
	tween.tween_callback(card.queue_free)


# The line along the top: up for `seconds`, or (0) till the next one
func _say(text: String, seconds := 0.0) -> void:
	_line.text = text
	_card.visible = true
	_line_left = seconds
