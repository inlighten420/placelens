# The inventory (I, on foot): everything you can put in the world, in one place, as simple as Minecraft's.
#
#   My things          every prop there is (yours from photos first, with a picture of each, and the game's own);
#                      click one to hold it: the game's build mode opens with it, a click puts it down (R turns it)
#   Make from photos   PlaceLens's maker (F7): photos in, a 3D thing out; and what it's doing now
#
# Something just made shows "New" until you've looked. The game stands still while it's open, as for F7.
extends CanvasLayer

const Brand := preload("placelens_brand.gd")

const OPEN_KEY := KEY_I
const PROP_PREFIX := "placelens_"
const THUMB := Vector2i(176, 132)	# px: a prop's picture
const COLUMNS := 4

var packs: Node	# placelens_packs.gd
var maker: Node	# placelens_maker.gd
var fresh := {}	# prop ids made since the inventory was last looked at: "New"

var _panel: PanelContainer
var _tabs: HBoxContainer
var _page: VBoxContainer
var _tab := 0
var _thumbs := {}	# model path -> ImageTexture (made once)
var _was_mouse := Input.MOUSE_MODE_CAPTURED
var _was_paused := false


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	layer = 61
	_panel = PanelContainer.new()
	_panel.visible = false
	_panel.add_theme_stylebox_override("panel", Brand.panel_look())
	_panel.custom_minimum_size = Vector2(840, 560)
	_panel.set_anchors_and_offsets_preset(Control.PRESET_CENTER, Control.PRESET_MODE_MINSIZE)
	_panel.grow_horizontal = Control.GROW_DIRECTION_BOTH
	_panel.grow_vertical = Control.GROW_DIRECTION_BOTH
	add_child(_panel)
	var rows := VBoxContainer.new()
	rows.add_theme_constant_override("separation", 14)
	_panel.add_child(rows)
	var close := Button.new()
	close.text = "Close (I)"
	close.pressed.connect(func(): show_it(false))
	rows.add_child(Brand.header("Inventory", close))
	_tabs = HBoxContainer.new()
	_tabs.add_theme_constant_override("separation", 6)
	rows.add_child(_tabs)
	for name in ["My things", "Make from photos"]:
		var tab := Button.new()
		tab.text = name
		tab.toggle_mode = true
		tab.custom_minimum_size = Vector2(170, 38)
		var index := _tabs.get_child_count()
		tab.pressed.connect(func(): _tab = index; _fill())
		_tabs.add_child(tab)
	rows.add_child(HSeparator.new())
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	rows.add_child(scroll)
	_page = VBoxContainer.new()
	_page.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_page.add_theme_constant_override("separation", 10)
	scroll.add_child(_page)


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey and event.pressed and not event.echo):
		return
	if _panel.visible and (event.keycode == OPEN_KEY or event.keycode == KEY_ESCAPE):
		show_it(false)
		get_viewport().set_input_as_handled()
	elif event.keycode == OPEN_KEY and _on_foot() and not _other_menu_open():
		show_it(true)
		get_viewport().set_input_as_handled()


func _on_foot() -> bool:	# (in a car, I is the engine)
	return packs.main != null and packs.main.get("walker") != null


func _other_menu_open() -> bool:
	var builder = packs.main.get("builder") if packs.main != null else null
	return (maker != null and maker._panel.visible) or (builder != null and builder.has_method("menu_open")
		and builder.menu_open())


func show_it(on: bool, tab := -1) -> void:
	if tab >= 0:
		_tab = tab
	if on == _panel.visible:
		if on:
			_fill()
		return
	if on:
		_was_mouse = Input.mouse_mode
		_was_paused = get_tree().paused
		var net = packs.main.get("net") if packs.main != null else null
		get_tree().paused = not (net != null and net.is_online())
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		_panel.visible = true
		_fill()
	else:
		_panel.visible = false
		get_tree().paused = _was_paused
		Input.mouse_mode = _was_mouse
		if _tab == 0:
			fresh.clear()	# (looked at: no longer new)


# Something just made: "New" on it, and a count on the tab
func made(prop_id: String) -> void:
	fresh[prop_id] = true
	if _panel.visible:
		_fill()


func _fill() -> void:
	for i in _tabs.get_child_count():
		var tab: Button = _tabs.get_child(i)
		tab.button_pressed = i == _tab
		if i == 0:
			tab.text = "My things" + ("  (%d new)" % fresh.size() if not fresh.is_empty() else "")
	for child in _page.get_children():
		child.queue_free()
	if _tab == 0:
		_fill_things()
	else:
		_fill_make()


# --- My things ---

func _keeper() -> Node:
	var builder = packs.main.get("builder") if packs.main != null else null
	var keeper = builder.get("props") if builder != null else null
	return keeper if keeper != null and keeper.has_method("add_kind") else null


func _fill_things() -> void:
	var keeper := _keeper()
	if keeper == null:
		_page.add_child(_note("This game has no build mode with props yet."))
		return
	var ids: Array = keeper.kinds.keys()
	ids.sort_custom(func(a, b):	# yours from photos first, newest made first; then the game's own by name
		var ours_a := String(a).begins_with(PROP_PREFIX)
		var ours_b := String(b).begins_with(PROP_PREFIX)
		if ours_a != ours_b:
			return ours_a
		if fresh.has(a) != fresh.has(b):
			return fresh.has(a)
		return String(keeper.kinds[a]["name"]) < String(keeper.kinds[b]["name"]))
	if ids.is_empty():
		_page.add_child(_note("Nothing yet. Make something from photos (the next tab), or in the workshop."))
		return
	_page.add_child(_note("Click a thing to hold it, then click where it goes. R turns it; B when you're done."))
	var grid := GridContainer.new()
	grid.columns = COLUMNS
	grid.add_theme_constant_override("h_separation", 12)
	grid.add_theme_constant_override("v_separation", 12)
	_page.add_child(grid)
	for id in ids:
		var cell := VBoxContainer.new()
		cell.add_child(_card(String(id), keeper.kinds[id]))
		if String(id).begins_with(PROP_PREFIX):	# (ours: made from photos, so ours to delete)
			var model := str(keeper.kinds[id].get("placelens_model", ""))
			var in_car: bool = model != "" and packs.car_wheel == model
			var wheel := Button.new()
			wheel.text = "In my car: take it out" if in_car else "Use as my car's steering wheel"
			wheel.add_theme_font_size_override("font_size", 12)
			wheel.pressed.connect(func():
				packs.set_car_wheel("" if in_car else model)
				_fill())
			cell.add_child(wheel)
			var bin := Button.new()
			bin.text = "Delete"
			bin.add_theme_font_size_override("font_size", 12)
			bin.pressed.connect(_ask_delete.bind(String(id), str(keeper.kinds[id].get("name", id))))
			cell.add_child(bin)
		grid.add_child(cell)


func _card(id: String, kind: Dictionary) -> Control:
	var button := Button.new()
	button.custom_minimum_size = Vector2(THUMB.x + 12, THUMB.y + 64)
	button.pressed.connect(func(): _hold(id))
	var stack := VBoxContainer.new()
	stack.mouse_filter = Control.MOUSE_FILTER_IGNORE
	stack.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 6)
	button.add_child(stack)
	var picture := TextureRect.new()
	picture.custom_minimum_size = Vector2(THUMB)
	picture.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	picture.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	picture.mouse_filter = Control.MOUSE_FILTER_IGNORE
	stack.add_child(picture)
	var model := str(kind.get("placelens_model", ""))
	if model != "":
		_picture_of(model, picture)
	var name := Label.new()
	name.text = str(kind.get("name", id)).trim_suffix(" (PlaceLens)")
	name.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	name.custom_minimum_size = Vector2(THUMB.x, 0)
	name.mouse_filter = Control.MOUSE_FILTER_IGNORE
	stack.add_child(name)
	var under := Label.new()
	under.add_theme_font_size_override("font_size", 12)
	under.modulate = Color(1, 1, 1, 0.7)
	under.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var parts: Array = kind.get("look", {}).get("parts", [])
	var size: Array = parts[0].get("size", []) if parts.size() == 1 else []
	under.text = ("New  -  " if fresh.has(id) else "") + ("PlaceLens, from photos" if model != "" else "Open-Planet prop") \
		+ ("  -  %.1f m tall" % float(size[1]) if size.size() == 3 and model != "" else "")
	if model != "":
		under.add_theme_color_override("font_color", Brand.ACCENT)
	if fresh.has(id):
		under.modulate = Color(0.5, 1.0, 0.6)
	stack.add_child(under)
	return button


# Delete one made from photos, after asking: out of My things, off the map, out of its pack for good
func _ask_delete(id: String, name: String) -> void:
	var ask := ConfirmationDialog.new()
	ask.title = "Delete it?"
	ask.dialog_text = "Delete %s?\n\nIt goes from My things and from the map (any you've placed too), for good. " \
		% name.trim_suffix(" (PlaceLens)") + "You can make it again from its photos."
	ask.ok_button_text = "Delete"
	ask.process_mode = Node.PROCESS_MODE_ALWAYS
	ask.confirmed.connect(func():
		packs.delete_thing(id)
		fresh.erase(id)
		_fill())
	ask.visibility_changed.connect(func(): if not ask.visible: ask.queue_free())
	add_child(ask)
	ask.popup_centered()


# Hold it: the game's build mode, this prop chosen (a click puts it down, as the game's own build menu does)
func _hold(id: String) -> void:
	var builder = packs.main.get("builder")
	show_it(false)
	if builder == null:
		return
	if not builder.active:
		builder.toggle()
	builder.props.chosen = id
	builder.props.turn = 0.0


# A picture of the model: drawn once off screen (a small scene: the model, a light, a camera framing it)
func _picture_of(path: String, into: TextureRect) -> void:
	if _thumbs.has(path):
		into.texture = _thumbs[path]
		return
	if DisplayServer.get_name() == "headless":
		return
	var model: Node3D = packs._model(path)
	if model == null:
		return
	var view := SubViewport.new()
	view.size = THUMB * 2
	view.own_world_3d = true
	view.transparent_bg = true
	view.render_target_update_mode = SubViewport.UPDATE_ONCE
	add_child(view)
	var holder := Node3D.new()
	view.add_child(holder)
	holder.add_child(model)
	var bounds := _bounds(model)
	var centre := bounds.get_center()
	var reach := maxf(bounds.size.length(), 0.2)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-40, 30, 0)
	view.add_child(light)
	var sky := WorldEnvironment.new()
	sky.environment = Environment.new()
	sky.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	sky.environment.ambient_light_color = Color(0.8, 0.8, 0.85)
	sky.environment.ambient_light_energy = 0.7
	view.add_child(sky)
	var camera := Camera3D.new()
	camera.fov = 35.0
	view.add_child(camera)
	camera.look_at_from_position(centre + Vector3(0.55, 0.35, 1.0).normalized() * reach * 1.7, centre)
	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw
	if not is_instance_valid(view):
		return
	var picture := ImageTexture.create_from_image(view.get_texture().get_image())
	_thumbs[path] = picture
	view.queue_free()
	if is_instance_valid(into):
		into.texture = picture


func _bounds(node: Node) -> AABB:
	var box := AABB()
	var first := true
	for mesh in node.find_children("*", "MeshInstance3D", true, false):
		var part: AABB = mesh.transform * mesh.get_aabb()
		box = part if first else box.merge(part)
		first = false
	return box


# --- Make from photos ---

func _fill_make() -> void:
	_page.add_child(_note("Take photos of something real (a walk round it, or one good photo) and PlaceLens makes it "
		+ "in 3D at its real size. Objects land in My things; buildings go up on the map where they really are."))
	var open := Button.new()
	open.text = "Make something from photos"
	open.custom_minimum_size = Vector2(320, 48)
	open.size_flags_horizontal = Control.SIZE_SHRINK_BEGIN
	open.pressed.connect(func():
		show_it(false)
		maker._show(true))
	_page.add_child(open)
	if maker != null and not maker.job.is_empty():
		_page.add_child(_note("Making now:\n" + maker._line.text))
		var cancel := Button.new()
		cancel.text = "Cancel it"
		cancel.size_flags_horizontal = Control.SIZE_SHRINK_BEGIN
		cancel.pressed.connect(func(): maker._cancel(); _fill())
		_page.add_child(cancel)
	elif maker != null and not maker._log_lines.is_empty():
		_page.add_child(_note("Last time:\n" + maker._told(maker.last)))


func _note(text: String) -> Label:
	var label := Label.new()
	label.text = text
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.custom_minimum_size = Vector2(780, 0)
	label.add_theme_font_size_override("font_size", 14)
	return label
