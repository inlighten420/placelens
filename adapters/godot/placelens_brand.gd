# PlaceLens's own look in the game, so it reads as an add-on to Open-Planet and not part of it: one accent colour,
# a header with its name on every PlaceLens window, a top edge in that colour, and a small badge in the corner.
extends RefCounted

const NAME := "PlaceLens"
const ACCENT := Color(0.33, 0.78, 1.0)	# lens blue: everything PlaceLens shows carries it
const ADDON := "Add-on for Open-Planet: real things from your photos, made with Meta's AI"


# "PlaceLens  <what this window is>  [close]" with the add-on line under it.
static func header(what: String, close: Control = null) -> VBoxContainer:
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 2)
	var line := HBoxContainer.new()
	line.add_theme_constant_override("separation", 12)
	box.add_child(line)
	var name := Label.new()
	name.text = NAME
	name.add_theme_font_size_override("font_size", 26)
	name.add_theme_color_override("font_color", ACCENT)
	line.add_child(name)
	var title := Label.new()
	title.text = what
	title.add_theme_font_size_override("font_size", 20)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title.size_flags_vertical = Control.SIZE_SHRINK_END
	line.add_child(title)
	if close != null:
		line.add_child(close)
	var sub := Label.new()
	sub.text = ADDON
	sub.add_theme_font_size_override("font_size", 13)
	sub.modulate = Color(1, 1, 1, 0.65)
	box.add_child(sub)
	return box


# A PlaceLens window's background: dark, rounded, with the accent along its top.
static func panel_look(alpha := 0.96, margin := 18) -> StyleBoxFlat:
	var look := StyleBoxFlat.new()
	look.bg_color = Color(0.08, 0.09, 0.11, alpha)
	look.set_corner_radius_all(10)
	look.set_content_margin_all(margin)
	look.border_width_top = 3
	look.border_color = ACCENT
	return look


# The small tag in the top-right corner, always up while the game runs: says what's PlaceLens and its keys.
static func badge() -> PanelContainer:
	var card := PanelContainer.new()
	var look := StyleBoxFlat.new()
	look.bg_color = Color(0.08, 0.09, 0.11, 0.7)
	look.set_corner_radius_all(6)
	look.content_margin_left = 10
	look.content_margin_right = 10
	look.content_margin_top = 5
	look.content_margin_bottom = 5
	look.border_width_left = 3
	look.border_color = ACCENT
	card.add_theme_stylebox_override("panel", look)
	card.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var line := HBoxContainer.new()
	line.add_theme_constant_override("separation", 8)
	line.mouse_filter = Control.MOUSE_FILTER_IGNORE
	card.add_child(line)
	var name := Label.new()
	name.text = NAME
	name.add_theme_font_size_override("font_size", 15)
	name.add_theme_color_override("font_color", ACCENT)
	line.add_child(name)
	var keys := Label.new()
	keys.text = "add-on   I  my things   F7  make from photos"
	keys.add_theme_font_size_override("font_size", 13)
	keys.modulate = Color(1, 1, 1, 0.75)
	keys.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	line.add_child(keys)
	card.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT, Control.PRESET_MODE_MINSIZE, 16)
	card.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	return card
