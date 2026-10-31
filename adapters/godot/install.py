"""Put the PlaceLens adapter into an Open-Planet Godot project, or take it out again.

    python adapters/godot/install.py <path to Open-Planet's Godot folder> [--pack out/pack/home] [--uninstall]

Installing copies placelens_packs.gd and placelens_maker.gd (F7 in the game: make something from photos) to
<project>/placelens/, writes placelens.cfg there (where this PlaceLens and its Python are, for F7), and adds one
autoload line to project.godot. --pack
copies a place pack to where the game looks when run from the editor (<project>/../Saved/placelens/<name>); an
exported game looks in its user data folder instead (%APPDATA%/Godot/app_userdata/Geodrive/placelens).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
SCRIPT = "placelens_packs.gd"
SCRIPTS = (SCRIPT, "placelens_maker.gd")
LINE = 'PlaceLens="*res://placelens/placelens_packs.gd"'


def python() -> Path:
    """The Python with PlaceLens in it: the repo's .venv, else the one running this."""
    venv = REPO / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return venv if venv.exists() else Path(sys.executable)


def install(project: Path) -> None:
    (project / "placelens").mkdir(exist_ok=True)
    for script in SCRIPTS:
        shutil.copyfile(HERE / script, project / "placelens" / script)
    (project / "placelens" / "placelens.cfg").write_text(
        f'[placelens]\n\nhome="{REPO.resolve().as_posix()}"\npython="{python().resolve().as_posix()}"\n',
        encoding="utf-8")
    config = project / "project.godot"
    text = config.read_text(encoding="utf-8")
    if LINE in text:
        return
    if "[autoload]" in text:
        text = text.replace("[autoload]\n", f"[autoload]\n\n{LINE}\n", 1)
    else:
        text = text.rstrip("\n") + f"\n\n[autoload]\n\n{LINE}\n"
    config.write_text(text, encoding="utf-8", newline="\n")


def uninstall(project: Path) -> None:
    config = project / "project.godot"
    text = config.read_text(encoding="utf-8")
    text = text.replace(f"\n\n[autoload]\n\n{LINE}\n", "\n").replace(f"{LINE}\n", "")
    config.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
    shutil.rmtree(project / "placelens", ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("project", type=Path, help="the Open-Planet Godot folder (the one with project.godot)")
    p.add_argument("--pack", type=Path, action="append", default=[], help="a place pack to copy in for the editor")
    p.add_argument("--uninstall", action="store_true", help="take the adapter out again")
    args = p.parse_args(argv)
    if not (args.project / "project.godot").exists():
        sys.exit(f"{args.project} has no project.godot")
    if args.uninstall:
        uninstall(args.project)
        print(f"PlaceLens adapter taken out of {args.project}")
        return
    install(args.project)
    print(f"PlaceLens adapter in {args.project} (autoload PlaceLens)")
    for pack in args.pack:
        dest = args.project.resolve().parent / "Saved" / "placelens" / pack.name
        shutil.copytree(pack, dest, dirs_exist_ok=True)
        print(f"pack {pack.name} -> {dest}")


if __name__ == "__main__":
    main()
