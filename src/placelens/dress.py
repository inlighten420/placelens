"""A prop built, not pasted: the photos say what its parts are and what they're made of, and the model is drawn clean.

A photo pasted on a model brings its shadows, the floor and the blur along. Instead:

1. Colour families: the object's pixels (inside SAM's cut-outs) grouped into a few colours (k-means in Lab, where
   distances look like differences to the eye): for a teddy, blue fur, grey paw pads, black eyes and nose.
2. Each spot on the carved surface takes the family the photos facing it agree on (a vote, so a shadow or a glint
   in one photo is outvoted), tidied so lone specks join their neighbours.
3. Muse looks at a photo and the families (a swatch picture, numbered) and says which part each is and what it's
   made of: "body, plush fur", "eyes, glossy plastic"; families that are one part in shade are joined.
4. Each part gets one clean colour (its family's middle, so without the shadows), Muse's roughness and shine, and a
   grain made here (fur, wood, stone, fabric) so it reads as that material up close. Small parts Muse calls
   features (eyes, nose, buttons) are made real: a little dome standing on the surface.

Axes as carve.py.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np

from .build import Part

FAMILIES = 5  # colour families looked for (Muse joins the ones that are one part)
SAMPLE = 60000  # pixels sampled from the cut-outs to find the families
SMOOTH_PASSES = 3  # rounds of "take your neighbours' family" to clear specks
FEATURE_MAX = 0.01  # a feature patch bigger than this share of the surface isn't one: it's shade (a teddy's eye is
# ~0.001, its nose ~0.003)
FEATURE_MIN = 0.0003  # ...and one smaller than this is a speck
AGREE = 0.6  # a spot is a part other than the main one only if this share of the photos' votes there says so (lighting
# changes from photo to photo as you walk round; a part's own colour doesn't)
LIGHTNESS = 0.4  # how much lightness counts against colour when sorting pixels: shade is darker, not another colour
DOME_RISE = 0.45  # a feature's dome stands this share of its radius out of the surface
GRAINS = ("fur", "fabric", "wood", "stone", "metal", "smooth")

MATERIAL_PROMPT = """Picture 1 is a photo of {name}. Picture 2 shows the colours found on it, numbered {numbers}.

For each numbered colour, say which part of the {name} it is and what that part is made of. Colours that are the \
same part (one in shade) get the same part name. A small detail like an eye, nose, button or bolt is a feature.

Answer in JSON only:
{{"families": [{{"number": 1, "part": "short name, e.g. body", "material": "e.g. plush fur", \
"grain": one of {grains}, "roughness": 0.0 (mirror) to 1.0 (matt), "metallic": 0.0 to 1.0, \
"feature": true if it's a small detail}}, ...]}}"""


@dataclass
class Family:
    number: int  # 1-based, as on the swatch
    rgb: np.ndarray  # 0..1, the family's middle colour
    share: float  # of the sampled pixels


@dataclass
class Material:
    part: str
    material: str = ""
    grain: str = "smooth"
    roughness: float = 0.8
    metallic: float = 0.0
    feature: bool = False


def families(shots, k: int = FAMILIES, seed: int = 0) -> list[Family]:
    """The object's main colours, from the real photos' cut-outs (drawings left out: their colours are invented)."""
    from scipy.cluster.vq import kmeans2
    from skimage.color import lab2rgb, rgb2lab

    rng = np.random.default_rng(seed)
    pixels = np.concatenate([s.image[s.mask] for s in shots if not s.imagined])
    pixels = pixels[rng.choice(len(pixels), min(SAMPLE, len(pixels)), replace=False)]
    lab = rgb2lab(pixels[None].astype(float) / 255)[0] * [LIGHTNESS, 1, 1]
    centres, labels = kmeans2(lab, k, seed=seed, minit="++")
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    rgb = lab2rgb((centres[order] / [LIGHTNESS, 1, 1])[None])[0]
    return [Family(n + 1, rgb[n], counts[order[n]] / counts.sum()) for n in range(k) if counts[order[n]] > 0]


def _nearest(pixels: np.ndarray, fams: list[Family]) -> np.ndarray:
    from skimage.color import rgb2lab

    lab = rgb2lab(pixels[None].astype(float) / 255)[0] * [LIGHTNESS, 1, 1]
    centres = rgb2lab(np.array([f.rgb for f in fams])[None])[0] * [LIGHTNESS, 1, 1]
    return np.argmin(((lab[:, None] - centres[None]) ** 2).sum(-1), axis=1)


def label_vertices(verts: np.ndarray, normals: np.ndarray, shots, height: float, fams: list[Family]) -> np.ndarray:
    """Each vertex's family (an index into `fams`): a vote of the photos facing it, the squarest counting most; the
    main family unless the votes clearly agree on another. Spots no photo sees take their nearest seen neighbour's."""
    votes = np.zeros((len(verts), len(fams)))
    for s in shots:
        if s.imagined:
            continue
        position = s.camera(height / 2)[0]
        towards = position - verts
        towards /= np.linalg.norm(towards, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", normals, towards)
        cols, rows, depth = s.project(verts, height / 2)
        h, w = s.mask.shape
        c, r = np.round(cols).astype(int), np.round(rows).astype(int)
        ok = (facing > 0.2) & (depth > 0) & (c >= 0) & (c < w) & (r >= 0) & (r < h)
        ok[ok] &= s.mask[r[ok], c[ok]]
        idx = np.flatnonzero(ok)
        fam = _nearest(s.image[r[idx], c[idx]], fams)
        np.add.at(votes, (idx, fam), facing[idx] ** 2)
    labels = votes.argmax(axis=1)
    total = votes.sum(axis=1)
    main = int(np.argmax([f.share for f in fams]))
    unsure = votes[np.arange(len(verts)), labels] < AGREE * np.maximum(total, 1e-12)
    labels[unsure] = main  # (the photos don't agree it's anything else: the main colour, lit differently)
    seen = total > 0
    if seen.any() and not seen.all():
        from scipy.spatial import cKDTree

        _, near = cKDTree(verts[seen]).query(verts[~seen])
        labels[~seen] = labels[seen][near]
    return labels


def smooth_labels(faces: np.ndarray, labels: np.ndarray, passes: int = SMOOTH_PASSES) -> np.ndarray:
    """Each vertex takes the family most of its neighbours (and itself) have, a few times: specks go, edges stay."""
    from scipy import sparse

    n = len(labels)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = sparse.coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]],
                                                       np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    adj = adj + sparse.identity(n, format="csr")
    k = labels.max() + 1
    for _ in range(passes):
        onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, k))
        labels = np.asarray((adj @ onehot).argmax(axis=1)).ravel()
    return labels


def swatch(fams: list[Family], path) -> None:
    """The families as numbered squares, for Muse."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160 * len(fams), 200), "white")
    draw = ImageDraw.Draw(img)
    for i, f in enumerate(fams):
        draw.rectangle((160 * i + 10, 10, 160 * i + 150, 150), fill=tuple(int(v * 255) for v in f.rgb))
        draw.text((160 * i + 70, 165), str(f.number), fill=(0, 0, 0))
    img.save(path)


def parse_materials(text: str) -> dict[int, Material]:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    out = {}
    for f in data.get("families", []) if isinstance(data, dict) else []:
        try:
            grain = str(f.get("grain", "smooth")).lower()
            out[int(f["number"])] = Material(
                str(f.get("part") or "part").strip().lower(), str(f.get("material") or ""),
                grain if grain in GRAINS else "smooth", float(np.clip(float(f.get("roughness", 0.8)), 0, 1)),
                float(np.clip(float(f.get("metallic", 0.0)), 0, 1)), bool(f.get("feature", False)))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def merge_materials(answers: list[dict[int, Material]]) -> dict[int, Material]:
    """Several answers to one: per family, the most given part name, grain and feature; the middle numbers."""
    from .muse import majority

    out = {}
    for n in sorted({k for a in answers for k in a}):
        given = [a[n] for a in answers if n in a]
        part, _ = majority([m.part for m in given])
        agree = [m for m in given if m.part == part]
        out[n] = Material(part, agree[0].material, majority([m.grain for m in agree])[0],
                          float(np.median([m.roughness for m in agree])), float(np.median([m.metallic for m in agree])),
                          bool(majority([m.feature for m in agree])[0]))
    return out


def grain(verts: np.ndarray, kind: str, size: float, seed: int = 0) -> np.ndarray:
    """A brightness factor per vertex (about 1) that reads as the material: soft speckle for fur, finer for fabric,
    bands for wood, blotches for stone, faint for metal, none for smooth."""
    if kind == "smooth":
        return np.ones(len(verts))
    rng = np.random.default_rng(seed)
    scale = {"fur": 0.012, "fabric": 0.006, "wood": 0.02, "stone": 0.04, "metal": 0.05}[kind] * max(size, 0.2) / 0.6
    strength = {"fur": 0.10, "fabric": 0.05, "wood": 0.12, "stone": 0.10, "metal": 0.03}[kind]
    if kind == "wood":  # bands across the object's longest way, a little wavy
        axis = np.argmax(verts.max(axis=0) - verts.min(axis=0))
        other = verts[:, (axis + 1) % 3]
        t = other / scale + 0.8 * np.sin(verts[:, axis] / (scale * 6))
        return 1 + strength * np.sin(t * math.pi)
    waves = np.zeros(len(verts))
    for _ in range(12):  # random waves summed: smooth noise at about `scale`
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        waves += np.sin(verts @ d / scale * rng.uniform(0.7, 1.4) + rng.uniform(0, 2 * math.pi))
    waves /= math.sqrt(12)
    speck = rng.normal(0, 0.5, len(verts)) if kind in ("fur", "fabric") else 0.0
    return 1 + strength * np.clip(waves + speck, -2.5, 2.5) / 2


def _dome(centre: np.ndarray, normal: np.ndarray, radius: float, rows: int = 6, cols: int = 12):
    """A low dome (a squashed half-sphere) standing on the surface at `centre`, facing `normal`."""
    a = np.cross(normal, [0.0, 1.0, 0.0] if abs(normal[1]) < 0.9 else [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    pos, nrm = [], []
    for i in range(rows + 1):
        phi = (math.pi / 2) * i / rows
        for j in range(cols):
            th = 2 * math.pi * j / cols
            d = math.cos(phi) * (math.cos(th) * a + math.sin(th) * b)
            pos.append(centre + radius * d + radius * DOME_RISE * math.sin(phi) * normal - 0.2 * radius * normal)
            n = math.cos(phi) * (math.cos(th) * a + math.sin(th) * b) + math.sin(phi) * normal
            nrm.append(n / np.linalg.norm(n))
    idx = []
    for i in range(rows):
        for j in range(cols):
            p, q = i * cols + j, i * cols + (j + 1) % cols
            idx += [p, q, p + cols, q, q + cols, p + cols]
    pos, nrm, idx = np.array(pos), np.array(nrm), np.array(idx).reshape(-1, 3)
    tri = pos[idx]
    if (np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * nrm[idx].mean(axis=1)).sum() < 0:
        idx = idx[:, ::-1]  # (wound to face out, or a game hides it)
    return pos, nrm, idx.ravel()


def _patches(faces: np.ndarray, ids: np.ndarray, n_verts: int) -> np.ndarray:
    """Which joined-up patch each of the faces `ids` is in (faces sharing a corner are joined)."""
    from scipy import sparse

    shared = sparse.coo_matrix((np.ones(len(ids) * 3), (np.repeat(np.arange(len(ids)), 3), faces[ids].ravel())),
                               shape=(len(ids), n_verts)).tocsr()
    return sparse.csgraph.connected_components(shared @ shared.T, directed=False)[1]


def _settle(faces: np.ndarray, fpart: np.ndarray, area: np.ndarray, features: set[int], n_verts: int) -> np.ndarray:
    """Patches that can't be what they're labelled go to the part around them: a feature's patch too big to be a
    feature (shade), and any patch too small to be anything (a speck)."""
    from scipy import sparse

    total = area.sum()
    fpart = fpart.copy()
    vert_faces = sparse.coo_matrix((np.ones(len(faces) * 3), (faces.ravel(), np.repeat(np.arange(len(faces)), 3))),
                                   shape=(n_verts, len(faces))).tocsr()
    for p in np.unique(fpart):
        ids = np.flatnonzero(fpart == p)
        comp = _patches(faces, ids, n_verts)
        for c in np.unique(comp):
            patch = ids[comp == c]
            share = area[patch].sum() / total
            if share >= FEATURE_MIN and not (p in features and share > FEATURE_MAX):
                continue
            around = vert_faces[np.unique(faces[patch])].indices  # faces touching the patch
            others = fpart[around][fpart[around] != p]
            if len(others):
                fpart[patch] = np.bincount(others).argmax()
    return fpart


def dress(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray, labels: np.ndarray, fams: list[Family],
          materials: dict[int, Material]) -> list[Part]:
    """The parts: the surface split by part (families of one part joined), each one clean colour with its grain and
    material; features also get domes."""
    size = float((verts.max(axis=0) - verts.min(axis=0)).max())
    part_of = [materials.get(f.number, Material(f"part {f.number}")).part for f in fams]
    names = list(dict.fromkeys(part_of))
    vpart = np.array([names.index(part_of[i]) for i in labels])
    fpart = np.array([np.bincount(vpart[f], minlength=len(names)).argmax() for f in faces])
    tri_area = 0.5 * np.linalg.norm(np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                                             verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)
    feature_parts = {names.index(materials[f.number].part) for f in fams
                     if f.number in materials and materials[f.number].feature}
    fpart = _settle(faces, fpart, tri_area, feature_parts, len(verts))
    parts = []
    for p, name in enumerate(names):
        members = [i for i, x in enumerate(part_of) if x == name]
        weights = np.array([fams[i].share for i in members])
        colour = (np.array([fams[i].rgb for i in members]) * weights[:, None]).sum(0) / weights.sum()
        mat = next((materials[fams[i].number] for i in members if fams[i].number in materials), Material(name))
        sel = faces[fpart == p]
        if len(sel) == 0:
            continue
        used, local = np.unique(sel, return_inverse=True)
        shade = grain(verts[used], mat.grain, size, seed=p)
        cols = np.clip(colour[None] * shade[:, None], 0, 1).astype(np.float32)
        parts.append(Part(name, verts[used].astype(np.float32), normals[used].astype(np.float32),
                          np.zeros((len(used), 2), np.float32), local.reshape(-1).astype(np.uint32), (1, 1, 1, 1),
                          colors=cols, roughness=mat.roughness, metallic=mat.metallic))
        if not mat.feature:
            continue
        # Features: each separate patch of this part that's small gets a dome.
        ids = np.flatnonzero(fpart == p)
        comp = _patches(faces, ids, len(verts))
        pos_all, nrm_all, idx_all = [], [], []
        for c in np.unique(comp):
            patch = ids[comp == c]
            area = tri_area[patch].sum()
            centre = (verts[faces[patch]].mean(axis=1) * tri_area[patch, None]).sum(0) / area
            normal = normals[faces[patch]].mean(axis=1).sum(0)
            normal /= np.linalg.norm(normal) + 1e-12
            pos, nrm, idx = _dome(centre, normal, math.sqrt(area / math.pi))
            idx_all.append(idx + sum(len(x) for x in pos_all))
            pos_all.append(pos)
            nrm_all.append(nrm)
        if pos_all:
            pos = np.concatenate(pos_all)
            parts.append(Part(f"{name} (made real)", pos.astype(np.float32), np.concatenate(nrm_all).astype(np.float32),
                              np.zeros((len(pos), 2), np.float32), np.concatenate(idx_all).astype(np.uint32),
                              (*colour, 1.0), roughness=min(mat.roughness, 0.25), metallic=mat.metallic))
    return parts
