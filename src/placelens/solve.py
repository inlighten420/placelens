"""Cameras from the photos themselves: photogrammetry's first step, helped by the phone's compass and Muse's size.

Photos taken walking round something share what stays still: the floor, the furniture, the object's own markings.
SIFT points matched between two photos give how the second camera turned and which way it moved (the essential
matrix). Fur and plain paint give few points, but the room around them usually gives plenty, so the whole photo is
matched. That needs the photos about 30-45° apart: further apart, too little looks the same.

Joined up, the pairs give every matched camera's turn relative to the others. "Up" is found without a level: people
hold a phone with its sideways axis level, so the sideways axes of all the photos lie flat and up is square to them.
The compass then turns the lot to the map, and Muse's size of the object sets how far each camera stood, with the
directions the cameras moved in fixing their distances relative to each other. A photo that matched none of the
others is left to the compass, as before.

Axes: as carve.py (metres, Y up, the first photo's side facing +Z), positions relative to the object's middle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RATIO = 0.75  # a match must be clearly better than the second best (Lowe's ratio test)
POINTS = 8000  # SIFT points per photo
MIN_INLIERS = 15  # matches agreeing with one camera move, for a pair to count
PIXEL_ERROR = 1.5  # pixels a match may be off its line and still agree
TILTED = 45.0  # degrees: a camera looking down more than this has a compass that means little (left out of turning)


@dataclass
class View:
    """One photo for solving: the picture, its lens, where the object's middle shows, and what the phone said."""

    image: np.ndarray  # uint8 RGB
    focal: float  # pixels
    centre: tuple[float, float]  # pixel of the object's middle (its cut-out's box centre)
    distance: float  # metres to the object's middle, from its size in the photo
    heading: float | None  # compass bearing the camera faced, degrees


@dataclass
class Pose:
    """A solved camera: position (relative to the object's middle) and its right, up and forward unit vectors."""

    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray


def _k(v: View) -> np.ndarray:
    h, w = v.image.shape[:2]
    return np.array([[v.focal, 0, w / 2], [0, v.focal, h / 2], [0, 0, 1]])


def features(views: list[View]) -> list[tuple]:
    import cv2

    sift = cv2.SIFT_create(nfeatures=POINTS)
    return [sift.detectAndCompute(cv2.cvtColor(v.image, cv2.COLOR_RGB2GRAY), None) for v in views]


def match(fa: tuple, fb: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Points matched both ways: each the other's best, and clearly better than the second best."""
    import cv2

    if fa[1] is None or fb[1] is None or len(fa[0]) < 2 or len(fb[0]) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def one_way(d1, d2) -> dict[int, int]:
        return {m.queryIdx: m.trainIdx for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < RATIO * n.distance}

    ab, ba = one_way(fa[1], fb[1]), one_way(fb[1], fa[1])
    pairs = [(i, j) for i, j in ab.items() if ba.get(j) == i]
    return (np.float64([fa[0][i].pt for i, _ in pairs]).reshape(-1, 2),
            np.float64([fb[0][j].pt for _, j in pairs]).reshape(-1, 2))


def relative(va: View, vb: View, pa: np.ndarray, pb: np.ndarray) -> tuple[int, np.ndarray, np.ndarray] | None:
    """How camera b sits relative to a: (agreeing matches, R, t) with x_b = R x_a + t in OpenCV camera axes (x right,
    y down, z forward); t is a direction only. None if too few matches agree."""
    import cv2

    if len(pa) < 8:
        return None
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2), _k(va), None).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2), _k(vb), None).reshape(-1, 2)
    E, inliers = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                      threshold=PIXEL_ERROR / math.sqrt(va.focal * vb.focal))
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=inliers)
    return (int(n), R, t.ravel()) if n >= MIN_INLIERS else None


def spanning_tree(n: int, edges: dict[tuple[int, int], tuple]) -> tuple[int, list[tuple[int, int]]]:
    """The strongest way to join the matched cameras (most agreeing matches first): its root (the camera with the
    most) and its edges in an order where each joins a camera already placed."""
    strength = [sum(e[0] for (i, j), e in edges.items() if k in (i, j)) for k in range(n)]
    root = int(np.argmax(strength))
    placed, order = {root}, []
    while True:
        joining = [(e[0], i, j) for (i, j), e in edges.items() if (i in placed) != (j in placed)]
        if not joining:
            return root, order
        _, i, j = max(joining)
        order.append((i, j))
        placed |= {i, j}


def _turn_y(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def bearing_of(forward: np.ndarray) -> float:
    """A direction's bearing in the carve frame, relative to the first photo's: 0 looking along -Z, turning right +."""
    return math.degrees(math.atan2(forward[0], -forward[2]))


def solve(views: list[View], first_bearing: float) -> tuple[dict[int, Pose], list[str]]:
    """Cameras for the views that match each other, in the carve frame (`first_bearing` = the compass bearing that
    becomes 0). Returns {view index: pose} and a line about each pair joined."""
    feats = features(views)
    edges: dict[tuple[int, int], tuple] = {}
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            pa, pb = match(feats[i], feats[j])
            r = relative(views[i], views[j], pa, pb)
            if r is not None:
                edges[(i, j)] = r
    notes = [f"{i}-{j}: {e[0]} matches agree" for (i, j), e in sorted(edges.items())]
    if not edges:
        return {}, notes
    root, order = spanning_tree(len(views), edges)

    # Turns: each camera's (camera -> world) rotation, the root's camera axes as the world for now.
    rot = {root: np.eye(3)}
    for i, j in order:
        if (i, j) in edges:
            _, R, _ = edges[(i, j)]
            a, b, R_ab = i, j, R
        else:
            _, R, _ = edges[(j, i)]
            a, b, R_ab = j, i, R
        if a in rot:
            rot[b] = rot[a] @ R_ab.T
        else:
            rot[a] = rot[b] @ R_ab

    # Up: square to every camera's sideways axis (phones are held level), on the side the photos' tops point to.
    rights = np.array([rot[k][:, 0] for k in rot])
    up = np.linalg.svd(rights)[2][-1]
    if sum(-rot[k][:, 1] @ up for k in rot) < 0:
        up = -up
    x = rights[0] - (rights[0] @ up) * up
    x /= np.linalg.norm(x)
    level = np.array([x, up, np.cross(x, up)])  # world-for-now -> Y up (rows: new axes)
    rot = {k: level @ R for k, R in rot.items()}

    # Compass: turn about up so the cameras face the bearings the phones said (the middle of their differences;
    # cameras looking steeply down are left out, their compasses wander).
    diffs = []
    for k, R in rot.items():
        forward = R[:, 2]
        if views[k].heading is not None and math.degrees(math.asin(min(1.0, abs(forward[1])))) < TILTED:
            wanted = (views[k].heading - first_bearing + 180) % 360 - 180
            diffs.append((wanted - bearing_of(forward) + 180) % 360 - 180)
    if diffs:
        turn = math.degrees(math.atan2(np.median(np.sin(np.radians(diffs))), np.median(np.cos(np.radians(diffs)))))
        rot = {k: _turn_y(-turn) @ R for k, R in rot.items()}

    # Distances: each camera stands back from the object's middle along its own sight line, as far as the object's
    # size says (loosely) and as the pairs' moves agree (firmly: where camera b went, seen from a).
    ks = sorted(rot)
    col = {k: n for n, k in enumerate(ks)}
    sight = {}
    for k in ks:
        v = views[k]
        h, w = v.image.shape[:2]
        ray = np.array([(v.centre[0] - w / 2) / v.focal, (v.centre[1] - h / 2) / v.focal, 1.0])
        sight[k] = rot[k] @ (ray / np.linalg.norm(ray))  # world direction from the camera to the object's middle
    rows = []
    for (i, j), (_, R, t) in edges.items():
        move = rot[i] @ (-R.T @ t)  # world direction camera i -> camera j
        move /= np.linalg.norm(move)
        # (position_j - position_i) x move = 0, with position_k = -d_k * sight_k: d_i (s_i x m) - d_j (s_j x m) = 0
        block = np.zeros((3, len(ks)))
        block[:, col[i]] = np.cross(sight[i], move)
        block[:, col[j]] = -np.cross(sight[j], move)
        rows.append(block)
    # The moves fix the distances' proportions only (all of them at 0 fits too): the proportions that fit best, then
    # scaled to the object's size as the level cameras see it.
    shape = np.linalg.svd(np.vstack(rows))[2][-1]
    shape = shape if shape.sum() > 0 else -shape
    level_ks = [k for k in ks if math.degrees(math.asin(min(1.0, abs(rot[k][1, 2])))) < TILTED]
    if (shape <= 0).any() or not level_ks:  # the moves don't hang together: go by size alone
        d = np.array([views[k].distance for k in ks])
    else:
        d = shape * float(np.median([views[k].distance / shape[col[k]] for k in level_ks]))
    notes.append("distances (m): " + ", ".join(f"{k}: {d[col[k]]:.2f} (size says {views[k].distance:.2f})"
                                                for k in ks))

    poses = {}
    for k in ks:
        R = rot[k]
        poses[k] = Pose(-d[col[k]] * sight[k], R[:, 0], -R[:, 1], R[:, 2])
    return poses, notes
