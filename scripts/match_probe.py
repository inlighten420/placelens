"""Probe: can matching points between walk-around photos correct their cameras? (Step 1 of photogrammetry-lite.)

For each pair of photos: SIFT points inside SAM 3.1's cut-out (a little widened), matched both ways with a ratio
test, then the essential matrix (RANSAC) and the relative turn of the camera. That turn is compared with what the
compass said (the angle round the object `prop` uses). Prints a table; saves match pictures for the best pairs.

    python scripts/match_probe.py out/props/IMG_0177_walk9 private
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from placelens.meta import parse_segments
from placelens.photo import read_camera
from placelens.pick import pixel_to_offset

RATIO = 0.75  # Lowe's ratio test
WIDEN = 0.03  # the cut-out widened by this share of its height: edges of the object are where points are richest
MIN_INLIERS = 15
WHOLE = "--whole" in sys.argv  # match the whole photo (room and floor too), not just the object


def load(folder: Path, originals: Path) -> list[dict]:
    shots = []
    for jpg in sorted(folder.glob("IMG_*.jpg")):
        sam = next(folder.glob(f"{jpg.stem}_sam_*.txt"), None)
        segs = parse_segments(sam.read_text(encoding="utf-8")) if sam else []
        if not segs:
            continue
        img = np.asarray(Image.open(jpg).convert("RGB"))
        h, w = img.shape[:2]
        mask = segs[0].full_mask((w, h))
        rows = np.flatnonzero(mask.any(axis=1))
        grow = max(1, int(WIDEN * (rows[-1] - rows[0])))
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=grow)
        original = next(originals.glob(f"{jpg.stem}.*"), None)
        cam = read_camera(original) if original else None
        fov = cam.fov if cam else 70.0
        focal = w / 2 / math.tan(math.radians(fov / 2))
        left, _, right, _ = segs[0].box
        bearing = (cam.heading or 0.0) + pixel_to_offset((left + right) / 2, w, fov) if cam else None
        shots.append({"name": jpg.stem, "img": img, "mask": mask, "focal": focal, "fov": fov, "bearing": bearing,
                      "K": np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])})
    return shots


def features(shots: list[dict]) -> None:
    sift = cv2.SIFT_create(nfeatures=8000)
    for s in shots:
        gray = cv2.cvtColor(s["img"], cv2.COLOR_RGB2GRAY)
        s["kp"], s["desc"] = sift.detectAndCompute(gray, None if WHOLE else s["mask"])


def match(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Points matched both ways (each is the other's best, and clearly better than the second best)."""
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def one_way(d1, d2):
        good = {}
        for m, n in bf.knnMatch(d1, d2, k=2):
            if m.distance < RATIO * n.distance:
                good[m.queryIdx] = m.trainIdx
        return good

    ab, ba = one_way(a["desc"], b["desc"]), one_way(b["desc"], a["desc"])
    pairs = [(i, j) for i, j in ab.items() if ba.get(j) == i]
    pa = np.float64([a["kp"][i].pt for i, _ in pairs])
    pb = np.float64([b["kp"][j].pt for _, j in pairs])
    return pa, pb


def relative(a: dict, b: dict, pa: np.ndarray, pb: np.ndarray):
    """Camera b's turn relative to a (essential matrix on normalised points; the two lenses may differ)."""
    if len(pa) < 8:
        return None
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2), a["K"], None).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2), b["K"], None).reshape(-1, 2)
    thresh = 1.5 / math.sqrt(a["focal"] * b["focal"])
    E, inl = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=thresh)
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, inl2 = cv2.recoverPose(E, na, nb, np.eye(3), mask=inl)
    forward_b_in_a = R.T @ np.array([0.0, 0.0, 1.0])  # (x right, y down, z forward)
    yaw = math.degrees(math.atan2(forward_b_in_a[0], forward_b_in_a[2]))  # b's heading minus a's, right positive
    pitch = math.degrees(math.asin(-forward_b_in_a[1]))
    total = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return {"inliers": int(n), "yaw": yaw, "pitch": pitch, "turn": total, "R": R, "t": t.ravel(),
            "mask": inl2.ravel().astype(bool)}


def wrap(a: float) -> float:
    return (a + 180) % 360 - 180


def main() -> None:
    folder, originals = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path("out/experiments/match")
    out.mkdir(parents=True, exist_ok=True)
    shots = load(folder, originals)
    features(shots)
    print("photo     fov   points  compass bearing")
    for s in shots:
        print(f"{s['name']}  {s['fov']:5.1f}  {len(s['kp']):6d}  {s['bearing']:.0f}" if s["bearing"] is not None
              else f"{s['name']}  {s['fov']:5.1f}  {len(s['kp']):6d}  -")
    print()
    print("pair                 matched inliers  turn(match) heading(compass)  diff")
    results = []
    for a, b in combinations(shots, 2):
        pa, pb = match(a, b)
        r = relative(a, b, pa, pb)
        compass = wrap(b["bearing"] - a["bearing"]) if a["bearing"] is not None and b["bearing"] is not None else None
        if r is None or r["inliers"] < MIN_INLIERS:
            print(f"{a['name']}-{b['name']}  {len(pa):7d}  {r['inliers'] if r else 0:7d}  (too few)")
            continue
        diff = wrap(r["yaw"] - compass) if compass is not None else None
        print(f"{a['name']}-{b['name']}  {len(pa):7d}  {r['inliers']:7d}  yaw {r['yaw']:+6.1f} pitch {r['pitch']:+5.1f}"
              f"  {compass:+7.1f}  {diff:+6.1f}" if compass is not None else "")
        results.append((r["inliers"], a, b, pa, pb, r))
    for _, a, b, pa, pb, r in sorted(results, key=lambda x: -x[0])[:3]:
        ka = [cv2.KeyPoint(float(x), float(y), 4) for x, y in pa[r["mask"]]]
        kb = [cv2.KeyPoint(float(x), float(y), 4) for x, y in pb[r["mask"]]]
        ms = [cv2.DMatch(i, i, 0) for i in range(len(ka))]
        pic = cv2.drawMatches(cv2.cvtColor(a["img"], cv2.COLOR_RGB2BGR), ka, cv2.cvtColor(b["img"], cv2.COLOR_RGB2BGR),
                              kb, ms[::max(1, len(ms) // 80)], None, matchColor=(0, 255, 0), flags=2)
        path = out / f"match_{a['name']}_{b['name']}.jpg"
        cv2.imwrite(str(path), cv2.resize(pic, None, fx=0.4, fy=0.4))
        print(f"picture: {path}")


if __name__ == "__main__":
    main()
