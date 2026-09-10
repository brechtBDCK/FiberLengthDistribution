from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure as skmeasure, morphology


NEIGHBORS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


@dataclass
class Fiber:
    id: str; centerline_length_px: float; endpoint_distance_px: float
    calibrated_length: float | None; orientation_endpoint_deg: float; orientation_pca_deg: float
    orientation_deg: float; tortuosity: float; mean_width_px: float; max_width_px: float
    centroid_yx: list[float]; endpoints_yx: list[list[float]]; confidence: float
    boundary_truncated: bool; rejected: bool = False; rejection_reason: str | None = None

def _neighbors(pixel, pixels):
    y, x = pixel
    return [(y + dy, x + dx) for dy, dx in NEIGHBORS if (y + dy, x + dx) in pixels]


def prune_spurs(skeleton: np.ndarray, maximum_length: int) -> np.ndarray:
    """Remove endpoint branches shorter than limit, retaining real long fibers."""
    result = skeleton.copy()
    for _ in range(maximum_length):
        graph = skeleton_graph(result)
        remove = []
        for endpoint in graph["endpoints"]:
            chain, previous, point = [endpoint], None, endpoint
            while len(chain) <= maximum_length:
                nexts = [n for n in graph["adjacency"][point] if n != previous]
                if len(nexts) != 1: break
                previous, point = point, nexts[0]; chain.append(point)
            if len(chain) <= maximum_length and len(graph["adjacency"][point]) >= 3: remove.extend(chain[:-1])
        if not remove: break
        result[tuple(np.asarray(remove).T)] = False
    return result


def skeleton_graph(skeleton: np.ndarray) -> dict:
    """Pixel graph. Degree 1=endpoints; degree >=3=junctions for debug/review."""
    pixels = set(map(tuple, np.argwhere(skeleton)))
    adjacency = {p: _neighbors(p, pixels) for p in pixels}
    endpoints = [p for p, n in adjacency.items() if len(n) == 1]
    junctions = [p for p, n in adjacency.items() if len(n) >= 3]
    # At crossings, opposite tangents belong together; nearby parallel paths never share a node.
    pairings = {p: _pair_junction_branches(p, adjacency[p]) for p in junctions}
    return {"adjacency": adjacency, "endpoints": endpoints, "junctions": junctions, "crossing_pairs": pairings}


def _pair_junction_branches(node, neighbors):
    """Greedy best tangent-continuity pairing at a skeleton crossing."""
    remaining = list(neighbors); pairs = []
    while len(remaining) > 1:
        a = remaining.pop(0); va = np.subtract(a, node)
        b = min(remaining, key=lambda q: np.dot(va, np.subtract(q, node)) / (np.linalg.norm(va)*np.linalg.norm(np.subtract(q, node))))
        remaining.remove(b); pairs.append((a, b))
    return pairs


def _furthest(adjacency, start):
    todo = [(0.0, start, None)]; seen = {}; farthest = (start, 0.0)
    while todo:
        distance, point, parent = heapq.heappop(todo)
        if point in seen: continue
        seen[point] = parent
        if distance > farthest[1]: farthest = point, distance
        for nxt in adjacency[point]:
            if nxt not in seen:
                heapq.heappush(todo, (distance + float(np.hypot(nxt[0]-point[0], nxt[1]-point[1])), nxt, point))
    path = []; point = farthest[0]
    while point is not None: path.append(point); point = seen[point]
    return path[::-1], farthest[1]


def _longest_endpoint_path(graph):
    """Fast two-sweep diameter approximation; avoids endpoint-pair O(E²) work."""
    endpoints = graph["endpoints"]
    if len(endpoints) < 2: return [], 0.0
    first, _ = _furthest(graph["adjacency"], endpoints[0])
    return _furthest(graph["adjacency"], first[-1])


def _orientation(points):
    a, b = np.asarray(points[0]), np.asarray(points[-1])
    endpoint = np.degrees(np.arctan2(b[0]-a[0], b[1]-a[1])) % 180
    xy = np.asarray(points)[:, ::-1]; _, _, vh = np.linalg.svd(xy - xy.mean(0), full_matrices=False)
    pca = np.degrees(np.arctan2(vh[0, 1], vh[0, 0])) % 180
    return float(endpoint), float(pca)


def local_orientation_distribution(points, window: int = 9) -> list[float]:
    """Axial tangent angles along a curved centerline."""
    p = np.asarray(points); return [float(np.degrees(np.arctan2(*(p[min(i+window, len(p)-1)]-p[max(i-window, 0)]))) % 180) for i in range(len(p))]


def measure(mask: np.ndarray, response: np.ndarray, cfg: dict, units_per_pixel: float | None = None, source_pixel_scale: float = 1, source_origin_yx=(0, 0)) -> tuple[list[Fiber], np.ndarray, dict]:
    labels = skmeasure.label(mask, connectivity=2); fibers = []; skeleton_all = np.zeros_like(mask)
    # EDT is O(image) but was previously recomputed O(number of components).
    width_map = 2 * distance_transform_edt(mask)
    for region in skmeasure.regionprops(labels, intensity_image=response):
        component = labels == region.label
        skeleton = prune_spurs(morphology.skeletonize(component), cfg["postprocess"]["spur_length_px"])
        skeleton_all |= skeleton
        graph = skeleton_graph(skeleton)
        path, length = _longest_endpoint_path(graph)
        minor = max(region.axis_minor_length, 1e-6); aspect = region.axis_major_length / minor
        circularity = 4*np.pi*region.area / max(region.perimeter**2, 1e-6)
        reason = None
        if region.area < cfg["postprocess"]["min_component_area"]: reason = "small_component"
        elif aspect < 2.0 and circularity > 0.5: reason = "round_speck"
        elif length < cfg["postprocess"]["min_length_px"]: reason = "short_skeleton"
        if not path: reason = reason or "no_endpoint_path"; path = [tuple(map(int, region.coords[0]))] * 2; length = 0
        endpoint, pca = _orientation(path); a, b = np.asarray(path[0]), np.asarray(path[-1]); distance = float(np.linalg.norm(b-a)) * source_pixel_scale
        length *= source_pixel_scale; widths = width_map[skeleton] * source_pixel_scale
        boundary = any(y in (0, mask.shape[0]-1) or x in (0, mask.shape[1]-1) for y, x in path)
        confidence = float(np.clip(0.55*(region.intensity_mean/(np.max(response)+1e-6)) + .25*min(1, aspect/8) + .2*min(1, length/100), 0, 1))
        centroid = (np.asarray(region.centroid) * source_pixel_scale + source_origin_yx).tolist(); endpoints = [(np.asarray(p) * source_pixel_scale + source_origin_yx).tolist() for p in (path[0], path[-1])]
        fibers.append(Fiber(f"fiber_{len(fibers)+1:05d}", length, distance, length*units_per_pixel if units_per_pixel else None, endpoint, pca, pca, length/max(distance, 1e-6), float(widths.mean()) if len(widths) else 0, float(widths.max()) if len(widths) else 0, centroid, endpoints, confidence, boundary, bool(reason), reason))
    return fibers, skeleton_all, {"labels": labels}
