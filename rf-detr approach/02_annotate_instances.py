#!/usr/bin/env python3
"""Browser-based instance-segmentation annotator for thin fibers.

Runs a local Flask app (default http://localhost:7860) and writes:

  <dataset>/<split>/_annotation_working.json   # resumable work state
  <dataset>/<split>/_annotations.coco.json     # ONLY completed tiles

The working file stores annotations for every tile, including unfinished tiles and
an uncommitted draft. The COCO file intentionally contains only tiles that have
been explicitly marked COMPLETE, so unfinished tiles are never treated as
negative/background training examples.

Annotation modes:
  * polyline: click along the centerline of one fiber, then Commit instance.
              The line is rasterized using the selected thickness.
  * polygon:  click around the exact boundary, then Commit instance.

Examples:
  python 02_annotate_instances.py --dataset fiber_dataset --split train --thickness 4
  python 02_annotate_instances.py --dataset fiber_dataset --split valid --thickness 4

Open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Fiber instance annotator</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background:#111; color:#eee; }
  #top { display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding:8px 10px; background:#1c1c1c; border-bottom:1px solid #333; }
  button, select, input { background:#292929; color:#eee; border:1px solid #555; border-radius:5px; padding:7px 10px; }
  button:hover { background:#353535; cursor:pointer; }
  button.primary { background:#215a35; border-color:#398b55; }
  button.warn { background:#633; border-color:#955; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  #status { margin-left:auto; font-size:14px; white-space:nowrap; }
  #sub { padding:6px 10px; background:#171717; border-bottom:1px solid #2d2d2d; display:flex; gap:16px; flex-wrap:wrap; font-size:13px; }
  #viewer { position:relative; width:100vw; height:calc(100vh - 112px); min-height:420px; overflow:hidden; background:#080808; }
  canvas { display:block; width:100%; height:100%; cursor:crosshair; }
  #hint { position:absolute; left:12px; bottom:10px; background:rgba(0,0,0,.72); padding:7px 10px; border-radius:5px; font-size:12px; pointer-events:none; }
  .done { color:#7ee787; font-weight:600; }
  .open { color:#ffcc66; font-weight:600; }
  .kbd { border:1px solid #555; background:#222; border-radius:3px; padding:1px 4px; }
</style>
</head>
<body>
<div id="top">
  <button id="prevBtn">◀ Prev</button>
  <button id="nextBtn">Next ▶</button>
  <button id="fitBtn">Fit</button>

  <label>Mode
    <select id="mode">
      <option value="polyline">Polyline</option>
      <option value="polygon">Polygon</option>
    </select>
  </label>
  <label>Thickness
    <input id="thickness" type="number" min="1" max="100" step="1" style="width:70px" />
  </label>

  <button id="undoBtn">Undo point</button>
  <button id="clearBtn">Clear draft</button>
  <button id="commitBtn">Commit instance</button>
  <button id="deleteBtn" class="warn">Delete last instance</button>
  <button id="saveBtn">Save</button>
  <button id="completeBtn" class="primary">✓ Complete + next unfinished</button>
  <button id="reopenBtn">Reopen tile</button>
  <span id="status"></span>
</div>
<div id="sub">
  <span id="fileLabel"></span>
  <span id="tileState"></span>
  <span id="instanceCount"></span>
  <span id="progress"></span>
</div>
<div id="viewer">
  <canvas id="canvas"></canvas>
  <div id="hint">
    Left click: add point · Right click: undo · Mouse wheel: zoom · Space/middle-drag: pan ·
    <span class="kbd">Enter</span> commit · <span class="kbd">U</span> undo ·
    <span class="kbd">C</span> clear · <span class="kbd">D</span> delete last ·
    <span class="kbd">M</span> mode · <span class="kbd">[ ]</span> thickness ·
    <span class="kbd">N/P</span> next/prev · <span class="kbd">F</span> fit ·
    <span class="kbd">S</span> save
  </div>
</div>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const viewer = document.getElementById('viewer');
const img = new Image();

let tile = null;
let index = 0;
let total = 0;
let completedCount = 0;
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let panning = false;
let spaceDown = false;
let panLast = null;
let imageLoaded = false;
let busy = false;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function resizeCanvas() {
  canvas.width = Math.max(1, viewer.clientWidth);
  canvas.height = Math.max(1, viewer.clientHeight);
  if (imageLoaded) render();
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function fitImage() {
  if (!imageLoaded) return;
  const pad = 20;
  scale = Math.min((canvas.width - 2*pad) / img.naturalWidth,
                   (canvas.height - 2*pad) / img.naturalHeight);
  scale = Math.max(scale, 0.01);
  offsetX = (canvas.width - img.naturalWidth * scale) / 2;
  offsetY = (canvas.height - img.naturalHeight * scale) / 2;
  render();
}

function colorFor(i, alpha=1) {
  const hue = (i * 137.508) % 360;
  return `hsla(${hue}, 90%, 60%, ${alpha})`;
}

function drawGeometry(inst, i, isDraft=false) {
  const pts = inst.points || [];
  if (!pts.length) return;
  const kind = inst.kind || inst.mode || 'polyline';
  const th = Number(inst.thickness || 4);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let k=1; k<pts.length; k++) ctx.lineTo(pts[k][0], pts[k][1]);
  const c = isDraft ? 'rgba(255,220,0,0.95)' : colorFor(i, 0.95);
  if (kind === 'polygon' && pts.length >= 3) {
    ctx.closePath();
    ctx.fillStyle = isDraft ? 'rgba(255,220,0,0.20)' : colorFor(i, 0.22);
    ctx.fill();
    ctx.strokeStyle = c;
    ctx.lineWidth = Math.max(1/scale, 1.5/scale);
    ctx.stroke();
  } else {
    ctx.strokeStyle = c;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = th;
    ctx.stroke();
  }

  const r = 3.5 / scale;
  ctx.fillStyle = c;
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, Math.PI*2);
    ctx.fill();
  }

  if (!isDraft) {
    ctx.font = `${Math.max(10/scale, 12/scale)}px sans-serif`;
    ctx.fillStyle = c;
    ctx.fillText(String(i), pts[0][0] + 5/scale, pts[0][1] - 5/scale);
  }
}

function render() {
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!imageLoaded || !tile) return;
  ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
  ctx.drawImage(img, 0, 0);
  (tile.instances || []).forEach((inst, k) => drawGeometry(inst, k+1, false));
  drawGeometry(tile.draft || {points:[]}, (tile.instances || []).length+1, true);
}

function canvasToImage(ev) {
  const rect = canvas.getBoundingClientRect();
  const cx = ev.clientX - rect.left;
  const cy = ev.clientY - rect.top;
  let x = (cx - offsetX) / scale;
  let y = (cy - offsetY) / scale;
  x = clamp(x, 0, img.naturalWidth - 1);
  y = clamp(y, 0, img.naturalHeight - 1);
  return [Math.round(x), Math.round(y)];
}

async function api(url, opts={}) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (_) {}
  if (!res.ok) {
    const msg = data && data.error ? data.error : `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data;
}

function payload() {
  return {
    instances: tile.instances || [],
    draft: tile.draft || {kind:'polyline', thickness:4, points:[]}
  };
}

async function saveCurrent(showMessage=false) {
  if (!tile || busy) return;
  const data = await api(`/api/tile/${index}`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload())
  });
  tile.completed = data.completed;
  completedCount = data.completed_count;
  updateUI();
  if (showMessage) flash(`Saved. Completed tiles exported: ${completedCount}/${total}`);
}

function flash(msg) {
  const s = document.getElementById('status');
  s.textContent = msg;
  setTimeout(() => { if (s.textContent === msg) s.textContent = ''; }, 2200);
}

async function setIndexServer(i) {
  await api('/api/current', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({index:i})});
}

async function loadTile(i, saveFirst=true) {
  if (busy) return;
  busy = true;
  try {
    if (tile && saveFirst) await saveCurrent(false);
    index = clamp(i, 0, total-1);
    await setIndexServer(index);
    const data = await api(`/api/tile/${index}`);
    tile = data.tile;
    completedCount = data.completed_count;
    document.getElementById('mode').value = tile.draft.kind || 'polyline';
    document.getElementById('thickness').value = tile.draft.thickness || 4;
    imageLoaded = false;
    img.onload = () => { imageLoaded = true; fitImage(); updateUI(); };
    img.src = `/api/image/${index}?v=${Date.now()}`;
    updateUI();
  } catch (e) {
    alert(e.message);
  } finally {
    busy = false;
  }
}

function updateUI() {
  if (!tile) return;
  document.getElementById('fileLabel').textContent = `${index+1}/${total} — ${tile.filename}`;
  document.getElementById('tileState').innerHTML = tile.completed
    ? '<span class="done">COMPLETE</span>'
    : '<span class="open">UNFINISHED — excluded from COCO</span>';
  document.getElementById('instanceCount').textContent = `instances: ${(tile.instances || []).length}`;
  document.getElementById('progress').textContent = `completed: ${completedCount}/${total}`;
  document.getElementById('prevBtn').disabled = index <= 0;
  document.getElementById('nextBtn').disabled = index >= total-1;
  document.getElementById('reopenBtn').disabled = !tile.completed;
}

async function mutate(fn) {
  if (!tile) return;
  fn();
  render();
  try { await saveCurrent(false); } catch(e) { alert(e.message); }
}

async function commitInstance() {
  const d = tile.draft;
  const need = d.kind === 'polygon' ? 3 : 2;
  if (!d.points || d.points.length < need) {
    flash(`${d.kind} needs at least ${need} points`);
    return;
  }
  await mutate(() => {
    tile.instances.push({kind:d.kind, thickness:Number(d.thickness), points:d.points.map(p => [p[0],p[1]])});
    d.points = [];
  });
}

async function completeAndNext() {
  if (tile.draft.points && tile.draft.points.length) {
    alert('There is an uncommitted draft. Commit it or clear it before marking the tile complete.');
    return;
  }
  try {
    const data = await api(`/api/complete/${index}`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload())
    });
    tile.completed = true;
    completedCount = data.completed_count;
    updateUI();
    flash('Tile completed and exported to COCO');
    if (data.next_index !== null && data.next_index !== undefined && data.next_index !== index) {
      await loadTile(data.next_index, false);
    }
  } catch(e) { alert(e.message); }
}

async function reopenTile() {
  try {
    const data = await api(`/api/reopen/${index}`, {method:'POST'});
    tile.completed = false;
    completedCount = data.completed_count;
    updateUI();
  } catch(e) { alert(e.message); }
}

canvas.addEventListener('contextmenu', e => { e.preventDefault(); if (tile) mutate(() => { tile.draft.points.pop(); }); });
canvas.addEventListener('mousedown', e => {
  if (e.button === 1 || spaceDown) {
    panning = true;
    panLast = [e.clientX, e.clientY];
    e.preventDefault();
    return;
  }
  if (e.button === 0 && tile && imageLoaded) {
    const p = canvasToImage(e);
    mutate(() => { tile.draft.points.push(p); });
  }
});
window.addEventListener('mousemove', e => {
  if (!panning || !panLast) return;
  offsetX += e.clientX - panLast[0];
  offsetY += e.clientY - panLast[1];
  panLast = [e.clientX, e.clientY];
  render();
});
window.addEventListener('mouseup', () => { panning=false; panLast=null; });
canvas.addEventListener('wheel', e => {
  if (!imageLoaded) return;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const ix = (mx - offsetX) / scale;
  const iy = (my - offsetY) / scale;
  const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
  const newScale = clamp(scale * factor, 0.05, 20);
  offsetX = mx - ix * newScale;
  offsetY = my - iy * newScale;
  scale = newScale;
  render();
}, {passive:false});

document.getElementById('mode').addEventListener('change', e => mutate(() => {
  tile.draft.kind = e.target.value;
  tile.draft.points = [];
}));
document.getElementById('thickness').addEventListener('change', e => mutate(() => {
  tile.draft.thickness = clamp(Number(e.target.value)||1, 1, 100);
  e.target.value = tile.draft.thickness;
}));
document.getElementById('prevBtn').onclick = () => loadTile(index-1);
document.getElementById('nextBtn').onclick = () => loadTile(index+1);
document.getElementById('fitBtn').onclick = fitImage;
document.getElementById('undoBtn').onclick = () => mutate(() => tile.draft.points.pop());
document.getElementById('clearBtn').onclick = () => mutate(() => { tile.draft.points=[]; });
document.getElementById('commitBtn').onclick = commitInstance;
document.getElementById('deleteBtn').onclick = () => mutate(() => tile.instances.pop());
document.getElementById('saveBtn').onclick = () => saveCurrent(true);
document.getElementById('completeBtn').onclick = completeAndNext;
document.getElementById('reopenBtn').onclick = reopenTile;

window.addEventListener('keydown', async e => {
  if (['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.code === 'Space') { spaceDown=true; e.preventDefault(); return; }
  if (e.key === 'Enter') { e.preventDefault(); await commitInstance(); }
  else if (e.key === 'u' || e.key === 'U' || e.key === 'Backspace') { e.preventDefault(); await mutate(() => tile.draft.points.pop()); }
  else if (e.key === 'c' || e.key === 'C') { await mutate(() => { tile.draft.points=[]; }); }
  else if (e.key === 'd' || e.key === 'D') { await mutate(() => tile.instances.pop()); }
  else if (e.key === 'm' || e.key === 'M') {
    await mutate(() => {
      tile.draft.kind = tile.draft.kind === 'polyline' ? 'polygon' : 'polyline';
      tile.draft.points = [];
      document.getElementById('mode').value = tile.draft.kind;
    });
  }
  else if (e.key === '[') {
    await mutate(() => { tile.draft.thickness = Math.max(1, Number(tile.draft.thickness)-1); document.getElementById('thickness').value=tile.draft.thickness; });
  }
  else if (e.key === ']') {
    await mutate(() => { tile.draft.thickness = Math.min(100, Number(tile.draft.thickness)+1); document.getElementById('thickness').value=tile.draft.thickness; });
  }
  else if (e.key === 'n' || e.key === 'N' || e.key === 'ArrowRight') { if (index < total-1) await loadTile(index+1); }
  else if (e.key === 'p' || e.key === 'P' || e.key === 'ArrowLeft') { if (index > 0) await loadTile(index-1); }
  else if (e.key === 'f' || e.key === 'F') fitImage();
  else if (e.key === 's' || e.key === 'S') { e.preventDefault(); await saveCurrent(true); }
});
window.addEventListener('keyup', e => { if (e.code === 'Space') spaceDown=false; });
window.addEventListener('beforeunload', () => {
  if (!tile) return;
  navigator.sendBeacon(`/api/beacon/${index}`, new Blob([JSON.stringify(payload())], {type:'application/json'}));
});

(async function boot() {
  try {
    const b = await api('/api/bootstrap');
    total = b.total;
    completedCount = b.completed_count;
    await loadTile(b.resume_index, false);
  } catch(e) {
    document.body.innerHTML = `<pre style="padding:20px;color:#f88">${e.message}</pre>`;
  }
})();
</script>
</body>
</html>
"""


def atomic_json_write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
    tmp.replace(path)


def mask_to_coco(mask: np.ndarray) -> tuple[list[list[float]], list[float], float] | None:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for c in contours:
        pts = c.reshape(-1, 2)
        if len(pts) < 3:
            continue
        flat = pts.astype(float).reshape(-1).tolist()
        if len(flat) >= 6:
            polygons.append(flat)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or not polygons:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
    area = float(np.count_nonzero(mask))
    return polygons, bbox, area


class AnnotationStore:
    def __init__(self, dataset: Path, split: str, default_thickness: int):
        self.split_dir = dataset / split
        if not self.split_dir.is_dir():
            raise SystemExit(f"Split directory not found: {self.split_dir}")

        self.files = sorted(p for p in self.split_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not self.files:
            raise SystemExit(f"No images found in {self.split_dir}")

        self.default_thickness = max(1, int(default_thickness))
        self.state_path = self.split_dir / "_annotation_working.json"
        self.coco_path = self.split_dir / "_annotations.coco.json"
        self.lock = threading.RLock()
        self.dim_cache: dict[str, tuple[int, int]] = {}
        self.state = self._load_state()
        self._ensure_tiles()
        self.save_state()
        self.export_coco()

    def _new_tile(self) -> dict[str, Any]:
        return {
            "instances": [],
            "draft": {"kind": "polyline", "thickness": self.default_thickness, "points": []},
        }

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            state.setdefault("version", 2)
            state.setdefault("last_index", 0)
            state.setdefault("completed", [])
            state.setdefault("tiles", {})
            return state
        return {"version": 2, "last_index": 0, "completed": [], "tiles": {}}

    def _ensure_tiles(self) -> None:
        names = {p.name for p in self.files}
        for p in self.files:
            if p.name not in self.state["tiles"]:
                self.state["tiles"][p.name] = self._new_tile()
            tile = self.state["tiles"][p.name]
            tile.setdefault("instances", [])
            tile.setdefault("draft", {})
            tile["draft"].setdefault("kind", "polyline")
            tile["draft"].setdefault("thickness", self.default_thickness)
            tile["draft"].setdefault("points", [])
        self.state["completed"] = [n for n in self.state.get("completed", []) if n in names]
        self.state["last_index"] = int(max(0, min(int(self.state.get("last_index", 0)), len(self.files)-1)))

    def dimensions(self, filename: str) -> tuple[int, int]:
        if filename not in self.dim_cache:
            with Image.open(self.split_dir / filename) as im:
                self.dim_cache[filename] = im.size
        return self.dim_cache[filename]

    def save_state(self) -> None:
        with self.lock:
            atomic_json_write(self.state_path, self.state)

    def sanitize_payload(self, index: int, payload: dict[str, Any]) -> dict[str, Any]:
        filename = self.files[index].name
        w, h = self.dimensions(filename)

        def clean_points(points: Any) -> list[list[int]]:
            out: list[list[int]] = []
            if not isinstance(points, list):
                return out
            for p in points:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    continue
                try:
                    x = int(round(float(p[0])))
                    y = int(round(float(p[1])))
                except (TypeError, ValueError):
                    continue
                out.append([max(0, min(w-1, x)), max(0, min(h-1, y))])
            return out

        instances: list[dict[str, Any]] = []
        for inst in payload.get("instances", []) if isinstance(payload, dict) else []:
            if not isinstance(inst, dict):
                continue
            kind = inst.get("kind", "polyline")
            kind = "polygon" if kind == "polygon" else "polyline"
            thickness = max(1, min(100, int(inst.get("thickness", self.default_thickness))))
            points = clean_points(inst.get("points", []))
            instances.append({"kind": kind, "thickness": thickness, "points": points})

        draft_in = payload.get("draft", {}) if isinstance(payload, dict) else {}
        kind = "polygon" if draft_in.get("kind") == "polygon" else "polyline"
        thickness = max(1, min(100, int(draft_in.get("thickness", self.default_thickness))))
        draft = {"kind": kind, "thickness": thickness, "points": clean_points(draft_in.get("points", []))}
        return {"instances": instances, "draft": draft}

    def save_tile(self, index: int, payload: dict[str, Any], uncomplete_if_changed: bool = True) -> bool:
        with self.lock:
            filename = self.files[index].name
            new_tile = self.sanitize_payload(index, payload)
            old_tile = self.state["tiles"].get(filename, self._new_tile())
            changed = old_tile != new_tile
            self.state["tiles"][filename] = new_tile
            self.state["last_index"] = index
            if changed and uncomplete_if_changed and filename in self.state["completed"]:
                self.state["completed"].remove(filename)
            self.save_state()
            if changed:
                self.export_coco()
            return filename in self.state["completed"]

    def instance_to_coco(self, filename: str, inst: dict[str, Any]):
        w, h = self.dimensions(filename)
        mask = np.zeros((h, w), np.uint8)
        pts = np.asarray(inst.get("points", []), dtype=np.int32)
        kind = inst.get("kind", "polyline")
        thickness = max(1, int(inst.get("thickness", self.default_thickness)))

        if kind == "polygon":
            if len(pts) < 3:
                return None
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
        else:
            if len(pts) < 2:
                return None
            cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 1, thickness=thickness, lineType=cv2.LINE_8)
            r = max(1, thickness // 2)
            cv2.circle(mask, tuple(pts[0]), r, 1, -1)
            cv2.circle(mask, tuple(pts[-1]), r, 1, -1)

        return mask_to_coco(mask)

    def validate_tile(self, index: int) -> tuple[bool, str]:
        filename = self.files[index].name
        tile = self.state["tiles"][filename]
        if tile.get("draft", {}).get("points"):
            return False, "Uncommitted draft points remain. Commit the instance or clear the draft first."
        for i, inst in enumerate(tile.get("instances", []), start=1):
            need = 3 if inst.get("kind") == "polygon" else 2
            if len(inst.get("points", [])) < need:
                return False, f"Instance {i} has too few points."
            if self.instance_to_coco(filename, inst) is None:
                return False, f"Instance {i} could not be converted to a valid COCO polygon."
        return True, ""

    def mark_complete(self, index: int, payload: dict[str, Any]) -> None:
        with self.lock:
            self.save_tile(index, payload, uncomplete_if_changed=False)
            ok, msg = self.validate_tile(index)
            if not ok:
                raise ValueError(msg)
            filename = self.files[index].name
            if filename not in self.state["completed"]:
                self.state["completed"].append(filename)
            self.state["last_index"] = index
            self.save_state()
            self.export_coco()

    def reopen(self, index: int) -> None:
        with self.lock:
            filename = self.files[index].name
            if filename in self.state["completed"]:
                self.state["completed"].remove(filename)
            self.state["last_index"] = index
            self.save_state()
            self.export_coco()

    def completed_count(self) -> int:
        return len(set(self.state.get("completed", [])))

    def resume_index(self) -> int:
        with self.lock:
            n = len(self.files)
            last = max(0, min(int(self.state.get("last_index", 0)), n-1))
            completed = set(self.state.get("completed", []))
            if self.files[last].name not in completed:
                return last
            for off in range(1, n+1):
                j = (last + off) % n
                if self.files[j].name not in completed:
                    return j
            return last

    def next_unfinished(self, index: int) -> int | None:
        completed = set(self.state.get("completed", []))
        n = len(self.files)
        for off in range(1, n+1):
            j = (index + off) % n
            if self.files[j].name not in completed:
                return j
        return None

    def tile_response(self, index: int) -> dict[str, Any]:
        filename = self.files[index].name
        tile = self.state["tiles"][filename]
        return {
            "filename": filename,
            "instances": tile.get("instances", []),
            "draft": tile.get("draft", self._new_tile()["draft"]),
            "completed": filename in set(self.state.get("completed", [])),
        }

    def export_coco(self) -> None:
        with self.lock:
            completed = set(self.state.get("completed", []))
            coco = {
                "info": {"description": "Fiber instance segmentation - completed tiles only", "version": "1.0"},
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "fiber", "supercategory": "fiber"}],
            }
            ann_id = 1
            image_id = 1
            for p in self.files:
                if p.name not in completed:
                    continue
                w, h = self.dimensions(p.name)
                coco["images"].append({"id": image_id, "file_name": p.name, "width": w, "height": h})
                tile = self.state["tiles"][p.name]
                for inst in tile.get("instances", []):
                    result = self.instance_to_coco(p.name, inst)
                    if result is None:
                        continue
                    segmentation, bbox, area = result
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": segmentation,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    })
                    ann_id += 1
                image_id += 1
            atomic_json_write(self.coco_path, coco)


def create_app(store: AnnotationStore) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index_page() -> str:
        return render_template_string(HTML)

    @app.get("/api/bootstrap")
    def bootstrap():
        return jsonify({
            "total": len(store.files),
            "completed_count": store.completed_count(),
            "resume_index": store.resume_index(),
            "state_file": str(store.state_path),
            "coco_file": str(store.coco_path),
        })

    @app.get("/api/tile/<int:index>")
    def get_tile(index: int):
        if index < 0 or index >= len(store.files):
            return jsonify({"error": "index out of range"}), 404
        return jsonify({"tile": store.tile_response(index), "completed_count": store.completed_count()})

    @app.post("/api/tile/<int:index>")
    def save_tile(index: int):
        if index < 0 or index >= len(store.files):
            return jsonify({"error": "index out of range"}), 404
        payload = request.get_json(silent=True) or {}
        completed = store.save_tile(index, payload, uncomplete_if_changed=True)
        return jsonify({"ok": True, "completed": completed, "completed_count": store.completed_count()})

    @app.post("/api/beacon/<int:index>")
    def beacon_save(index: int):
        if 0 <= index < len(store.files):
            payload = request.get_json(silent=True) or {}
            store.save_tile(index, payload, uncomplete_if_changed=True)
        return Response(status=204)

    @app.post("/api/current")
    def current():
        payload = request.get_json(silent=True) or {}
        try:
            idx = int(payload.get("index", 0))
        except (TypeError, ValueError):
            idx = 0
        idx = max(0, min(idx, len(store.files)-1))
        store.state["last_index"] = idx
        store.save_state()
        return jsonify({"ok": True})

    @app.post("/api/complete/<int:index>")
    def complete(index: int):
        if index < 0 or index >= len(store.files):
            return jsonify({"error": "index out of range"}), 404
        payload = request.get_json(silent=True) or {}
        try:
            store.mark_complete(index, payload)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        nxt = store.next_unfinished(index)
        return jsonify({"ok": True, "completed_count": store.completed_count(), "next_index": nxt})

    @app.post("/api/reopen/<int:index>")
    def reopen(index: int):
        if index < 0 or index >= len(store.files):
            return jsonify({"error": "index out of range"}), 404
        store.reopen(index)
        return jsonify({"ok": True, "completed_count": store.completed_count()})

    @app.get("/api/image/<int:index>")
    def image(index: int):
        if index < 0 or index >= len(store.files):
            return jsonify({"error": "index out of range"}), 404
        return send_file(store.files[index])

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path, help="Dataset root made by the tiling script")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="train")
    ap.add_argument("--thickness", type=int, default=4, help="Default polyline mask width in image pixels")
    ap.add_argument("--host", default="127.0.0.1", help="Flask bind host; use 0.0.0.0 for remote/LAN access")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = ap.parse_args()

    store = AnnotationStore(args.dataset, args.split, args.thickness)
    app = create_app(store)

    shown_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    url = f"http://{shown_host}:{args.port}"
    print("\nFiber instance annotator")
    print(f"  Open:          {url}")
    print(f"  Split:         {store.split_dir}")
    print(f"  Resume state:  {store.state_path}")
    print(f"  Training COCO: {store.coco_path}")
    print(f"  Completed:     {store.completed_count()}/{len(store.files)}")
    print("\nOnly tiles explicitly marked COMPLETE are written to the training COCO file.\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
