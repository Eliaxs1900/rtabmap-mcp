# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.10,<2", "numpy>=1.26", "pillow>=10"]
# ///
"""MCP server: read and preprocess RTAB-Map databases (.db) such as the ones
exported by the RTAB-Map iOS app (iPhone/iPad LiDAR scans).

Pipeline the tools expose:
  1. rtabmap_info            -> what is inside the .db (frames, sensor, intrinsics)
  2. rtabmap_get_frame       -> save one RGB image + depth preview to look at
  3. rtabmap_build_cloud     -> depth maps + poses -> world point cloud, voxel
                                 downsampled, optionally levelled (floor at z=0)
                                 and yaw-aligned to the walls; saved as .npz + .ply
  4. rtabmap_plan_image      -> top-down PNG of the cloud, coloured by height band
  5. rtabmap_height_grid     -> table of top-surface heights (for measuring furniture)
  6. rtabmap_axis_histogram  -> point histogram along one axis inside a box
                                 (finds faces / walls / edges)
  7. rtabmap_transform_cloud -> crop / rotate / shift a processed cloud and export
                                 it (e.g. into a Blender scene's coordinates)

Outputs: every tool that writes files takes an optional path. By default they go to a
private temporary folder ($TMPDIR/rtabmap_mcp/<scan>/, per user). Scans, photos and
outputs are private: nothing is uploaded anywhere. Results are plain text (compact JSON
or a table) so they read well in a chat.

Format notes (learned the hard way):
  * Depth frames are PNG. A float32 depth image (metres) is stored by OpenCV as
    an 8-bit 4-channel PNG, i.e. the float bytes in BGRA order: swap R and B
    before reinterpreting as float32. 16-bit PNG depth is millimetres.
  * Calibration blob (mono camera, RTAB-Map >= 0.18): 4 int32 header
    (major, minor, patch, type) + width, height + sizes of K, D, R, P,
    local_transform, then K/D/R/P as float64 and local_transform as float32.
    Depth is lower resolution than RGB: scale K by depth_w / rgb_w.
  * Node.pose is a row-major 3x4 float32 (world <- base); the camera's
    local_transform maps optical frame -> base. Node.pose is raw odometry.
  * Admin.opt_ids / opt_poses hold the poses of the optimized graph (loop
    closures applied), saved when the map was closed. Both use RTAB-Map's
    compressData2(): a zlib stream followed by int32 rows, cols, OpenCV type.
    Nodes with weight -9 were dropped from the graph (e.g. the camera did not
    move) and are not in opt_ids. In the two test scans the optimized poses
    move frames 1-3.5 cm and 0.1-0.3 degrees away from odometry.
  * Data.depth_confidence: compressData2 uint8 at depth resolution, ARKit
    confidence stored as 0 / 50 / 100 (low / medium / high).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import struct
import tempfile
import zlib
from contextlib import closing
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

INSTRUCTIONS = """\
Reads RTAB-Map .db scans (iPhone/iPad LiDAR, RTAB-Map iOS app) and turns them into measurements.
Workflow: rtabmap_info -> rtabmap_build_cloud (writes a .npz) -> rtabmap_plan_image to see the
layout -> rtabmap_axis_histogram (walls, faces, floor, ceiling) and rtabmap_height_grid (surface
heights) on that .npz. rtabmap_get_frame shows the photo of one frame; rtabmap_transform_cloud
crops and moves a piece into another scene's coordinates.
Units are metres. A levelled cloud has z up, the floor at z=0 and the main walls along x and y;
its x/y origin is the centre of the scanned floor, so compare coordinates only within one .npz
(distances and heights are comparable across clouds).
Scans, photos and every output are private: files are only written locally (by default to a
private temporary folder) and must never be uploaded or shared."""

mcp = FastMCP("rtabmap", instructions=INSTRUCTIONS)

PNG_SIG = b"\x89PNG\r\n\x1a\n"
JPEG_SIG = b"\xff\xd8\xff"


# ----------------------------------------------------------------- low level
def _db(path: str) -> sqlite3.Connection:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"No existe el archivo: {p}")
    # as_uri() escapes spaces, '#', '?' and '%'. A raw f"file:{p}" URI was cut at '#', lost
    # mode=ro and made SQLite create an empty database at the truncated path.
    uri, err = p.resolve().as_uri(), None
    for query in ("mode=ro", "mode=ro&immutable=1"):  # immutable: read-only folders, no locking
        try:
            con = sqlite3.connect(f"{uri}?{query}", uri=True)
        except sqlite3.Error as e:
            err = e
            continue
        try:
            tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            break
        except sqlite3.Error as e:
            con.close()
            err = e
    else:
        raise ValueError(f"No se puede abrir {p} como base de datos SQLite ({err}). Si está en una "
                         "carpeta protegida de macOS, cópialo antes a una carpeta normal.")
    if not {"Node", "Data"} <= tables:
        con.close()
        raise ValueError(f"{p.name} no es una base de datos RTAB-Map (faltan las tablas Node/Data).")
    return con


def _parse_calibration(blob: bytes) -> dict:
    if not blob or len(blob) < 44:
        raise ValueError("Fotograma sin calibración de cámara.")
    major, minor, patch, ctype = struct.unpack_from("<4i", blob, 0)
    if not (0 <= major < 100 and 0 <= minor < 1000 and 0 <= patch < 1000 and ctype in (0, 1)):
        raise ValueError("Calibración en el formato anterior a RTAB-Map 0.18 (lista de floats): no soportada.")
    if ctype != 0:
        raise ValueError(f"Cámara tipo {ctype} (estéreo) no soportada; sólo RGB-D mono.")
    w, h, kN, dN, rN, pN, ltN = struct.unpack_from("<7i", blob, 16)
    base = 44 + 8 * (kN + dN + rN + pN)
    if min(w, h, kN, dN, rN, pN, ltN) < 0 or base > len(blob):
        raise ValueError("Calibración incompleta o dañada.")
    if len(blob) == base:
        ltN = 0  # null local transform: declared but not stored
    elif len(blob) > base + 4 * ltN:
        raise ValueError("Calibración con varias cámaras: sólo se soporta una.")
    elif len(blob) < base + 4 * ltN:
        raise ValueError("Calibración incompleta o dañada.")
    if kN != 9 or w == 0 or h == 0:
        raise ValueError("La calibración no tiene matriz K o tamaño de imagen.")
    off = 44
    def d(n):
        nonlocal off
        v = np.frombuffer(blob, "<f8", n, off); off += 8 * n; return v
    K, D, R, P = d(kN), d(dN), d(rN), d(pN)
    LT = np.frombuffer(blob, "<f4", ltN, off)
    K = K.reshape(3, 3)
    LT = LT.reshape(3, 4).astype(np.float64) if ltN == 12 else np.hstack([np.eye(3), np.zeros((3, 1))])
    return {"version": f"{major}.{minor}.{patch}", "width": w, "height": h, "K": K, "local_transform": LT}


def _decode_depth(blob: bytes) -> np.ndarray:
    """Return depth in metres (float32, 0 = invalid)."""
    if not blob:
        raise ValueError("Fotograma sin profundidad.")
    if blob[:8] == PNG_SIG:
        im = Image.open(io.BytesIO(blob))
        if im.mode == "RGBA":  # float32 packed by OpenCV as BGRA
            a = np.asarray(im)[:, :, [2, 1, 0, 3]].copy()
            return a.view("<f4")[:, :, 0]
        a = np.asarray(im)
        if a.dtype in (np.uint16, np.int32) or im.mode.startswith("I"):
            return a.astype(np.float32) / 1000.0
        raise ValueError(f"PNG de profundidad con modo {im.mode} no soportado.")
    if blob[:3] == b"RVL":
        raise ValueError("Profundidad comprimida en RVL: no soportada todavía.")
    try:  # very old DBs: zlib raw
        raw = zlib.decompress(blob)
    except zlib.error:
        raise ValueError("Formato de profundidad desconocido.") from None
    raise ValueError(f"Profundidad zlib cruda ({len(raw)} bytes) no soportada.")


def _pose(blob: bytes | None) -> np.ndarray | None:
    """Node.pose as a 3x4 float64 matrix; None when missing or null (all zeros)."""
    if not blob or len(blob) < 48:
        return None
    T = np.frombuffer(blob, "<f4", 12).reshape(3, 4).astype(np.float64)
    return T if T.any() else None


def _unpack_mat(blob: bytes, dtype) -> tuple[np.ndarray, int, int]:
    """RTAB-Map compressData2(): zlib stream followed by int32 rows, cols, OpenCV type."""
    if not blob or len(blob) < 13:
        raise ValueError("bloque comprimido vacío")
    rows, cols, _ = struct.unpack_from("<3i", blob, len(blob) - 12)
    return np.frombuffer(zlib.decompress(blob[:-12]), dtype), rows, cols


def _optimized_poses(con: sqlite3.Connection) -> dict[int, np.ndarray] | None:
    """Admin.opt_ids / opt_poses: the poses of RTAB-Map's optimized graph (loop closures
    applied) saved when the map was closed. None if the database has none."""
    try:
        row = con.execute("select opt_ids, opt_poses from Admin").fetchone()
        ids, P = _unpack_mat(row[0], "<i4")[0], _unpack_mat(row[1], "<f4")[0]
    except (sqlite3.Error, TypeError, ValueError, zlib.error, struct.error):
        return None
    if not len(ids) or len(P) != 12 * len(ids):
        return None
    P = P.reshape(-1, 3, 4).astype(np.float64)
    return {int(i): P[k] for k, i in enumerate(ids)}


CONFIDENCE = {"low": 0, "medium": 1, "high": 2}


def _confidence(blob: bytes | None) -> np.ndarray | None:
    """Data.depth_confidence as ARKit levels 0 low / 1 medium / 2 high; None if missing."""
    try:
        a, rows, cols = _unpack_mat(blob, np.uint8)
    except (TypeError, ValueError, zlib.error, struct.error):
        return None
    if rows * cols != a.size:
        return None
    a = a.reshape(rows, cols)
    return a // 50 if a.max() > 2 else a  # the iOS app stores 0 / 50 / 100


def _ranges(ids) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1..3, 7, 9..10'."""
    ids, out, i = sorted(ids), [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1: j += 1
        out.append(f"{ids[i]}..{ids[j]}" if j > i else str(ids[i])); i = j + 1
    return ", ".join(out)


def _frame_points(depth_blob, calib_blob, T, dmin, dmax, step):
    D = _decode_depth(depth_blob)
    cal = _parse_calibration(calib_blob)
    h, w = D.shape
    sx, sy = w / cal["width"], h / cal["height"]
    fx, fy, cx, cy = cal["K"][0, 0] * sx, cal["K"][1, 1] * sy, cal["K"][0, 2] * sx, cal["K"][1, 2] * sy
    v, u = np.mgrid[0:h:step, 0:w:step]
    Z = D[0:h:step, 0:w:step]
    m = np.isfinite(Z) & (Z > dmin) & (Z < dmax)
    z = Z[m].astype(np.float64)
    P = np.stack([(u[m] - cx) * z / fx, (v[m] - cy) * z / fy, z], 1)
    LT = cal["local_transform"]; P = P @ LT[:, :3].T + LT[:, 3]
    P = P @ T[:, :3].T + T[:, 3]
    return P, (u[m], v[m], (h, w))


def _voxel(P, size, *extra):
    """Keep the first point (in input order) of each voxel; output sorted by voxel (x, y, z).
    The voxel indices are packed into one int64 with the same lexicographic order, which gives
    exactly the result of np.unique(k, axis=0) about 5x faster and with less memory."""
    k = np.floor(P / size).astype(np.int64)
    if not len(k):
        return (P,) + extra
    k -= k.min(0)
    n = k.max(0) + 1
    if float(n[0]) * float(n[1]) * float(n[2]) < 2.0 ** 62:
        _, idx = np.unique((k[:, 0] * n[1] + k[:, 1]) * n[2] + k[:, 2], return_index=True)
    else:  # absurd extent / voxel ratio: fall back to the row-wise unique
        _, idx = np.unique(k, axis=0, return_index=True)
    return (P[idx],) + tuple(e[idx] for e in extra)


def _level_and_align(P: np.ndarray, align_walls: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Floor -> z=0 (fit on the lowest dense height band) and optional yaw so walls are axis aligned."""
    z = P[:, 2]
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.02))
    thr = 0.25 * h.max()
    i = int(np.argmax(h >= thr))  # lowest strong peak = floor
    i = i + int(np.argmax(h[i:i + 5]))
    fz = e[i] + 0.01
    fl = P[np.abs(z - fz) < 0.04]
    if len(fl) < 3:
        raise ValueError("No se encuentra el suelo en la nube; prueba con level_floor=false.")
    c = fl.mean(0)
    _, _, vt = np.linalg.svd(fl - c, full_matrices=False)
    n = vt[2] * np.sign(vt[2][2])
    x = np.cross([0, 1, 0], n); x /= np.linalg.norm(x); y = np.cross(n, x)
    R = np.stack([x, y, n])
    W = (P - c) @ R.T
    info = {"floor_normal": n.round(4).tolist(), "floor_height_raw": round(float(c[2]), 3)}
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ c
    if align_walls:
        top = W[:, 2].max()
        H = W[(W[:, 2] > 0.6 * top) & (W[:, 2] < 0.9 * top)][:, :2]
        if len(H) > 1000:
            best, ang = -1, 0.0
            for a in np.radians(np.arange(0, 90, 0.25)):
                ca, sa = np.cos(a), np.sin(a)
                s_ = (np.histogram(H @ [ca, sa], bins=400)[0].astype(np.float64) ** 2).sum() + \
                     (np.histogram(H @ [-sa, ca], bins=400)[0].astype(np.float64) ** 2).sum()
                if s_ > best: best, ang = s_, a
            ca, sa = np.cos(ang), np.sin(ang)
            Rz = np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]])
            W = W @ Rz.T
            Mz = np.eye(4); Mz[:3, :3] = Rz; M = Mz @ M
            info["yaw_deg"] = round(float(np.degrees(ang)), 2)
        else:
            info["yaw_deg"] = None  # too few wall points to align: left as scanned
    return W, M, info


def _write_ply(path: Path, P: np.ndarray, C: np.ndarray | None):
    n = len(P)
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
           "property float x", "property float y", "property float z"]
    if C is not None:
        hdr += ["property uchar red", "property uchar green", "property uchar blue"]
    hdr.append("end_header")
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if C is not None else [])
    a = np.empty(n, dtype=dt)
    a["x"], a["y"], a["z"] = P[:, 0], P[:, 1], P[:, 2]
    if C is not None:
        a["r"], a["g"], a["b"] = C[:, 0], C[:, 1], C[:, 2]
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode()); f.write(a.tobytes())


def _out_base(path: str, *exts: str) -> Path:
    """Absolute output path; strips a known extension but keeps dots inside the name."""
    p = Path(path).expanduser()
    if p.suffix.lower() in exts:
        p = p.with_suffix("")
    return p.resolve()


def _load_cloud(path: str):
    p = Path(path).expanduser()
    if p.suffix.lower() != ".npz":  # accept the .ply or the base name rtabmap_build_cloud was given
        p = next((q for q in (Path(f"{p}.npz"), p.with_suffix(".npz")) if q.is_file()), p)
        if p.suffix.lower() != ".npz":
            raise ValueError(f"Pasa el .npz que genera rtabmap_build_cloud (recibido: {path}).")
    if not p.is_file():
        raise FileNotFoundError(f"No existe la nube: {p}")
    with np.load(p, allow_pickle=False) as z:
        if "points" not in z.files:
            raise ValueError(f"{p.name} no es una nube de rtabmap_build_cloud (no tiene 'points').")
        meta = {k: z[k] for k in z.files if k not in ("points", "colors")}
        return p, z["points"].astype(np.float64), (z["colors"] if "colors" in z.files else None), meta


def _summ(P):
    return {"points": int(len(P)),
            "min": P.min(0).round(3).tolist() if len(P) else None,
            "max": P.max(0).round(3).tolist() if len(P) else None}


def _vec(v, name: str, n: int = 3):
    if v is None:
        return None
    if len(v) != n:
        raise ValueError(f"{name} necesita {n} números (recibido {v}).")
    return np.asarray(v, float)


def _range(v, name: str):
    if v is None:
        return None
    if len(v) != 2 or not v[0] < v[1]:
        raise ValueError(f"{name} tiene que ser [mínimo, máximo] con mínimo < máximo (recibido {v}).")
    return float(v[0]), float(v[1])


def _nz(v) -> float:
    """Rounded to 2 decimals without printing '-0.00'."""
    return round(float(v), 2) + 0.0


OUT_ROOT = Path(tempfile.gettempdir()).resolve() / "rtabmap_mcp"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "scan"


def _default_dir(db_path: str) -> Path:
    """Private temporary folder for one scan's outputs ($TMPDIR is per user on macOS)."""
    return OUT_ROOT / _safe(Path(db_path).expanduser().stem)


def _short_hash(**params) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _out(d: dict) -> str:
    """JSON with one key per line and each value on one line: short and easy to read."""
    return "{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False, separators=(', ', ': '))}"
                               for k, v in d.items()) + "\n}"


# --------------------------------------------------------------------- tools
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITES_FILES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
DbPath = Annotated[str, Field(description="Path to the RTAB-Map .db file (iPhone/iPad LiDAR scan); ~ is expanded.")]
CloudPath = Annotated[str, Field(description="Cloud .npz written by rtabmap_build_cloud or rtabmap_transform_cloud "
                                             "(its .ply path or the name without extension also work).")]


@mcp.tool(title="RTAB-Map scan: summary", annotations=READ_ONLY, structured_output=False)
def rtabmap_info(db_path: DbPath) -> str:
    """What is inside a scan: RTAB-Map version, frames and their ids, camera and depth
    resolution, intrinsics, depth range of the first frame, how many frames are in RTAB-Map's
    optimized graph and how many loop closures it found, the LiDAR confidence mix, and the
    camera path (length and extent in RTAB-Map's world frame: x forward, y left, z up).
    Run it first: it tells which frame ids exist and whether the scan is usable."""
    p = Path(db_path).expanduser()
    with closing(_db(db_path)) as con:
        out = {"file": str(p.absolute()), "size_mb": round(os.path.getsize(p) / 1e6, 1)}
        try: out["rtabmap_version"] = con.execute("select version from Admin").fetchone()[0]
        except Exception: pass
        out["nodes"] = con.execute("select count(*) from Node").fetchone()[0]
        out["frames_with_depth"] = con.execute("select count(*) from Data where depth is not null").fetchone()[0]
        row = con.execute("select id, image, depth, calibration from Data where depth is not null order by id limit 1").fetchone()
        if row:
            cal = _parse_calibration(row[3]); D = _decode_depth(row[2])
            K = cal["K"]
            out["rgb_size"] = [cal["width"], cal["height"]]
            out["depth_size"] = [D.shape[1], D.shape[0]]
            out["intrinsics_rgb"] = {"fx": round(K[0, 0], 2), "fy": round(K[1, 1], 2), "cx": round(K[0, 2], 2), "cy": round(K[1, 2], 2)}
            out["depth_encoding"] = "png-float32(BGRA)" if row[2][:8] == PNG_SIG and Image.open(io.BytesIO(row[2])).mode == "RGBA" else "png-16bit(mm)"
            v = D[np.isfinite(D) & (D > 0)]
            if len(v):
                out["sample_depth_m"] = {"min": round(float(v.min()), 3), "median": round(float(np.median(v)), 3), "max": round(float(v.max()), 3)}
            out["rgb_format"] = "jpeg" if row[1] and row[1][:2] == b"\xff\xd8" else ("png" if row[1] and row[1][:4] == b"\x89PNG" else None)
        T = [t[:, 3] for t in (_pose(r[0]) for r in con.execute("select pose from Node order by id")) if t is not None]
        if T:
            T = np.array(T)
            out["trajectory_m"] = round(float(np.linalg.norm(np.diff(T, axis=0), axis=1).sum()), 2)
            out["camera_extent"] = {"min": T.min(0).round(2).tolist(), "max": T.max(0).round(2).tolist()}
        ids = [r[0] for r in con.execute("select id from Data where depth is not null order by id")]
        out["frame_ids"] = _ranges(ids) if ids else None
        out["sessions"] = con.execute("select count(distinct map_id) from Node").fetchone()[0]
        opt = _optimized_poses(con)
        out["optimized_poses"] = len(opt) if opt else None
        if opt:
            outside = sorted(set(ids) - set(opt))
            out["frames_outside_graph"] = _ranges(outside) if outside else None
        try:
            links = dict(con.execute("select type, count(*) from Link group by type").fetchall())
            out["loop_closures"] = int(sum(links.get(t, 0) for t in (1, 2, 3, 4)))
        except sqlite3.Error:
            pass
        if "depth_confidence" in {r[1] for r in con.execute("pragma table_info(Data)")}:
            lv = np.zeros(3, np.int64)
            for (b,) in con.execute("select depth_confidence from Data where depth is not null"):
                c = _confidence(b)
                if c is not None:
                    lv += np.bincount(c.ravel(), minlength=3)[:3]
            if lv.sum():
                out["depth_confidence_pct"] = {k: round(100 * float(n) / lv.sum(), 1) for k, n in zip(CONFIDENCE, lv)}
    return _out(out)


@mcp.tool(title="RTAB-Map scan: photo and depth of one frame", annotations=WRITES_FILES, structured_output=False)
def rtabmap_get_frame(
    db_path: DbPath,
    frame_id: Annotated[int, Field(description="Frame id; rtabmap_info lists them (frame_ids).")],
    out_dir: Annotated[str | None, Field(description="Folder for the two images. Default: a private temporary folder for this scan.")] = None,
) -> str:
    """Save what the camera saw in one frame: the colour photo exactly as stored in the scan
    (JPEG, not recompressed) and a greyscale depth preview (PNG at depth resolution, black = 0 m,
    white = 4 m or more). Returns both paths and the frame's depth range in m; open the images to
    look at them. File names start with the scan name, so frames of two scans never clash."""
    with closing(_db(db_path)) as con:
        row = con.execute("select image, depth from Data where id=?", (frame_id,)).fetchone()
        if not row:
            lo, hi = con.execute("select min(id), max(id) from Data").fetchone()
            raise ValueError(f"No hay fotograma {frame_id}; los ids van de {lo} a {hi}.")
    od = (Path(out_dir).expanduser() if out_dir else _default_dir(db_path) / "frames").resolve()
    od.mkdir(parents=True, exist_ok=True)
    name = f"{_safe(Path(db_path).expanduser().stem)}_frame_{frame_id}"
    res = {}
    if row[0]:
        if row[0][:3] == JPEG_SIG:  # the stored JPEG as is: no second lossy compression
            p = od / f"{name}_rgb.jpg"; p.write_bytes(row[0])
        else:  # PNG or other formats: lossless
            p = od / f"{name}_rgb.png"; Image.open(io.BytesIO(row[0])).convert("RGB").save(p)
        res["rgb"] = str(p)
    if row[1]:
        D = _decode_depth(row[1]); im = (np.clip(np.nan_to_num(D) / 4.0, 0, 1) * 255).astype(np.uint8)
        p = od / f"{name}_depth.png"; Image.fromarray(im).save(p); res["depth"] = str(p)
        v = D[np.isfinite(D) & (D > 0)]; res["depth_m"] = [round(float(v.min()), 3), round(float(v.max()), 3)] if len(v) else None
    if not res:
        raise ValueError(f"El fotograma {frame_id} no tiene imagen ni profundidad.")
    return _out(res)


BUILD_DEFAULTS = {"vox": 0.01, "min": 0.2, "max": 3.5, "step": 1, "level": True, "align": True,
                  "colors": True, "poses": "optimized", "conf": "low"}


@mcp.tool(title="RTAB-Map scan: build a point cloud", annotations=WRITES_FILES, structured_output=False)
def rtabmap_build_cloud(
    db_path: DbPath,
    out_path: Annotated[str | None, Field(description="Output path without extension: writes <out_path>.npz and <out_path>.ply. "
                                                      "Default: a private temporary folder, named after the options that differ from the defaults.")] = None,
    voxel_size: Annotated[float, Field(ge=0, description="Keep one point per cube of this size, in m (0.01 = 1 cm). 0 keeps every point (much bigger files).")] = 0.01,
    min_depth: Annotated[float, Field(ge=0, description="Ignore LiDAR readings closer than this, in m.")] = 0.2,
    max_depth: Annotated[float, Field(gt=0, description="Ignore LiDAR readings farther than this, in m. The iPhone/iPad LiDAR is most accurate up to about 3.5 m; 5 m reaches more of a big room.")] = 3.5,
    pixel_step: Annotated[int, Field(ge=1, description="Use every n-th depth pixel in each direction (2 = 4x fewer points, faster).")] = 1,
    level_floor: Annotated[bool, Field(description="Fit the floor plane and put it at z=0 with z up; the x/y origin becomes the centre of the scanned floor. False keeps RTAB-Map's world frame.")] = True,
    align_walls: Annotated[bool, Field(description="With level_floor: also rotate about z so the main walls run along x and y.")] = True,
    colors: Annotated[bool, Field(description="Colour each point from the frame photos.")] = True,
    write_ply: Annotated[bool, Field(description="Also write a .ply to open in Blender, MeshLab or CloudCompare.")] = True,
    poses: Annotated[Literal["optimized", "odometry"], Field(description="'optimized': RTAB-Map's graph with its loop closures, skipping the frames it dropped (recommended). "
                                                                         "'odometry': raw per-frame poses, as in clouds built before 2026-11-13.")] = "optimized",
    min_confidence: Annotated[Literal["low", "medium", "high"], Field(description="Drop LiDAR pixels below this ARKit confidence. 'low' keeps them all; 'medium' drops a few % (edges, dark or shiny surfaces).")] = "low",
) -> str:
    """Fuse the depth frames of a scan into one point cloud in metres and save it as .npz (the
    input of the other tools) plus an optional .ply. By default the floor goes to z=0 with z up
    and the main walls run along x and y. Returns the paths, the number of points, the bounding
    box, the estimated ceiling height and how the cloud was levelled. Takes a few seconds.
    x/y positions depend on what was scanned (origin = centre of the scanned floor): compare
    coordinates only within one cloud; distances and heights are comparable across clouds."""
    if voxel_size < 0 or pixel_step < 1 or not 0 <= min_depth < max_depth:
        raise ValueError("Valores fuera de rango: voxel_size >= 0, pixel_step >= 1 y 0 <= min_depth < max_depth.")
    if poses not in ("optimized", "odometry"):
        raise ValueError("poses tiene que ser 'optimized' u 'odometry'.")
    if min_confidence not in CONFIDENCE:
        raise ValueError("min_confidence tiene que ser 'low', 'medium' o 'high'.")
    if out_path:
        out = _out_base(out_path, ".npz", ".ply")
    else:
        opts = {"vox": voxel_size, "min": min_depth, "max": max_depth, "step": pixel_step, "level": level_floor,
                "align": align_walls, "colors": colors, "poses": poses, "conf": min_confidence}
        out = _default_dir(db_path) / ("cloud" + "".join(f"_{k}-{v}" for k, v in opts.items() if v != BUILD_DEFAULTS[k]))
    out.parent.mkdir(parents=True, exist_ok=True)
    thr = CONFIDENCE[min_confidence]
    with closing(_db(db_path)) as con:
        pose_of, source = (_optimized_poses(con) if poses == "optimized" else None), "optimized"
        if not pose_of:
            source = "odometry" if poses == "odometry" else "odometry (la base no guarda poses optimizadas)"
            pose_of = {i: T for i, T in ((i, _pose(p)) for i, p in con.execute("select id, pose from Node")) if T is not None}
        sessions = con.execute("select count(distinct map_id) from Node").fetchone()[0]
        has_conf = "depth_confidence" in {r[1] for r in con.execute("pragma table_info(Data)")}
        conf_col = "depth_confidence" if thr and has_conf else "null"
        Ps, Cs, used, total, kept, skipped, unfiltered = [], [], 0, 0, 0, [], 0
        for fid, img, depth, cal, conf in con.execute(
                f"select id, image, depth, calibration, {conf_col} from Data where depth is not null order by id"):
            if fid not in pose_of:
                skipped.append(fid); continue
            P, (u, v, (h, w)) = _frame_points(depth, cal, pose_of[fid], min_depth, max_depth, pixel_step)
            if thr:
                cf = _confidence(conf)
                if cf is not None and cf.shape == (h, w):
                    keep = cf[v, u] >= thr; P, u, v = P[keep], u[keep], v[keep]
                else:
                    unfiltered += 1
            if colors:
                if img:
                    I = np.asarray(Image.open(io.BytesIO(img)).convert("RGB"))
                    iu = np.clip((u + 0.5) * I.shape[1] / w, 0, I.shape[1] - 1).astype(int)
                    iv = np.clip((v + 0.5) * I.shape[0] / h, 0, I.shape[0] - 1).astype(int)
                    Cs.append(I[iv, iu])
                else:  # frame without image: grey, so colours stay aligned with the points
                    Cs.append(np.full((len(P), 3), 128, np.uint8))
            Ps.append(P); used += 1; total += len(P)
            # Keep memory bounded. Voxelizing in chunks gives the same cloud as doing it once
            # (every voxel keeps its earliest point either way); only re-voxelize once enough
            # new points arrived, so a big voxelized set is not re-sorted on every frame.
            if voxel_size > 0 and total - kept > max(4_000_000, kept):
                P_ = np.concatenate(Ps); C_ = np.concatenate(Cs) if Cs else None
                r = _voxel(P_, voxel_size, *( [C_] if C_ is not None else []))
                Ps = [r[0]]; Cs = [r[1]] if C_ is not None else []
                total = kept = len(r[0]); del P_, C_, r
    P = np.concatenate(Ps) if Ps else np.empty((0, 3))
    if not len(P): raise ValueError("No se generó ningún punto (revisa min_depth / max_depth).")
    C = np.concatenate(Cs) if Cs else None
    if voxel_size > 0:
        r = _voxel(P, voxel_size, *([C] if C is not None else [])); P = r[0]; C = r[1] if C is not None else None
    info = {"frames_used": used, "poses": source}
    if skipped:
        why = "fuera del grafo optimizado (RTAB-Map las descartó)" if source == "optimized" else "sin pose"
        info["frames_skipped"] = f"{len(skipped)}: {_ranges(skipped)} ({why})"
    if thr:
        info["min_confidence"] = min_confidence
        if unfiltered: info["frames_without_confidence"] = unfiltered
    if source != "optimized" and sessions > 1:
        info["warning"] = f"{sessions} sesiones (map_id): con poses de odometría no quedan alineadas entre sí."
    M = np.eye(4)
    if level_floor:
        P, M, li = _level_and_align(P, align_walls); info.update(li)
    np.savez_compressed(f"{out}.npz", points=P.astype(np.float32),
                        **({"colors": C.astype(np.uint8)} if C is not None else {}),
                        transform=M, source=str(Path(db_path).expanduser()), poses=source)
    info["npz"] = f"{out}.npz"
    if write_ply:
        _write_ply(Path(f"{out}.ply"), P, C); info["ply"] = f"{out}.ply"
    info.update(_summ(P))
    if level_floor:  # centre of the densest 2 cm band in the upper half
        zz = P[:, 2]; hh, ee = np.histogram(zz, bins=np.arange(0, zz.max() + 0.02, 0.02))
        info["ceiling_estimate_m"] = round(float(ee[np.argmax(hh[len(hh)//2:]) + len(hh)//2]) + 0.01, 2) if len(hh) > 4 else None
    walls = level_floor and align_walls and info.get("yaw_deg") is not None
    info["axes"] = ("metres; z up, floor at z=0" + (", main walls along x and y" if walls else "")
                    + ", x/y origin = centre of the scanned floor") if level_floor else \
        "metres; RTAB-Map world frame (x forward, y left, z up, origin = first camera position)"
    return _out(info)


PALETTE = [((255, 60, 60), "red"), ((60, 255, 60), "green"), ((60, 120, 255), "blue"),
           ((255, 220, 0), "yellow"), ((255, 0, 255), "magenta"), ((0, 255, 255), "cyan")]


@mcp.tool(title="Point cloud: top-down plan image", annotations=WRITES_FILES, structured_output=False)
def rtabmap_plan_image(
    cloud_npz: CloudPath,
    out_png: Annotated[str | None, Field(description="Output .png path. Default: next to the cloud.")] = None,
    resolution: Annotated[float, Field(gt=0, description="Metres per pixel (0.005 = 5 mm, 0.01 = 1 cm). At most 25 Mpx.")] = 0.005,
    bands: Annotated[list[list[float]] | None, Field(description="Up to 6 height bands [zmin, zmax] in m, drawn red, green, blue, yellow, magenta and cyan; later bands are drawn on top. "
                                                                 "Default [[0.15, 0.35], [0.40, 0.55], [0.58, 1.1]] (front faces, seat level and backs of a sofa); "
                                                                 "for a whole room e.g. [[0.05, 0.6], [0.6, 1.3], [1.3, 2.05]].")] = None,
    grid: Annotated[float, Field(gt=0, description="Spacing of the thin grid lines in m; bright lines every 0.5 m.")] = 0.1,
    x_range: Annotated[list[float] | None, Field(description="[x_min, x_max] in m to crop the view. Default: the whole cloud.")] = None,
    y_range: Annotated[list[float] | None, Field(description="[y_min, y_max] in m to crop the view. Default: the whole cloud.")] = None,
    labels: Annotated[bool, Field(description="Write the coordinate of each bright 0.5 m line along the top (x) and left (y) edges.")] = True,
) -> str:
    """Draw a cloud seen from above as a PNG: +x to the right, +y up, black background, each
    height band in its own colour (points outside every band are not drawn), thin grid lines
    every `grid` m and bright, labelled ones every 0.5 m. Returns the PNG path, the area it
    covers and the scale; open the image to see the layout, then measure with
    rtabmap_axis_histogram and rtabmap_height_grid."""
    p, P, _, _ = _load_cloud(cloud_npz)
    bands = bands or [[0.15, 0.35], [0.40, 0.55], [0.58, 1.1]]
    if len(bands) > len(PALETTE) or any(len(b) != 2 or not b[0] < b[1] for b in bands):
        raise ValueError(f"bands: hasta {len(PALETTE)} bandas [zmin, zmax] con zmin < zmax.")
    if resolution <= 0 or grid <= 0:
        raise ValueError("resolution y grid tienen que ser > 0.")
    if not len(P):
        raise ValueError("La nube está vacía.")
    x0, x1 = _range(x_range, "x_range") or (float(P[:, 0].min()), float(P[:, 0].max()))
    y0, y1 = _range(y_range, "y_range") or (float(P[:, 1].min()), float(P[:, 1].max()))
    w = int((x1 - x0) / resolution) + 1; h = int((y1 - y0) / resolution) + 1
    if w * h > 25_000_000: raise ValueError("Imagen demasiado grande: sube resolution o recorta.")
    I = np.zeros((h, w, 3), np.uint8)
    for g, col in ((grid, 50), (0.5, 120)):
        for xv in np.arange(np.ceil(x0 / g) * g, x1, g): I[:, int((xv - x0) / resolution)] = col
        for yv in np.arange(np.ceil(y0 / g) * g, y1, g): I[h - 1 - int((yv - y0) / resolution), :] = col
    for (a, b), (col, _) in zip(bands, PALETTE):
        Q = P[(P[:, 2] > a) & (P[:, 2] < b) & (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
        # cell = floor(), like the grid lines (the old int(h-1-y) put points one row too high)
        I[h - 1 - np.floor((Q[:, 1] - y0) / resolution).astype(int), np.floor((Q[:, 0] - x0) / resolution).astype(int)] = col
    im = Image.fromarray(I)
    if labels:
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default(size=11)
        except TypeError:  # Pillow < 10.1: fixed bitmap font
            font = ImageFont.load_default()
        txt = {"fill": (235, 235, 235), "font": font, "stroke_width": 2, "stroke_fill": (0, 0, 0)}
        for xv in np.arange(np.ceil(x0 / 0.5) * 0.5, x1, 0.5):
            dr.text((int((xv - x0) / resolution) + 3, 2), f"x={_nz(xv):g}", **txt)
        for yv in np.arange(np.ceil(y0 / 0.5) * 0.5, y1, 0.5):
            dr.text((3, h - 1 - int((yv - y0) / resolution) - 14), f"y={_nz(yv):g}", **txt)
    if out_png:
        o = Path(out_png).expanduser()
        if o.suffix.lower() != ".png":
            o = o.with_name(o.name + ".png")  # always lossless PNG
    else:
        o = p.with_name(f"{p.stem}_plan-{_short_hash(resolution=resolution, bands=bands, grid=grid, x=[x0, x1], y=[y0, y1], labels=labels)}.png")
    o = o.resolve(); o.parent.mkdir(parents=True, exist_ok=True)
    im.save(o)
    return _out({"png": str(o), "x_min": round(x0, 3), "x_max": round(x1, 3), "y_min": round(y0, 3), "y_max": round(y1, 3),
                 "px_per_m": round(1 / resolution), "size_px": [w, h], "bands": bands,
                 "colors": [n for _, n in PALETTE[:len(bands)]]})


@mcp.tool(title="Point cloud: table of surface heights", annotations=READ_ONLY, structured_output=False)
def rtabmap_height_grid(
    cloud_npz: CloudPath,
    x_range: Annotated[list[float], Field(description="[x_min, x_max] in m: one row every `step` (rows = x).")],
    y_range: Annotated[list[float], Field(description="[y_min, y_max] in m: one column every `step` (columns = y).")],
    step: Annotated[float, Field(gt=0, description="Cell size in m; each value covers the cell centred on its x, y (± step/2). At most 4000 cells.")] = 0.05,
    percentile: Annotated[float, Field(ge=0, le=100, description="Height given for each cell: this percentile of its points' z (98 ≈ top surface without stray points, 50 = median).")] = 98,
    z_max: Annotated[float, Field(description="Ignore points higher than this, in m (e.g. a lamp or the ceiling above the furniture).")] = 2.0,
) -> str:
    """Text table of the surface height seen from above, cell by cell: for each cell the
    `percentile` of the heights of its points, in m, ignoring points above z_max. Rows are x,
    columns are y, '-' means fewer than 5 points in the cell. Use it to read seat, armrest and
    back heights, depths and where surfaces end."""
    xr, yr = _range(x_range, "x_range"), _range(y_range, "y_range")
    if step <= 0 or not 0 <= percentile <= 100:
        raise ValueError("step tiene que ser > 0 y percentile estar entre 0 y 100.")
    _, P, _, _ = _load_cloud(cloud_npz)
    P = P[P[:, 2] < z_max]
    xs = np.arange(xr[0], xr[1] + 1e-9, step); ys = np.arange(yr[0], yr[1] + 1e-9, step)
    if len(xs) * len(ys) > 4000: raise ValueError("Demasiadas celdas (>4000): sube step o reduce el rango.")
    ix = np.round((P[:, 0] - xs[0]) / step).astype(int); iy = np.round((P[:, 1] - ys[0]) / step).astype(int)
    m = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    ix, iy, z = ix[m], iy[m], P[m, 2]
    order = np.lexsort((z, iy, ix)); ix, iy, z = ix[order], iy[order], z[order]
    key = ix * len(ys) + iy
    G = np.full((len(xs), len(ys)), np.nan)
    uk, start, cnt = np.unique(key, return_index=True, return_counts=True)
    for k, s, c in zip(uk, start, cnt):
        if c >= 5: G[k // len(ys), k % len(ys)] = z[s + min(c - 1, int(round(percentile / 100 * (c - 1))))]
    lines = [f"# z in m = percentile {percentile:g} of the points of each {step:g} m cell centred on (x, y), "
             f"points above z={z_max:g} ignored; '-' = fewer than 5 points",
             "x \\ y   " + " ".join(f"{_nz(y):+6.2f}" for y in ys)]
    for i, x in enumerate(xs):
        lines.append(f"{_nz(x):+6.2f}  " + " ".join(" {:5.2f}".format(_nz(v)) if np.isfinite(v) else "   -  " for v in G[i]))
    return "\n".join(lines)


def _hist_text(c, h) -> str:
    """'centre:count' pairs; 3 or more empty bins in a row become 'from..to:0'."""
    out, i = [], 0
    while i < len(h):
        j = i
        while h[i] == 0 and j + 1 < len(h) and h[j + 1] == 0: j += 1
        if j - i >= 2:
            out.append(f"{c[i]:+.3f}..{c[j]:+.3f}:0"); i = j + 1
        else:
            out.append(f"{c[i]:+.3f}:{h[i]}"); i += 1
    return " ".join(out)


@mcp.tool(title="Point cloud: histogram along an axis", annotations=READ_ONLY, structured_output=False)
def rtabmap_axis_histogram(
    cloud_npz: CloudPath,
    axis: Annotated[str, Field(description="'x', 'y' or 'z': the axis to count along.")],
    box_min: Annotated[list[float], Field(description="[x, y, z] lower corner of the box, in m.")],
    box_max: Annotated[list[float], Field(description="[x, y, z] upper corner of the box, in m.")],
    bin_size: Annotated[float, Field(gt=0, description="Bin width in m (at most 2000 bins).")] = 0.01,
) -> str:
    """Count the points inside a box along one axis. A surface across that axis (a wall, the
    front of a sofa, the side of an armrest, the floor, the ceiling) is a peak at its coordinate.
    Returns the points in the box, the 8 strongest peaks as [bin centre, points] and the whole
    histogram as 'bin_centre:count' (runs of empty bins as 'from..to:0')."""
    ax = str(axis).strip().lower()
    if ax not in ("x", "y", "z"):
        raise ValueError(f"axis tiene que ser 'x', 'y' o 'z' (recibido {axis!r}).")
    a = "xyz".index(ax)
    lo, hi = _vec(box_min, "box_min"), _vec(box_max, "box_max")
    if np.any(lo >= hi):
        raise ValueError("box_min tiene que ser menor que box_max en x, y y z.")
    if bin_size <= 0 or (hi[a] - lo[a]) / bin_size > 2000:
        raise ValueError("bin_size tiene que ser > 0 y dar como mucho 2000 intervalos.")
    _, P, _, _ = _load_cloud(cloud_npz)
    Q = P[np.all((P >= lo) & (P <= hi), axis=1)]
    if not len(Q): return "Sin puntos en esa caja."
    h, e = np.histogram(Q[:, a], bins=np.arange(lo[a], hi[a] + bin_size, bin_size))
    c = (e[:-1] + e[1:]) / 2
    peaks = [(round(float(c[i]), 3), int(h[i])) for i in range(len(h))
             if h[i] > 0 and h[i] >= h[max(i - 1, 0)] and h[i] >= h[min(i + 1, len(h) - 1)]]
    peaks = sorted(peaks, key=lambda t: -t[1])[:8]
    return _out({"points_in_box": int(len(Q)), "peaks": peaks, "histogram": _hist_text(c, h)})


@mcp.tool(title="Point cloud: crop, rotate and move", annotations=WRITES_FILES, structured_output=False)
def rtabmap_transform_cloud(
    cloud_npz: CloudPath,
    out_path: Annotated[str | None, Field(description="Output path without extension: writes <out_path>.npz (+ .ply). Default: next to the input cloud.")] = None,
    crop_min: Annotated[list[float] | None, Field(description="[x, y, z] in m: drop points below this corner (input cloud's coordinates).")] = None,
    crop_max: Annotated[list[float] | None, Field(description="[x, y, z] in m: drop points above this corner (input cloud's coordinates).")] = None,
    rotate_z_deg: Annotated[float, Field(description="Rotation about +z in degrees, counter-clockwise seen from above; 180 turns a sofa facing +y to face -y.")] = 0.0,
    translate: Annotated[list[float] | None, Field(description="[dx, dy, dz] in m, added at the end.")] = None,
    mirror_x: Annotated[bool, Field(description="Flip x (x -> -x) after the rotation: gives the mirror image.")] = False,
    voxel_size: Annotated[float, Field(ge=0, description="After the crop, keep one point per cube of this size in m; 0 keeps all.")] = 0.0,
    write_ply: Annotated[bool, Field(description="Also write a .ply.")] = True,
) -> str:
    """Cut a piece out of a cloud and place it in another scene's coordinates (e.g. a Blender
    file). Steps, in this order: crop to [crop_min, crop_max], voxel downsample, rotate about z,
    mirror x, translate. Writes a new .npz (usable by the other tools) that also keeps the
    accumulated transform from the scan, plus an optional .ply."""
    cmin, cmax, tr = _vec(crop_min, "crop_min"), _vec(crop_max, "crop_max"), _vec(translate, "translate")
    if voxel_size < 0:
        raise ValueError("voxel_size tiene que ser >= 0.")
    src, P, C, meta = _load_cloud(cloud_npz)
    m = np.ones(len(P), bool)
    if cmin is not None: m &= np.all(P >= cmin, axis=1)
    if cmax is not None: m &= np.all(P <= cmax, axis=1)
    P = P[m]; C = C[m] if C is not None else None
    if not len(P):
        raise ValueError("El recorte no deja ningún punto: revisa crop_min / crop_max.")
    if voxel_size > 0:
        r = _voxel(P, voxel_size, *([C] if C is not None else [])); P = r[0]; C = r[1] if C is not None else None
    a = np.radians(rotate_z_deg); ca, sa = np.cos(a), np.sin(a)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    P = P @ Rz.T
    if mirror_x: P[:, 0] *= -1
    if tr is not None: P = P + tr
    A = np.eye(4); A[:3, :3] = Rz  # same steps as a 4x4 matrix, to keep the chain from the scan
    if mirror_x: A[0, :3] *= -1
    if tr is not None: A[:3, 3] = tr
    extra = {k: meta[k] for k in ("source", "poses") if k in meta}
    if "transform" in meta:
        extra["transform"] = A @ meta["transform"]
    if out_path:
        out = _out_base(out_path, ".npz", ".ply")
    else:
        out = src.with_name(f"{src.stem}_tf-{_short_hash(crop_min=crop_min, crop_max=crop_max, rotate_z_deg=rotate_z_deg, translate=translate, mirror_x=mirror_x, voxel_size=voxel_size)}").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{out}.npz", points=P.astype(np.float32), **({"colors": C} if C is not None else {}), **extra)
    res = {"npz": f"{out}.npz", **_summ(P)}
    if write_ply: _write_ply(Path(f"{out}.ply"), P, C); res["ply"] = f"{out}.ply"
    return _out(res)


if __name__ == "__main__":
    mcp.run()
