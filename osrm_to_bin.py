#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "lightgbm>=4.7.0",
#   "numpy>=2.5.3",
#   "onnx>=1.23.0",
#   "onnxruntime>=1.30.0",
#   "osmium>=4.3.1",
#   "pandas>=3.0.6",
#   "pyarrow>=25.0.1",
#   "requests>=2.34.2",
#   "scikit-learn>=1.9.1",
# ]
# ///
"""End-to-end: an OSM extract URL -> ``server/models/model.bin`` in one command.

This is the "all included" driver for the whole offline pipeline. Given a link to
an OSM extract (``.osm.pbf``), it:

1. downloads the extract into a per-URL cache folder (and reuses it next time);
2. builds the OSRM MLD graph in a throwaway Docker container;
3. serves it, samples the multi-resolution grid + random off-network points, and
   asks OSRM for the ground-truth ``(distance_m, duration_s)`` labels;
4. trains the two LightGBM regressors and exports ``model.onnx``;
5. re-serialises the trees into the compact ``model.bin`` the .NET server reads;
6. copies the artefacts into ``server/models/``.

Every stage is cached on disk, so re-running with the **same URL** is cheap: the
downloaded extract, OSRM graph, samples and trained model are all reused and only
the final copy is performed. Changing a training knob (leaves, trees, ...) with
the same URL reuses the data and retrains; pass ``--force`` to rebuild everything.

Examples
--------
    # Greater Manchester (the repo's reference region)
    uv run osrm_to_bin.py \\
        https://download.geofabrik.de/europe/united-kingdom/england/greater-manchester-latest.osm.pbf

    # A smaller model, custom cache, explicit bbox (skips header sniffing)
    uv run osrm_to_bin.py https://.../region-latest.osm.pbf \\
        --num-leaves 127 --bbox 53.3,-2.8,53.7,-1.9

    # Reuse / rebuild
    uv run osrm_to_bin.py <same-url>        # instant: everything cached
    uv run osrm_to_bin.py <same-url> --force
    uv run osrm_to_bin.py <same-url> --redownload

Requirements: Docker, ``uv`` and the network. The downloaded extract is cached
under ``~/.cache/osrm-approx/<host>-<url-hash>/`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

# --------------------------------------------------------------------------------------
# Wire the sibling pipeline modules together (they remain the single source of truth
# for each stage: grid build, OSRM /table fetch, off-network sampling, training, export).
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import export_binary  # noqa: E402
import fetch_osrm_matrix  # noqa: E402
import generate_grid  # noqa: E402
import gen_offnetwork  # noqa: E402
import train_export_onnx  # noqa: E402

DEFAULT_CACHE = Path(os.environ.get("OSRM_APPROX_CACHE", Path.home() / ".cache" / "osrm-approx"))
DEFAULT_OSRM_PORT = 5001
OSRM_INTERNAL_PORT = 5000
MAX_TABLE_SIZE = 10_000


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[osrm-to-bin] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    """Echo and execute a command, raising on failure."""
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def call_main(module, argv: list[str]) -> int:
    """Invoke a sibling ``main()`` with a synthetic argv.

    The pipeline scripts each own an argparse parser that reads ``sys.argv``; this
    lets us drive them unchanged and keeps them runnable standalone.
    """
    saved = sys.argv
    sys.argv = [module.__name__, *argv]
    try:
        return module.main()
    finally:
        sys.argv = saved


def _docker_user_args() -> list[str]:
    """Run the OSRM container as the invoking user so cache files stay user-owned."""
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return ["--user", f"{os.getuid()}:{os.getgid()}"]
    return []


def docker_available() -> bool:
    return shutil.which("docker") is not None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def find_free_port(preferred: int) -> int:
    """Return ``preferred`` if it is free, else the next free port, else an ephemeral one."""
    if _port_is_free(preferred):
        return preferred
    for candidate in range(preferred + 1, preferred + 51):
        if _port_is_free(candidate):
            log(f"osrm: port {preferred} is busy; using {candidate}")
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        candidate = int(probe.getsockname()[1])
    log(f"osrm: port {preferred} is busy; using ephemeral {candidate}")
    return candidate


# --------------------------------------------------------------------------------------
# Cache / region layout
# --------------------------------------------------------------------------------------
def region_dir_for(cache_dir: Path, url: str) -> Path:
    """Stable per-URL cache directory: same URL -> same folder -> full reuse."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    host = urlparse(url).netloc.replace(":", "_").replace(".", "_") or "local"
    return cache_dir / f"{host}-{digest}"


def pbf_filename_for(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name or not name.endswith((".pbf", ".osm.pbf")):
        name = "region.osm.pbf"
    return name


def find_osrm_graph(region: Path) -> Path | None:
    """The OSRM graph is the single ``*.osrm`` file (not the ``*.osrm.*`` sidecars)."""
    graphs = sorted(region.glob("*.osrm"))
    return graphs[0] if graphs else None


def reset_region(region: Path, pbf_name: str, keep_pbf: bool) -> None:
    """Delete derived artefacts (and optionally the download) before a forced rebuild."""
    if not region.exists():
        return
    for path in region.iterdir():
        if keep_pbf and path.name == pbf_name:
            continue
        log(f"reset: removing {path.name}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


# --------------------------------------------------------------------------------------
# Step 1 — download (with progress, atomic, resumable-by-presence)
# --------------------------------------------------------------------------------------
def download_pbf(url: str, dest: Path, redownload: bool) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not redownload:
        log(f"download: reusing {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()

    log(f"download: {url}")
    headers = {"User-Agent": "osrm-approx/osrm_to_bin"}
    started = time.time()
    with requests.get(url, stream=True, timeout=120, headers=headers) as response:
        response.raise_for_status()
        ctype = response.headers.get("content-type", "")
        if ctype.startswith("text/"):
            raise RuntimeError(f"{url} returned {ctype!r}, not an OSM extract")
        total = int(response.headers.get("content-length", 0) or 0)
        done = 0
        next_report = 25
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    if pct >= next_report:
                        log(f"download: {pct:5.1f}% ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")
                        next_report += 25

    if done < 1024:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded only {done} bytes from {url}; aborting")
    # Real OSM PBF files open with a BlobHeader of type "OSMHeader"; catch an
    # HTML/proxy error page served with a binary content-type.
    with tmp.open("rb") as handle:
        head = handle.read(64)
    if b"OSMHeader" not in head:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url} is not an OSM PBF (missing OSMHeader magic); aborting")
    tmp.replace(dest)
    log(f"download: wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB in {time.time() - started:.0f}s)")
    return dest


# --------------------------------------------------------------------------------------
# Step 2 — derive the region bbox, then size the multi-resolution grid to it
# --------------------------------------------------------------------------------------
def derive_bbox(pbf_path: Path, override: tuple | None) -> tuple[float, float, float, float]:
    """Return ``(min_lat, min_lon, max_lat, max_lon)``.

    Prefers the PBF header's bounding box (O(1)); falls back to scanning every node.
    """
    if override is not None:
        log(f"bbox: using --bbox {override}")
        return override

    try:
        import osmium  # imported lazily so --bbox users never need it

        reader = osmium.io.Reader(str(pbf_path))
        try:
            box = reader.header().box()
            if box.valid():
                # `bottom_left`/`top_right` are properties in pyosmium, not methods.
                bl, tr = box.bottom_left, box.top_right
                bbox = (float(bl.lat), float(bl.lon), float(tr.lat), float(tr.lon))
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    log(f"bbox: from PBF header {bbox}")
                    return bbox
        finally:
            reader.close()
    except Exception as exc:  # noqa: BLE001 - fall through to the scan
        log(f"bbox: header read failed ({exc}); scanning nodes")

    import osmium

    class _BBox(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.min_lat, self.max_lat = 90.0, -90.0
            self.min_lon, self.max_lon = 180.0, -180.0
            self.n = 0

        def node(self, node) -> None:  # noqa: N802 - pyosmium API
            if not node.location.valid():
                return
            lat, lon = float(node.location.lat), float(node.location.lon)
            self.min_lat, self.max_lat = min(self.min_lat, lat), max(self.max_lat, lat)
            self.min_lon, self.max_lon = min(self.min_lon, lon), max(self.max_lon, lon)
            self.n += 1

    scan = _BBox()
    osmium.apply(str(pbf_path), scan)
    if scan.n == 0:
        raise RuntimeError("could not determine a bounding box (no valid nodes); pass --bbox")
    bbox = (scan.min_lat, scan.min_lon, scan.max_lat, scan.max_lon)
    log(f"bbox: scanned {scan.n:,} nodes -> {bbox}")
    return bbox


def compute_rings(bbox: tuple[float, float, float, float]) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Adapt the multi-resolution ring grid to any region.

    The outermost ring must cover the whole bbox: all four corners lie within
    ``R`` of the centre, and a disk is convex, so the full rectangle does too.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    centre = ((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)
    corners = [(min_lat, min_lon), (min_lat, max_lon), (max_lat, min_lon), (max_lat, max_lon)]
    radius_km = max(
        float(generate_grid.haversine_m(centre[0], centre[1], la, lo)) for la, lo in corners
    ) / 1000.0
    radius_km = max(radius_km * 1.05, 1.0)

    span = max(max_lat - min_lat, max_lon - min_lon)
    outer_step = max(0.01, span / 100.0)  # never finer than the reference ~1 km fringe
    candidates = [
        (min(3.0, radius_km), 0.0025),   # dense ~250 m core
        (min(10.0, radius_km), 0.005),   # ~500 m mid ring
        (radius_km, outer_step),         # ~1 km+ fringe
    ]
    rings: list[tuple[float, float]] = []
    for radius, step in candidates:
        if radius <= (rings[-1][0] if rings else 0.0) + 1e-9:
            continue
        rings.append((radius, step))
    if not rings:
        rings = [(radius_km, outer_step)]
    log("rings: " + ", ".join(f"<{r:.1f}km@{s:g}deg" for r, s in rings))
    return rings, centre


def patch_region_globals(bbox: tuple[float, float, float, float], centre: tuple[float, float]) -> None:
    """Point the sibling generators at this region instead of Greater Manchester."""
    min_lat, min_lon, max_lat, max_lon = bbox
    generate_grid.BBOX = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}
    generate_grid.CITY_CENTRE = centre
    gen_offnetwork.BBOX = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}


# --------------------------------------------------------------------------------------
# Step 3 — Docker: pre-process the graph, then serve it
# --------------------------------------------------------------------------------------
def build_osrm_graph(region: Path, pbf: Path, image: str) -> Path:
    if not docker_available():
        raise RuntimeError("Docker is required to pre-process the extract (or pass --osrm-url)")
    pbf_name = shlex.quote(pbf.name)
    script = (
        "set -e; cd /data; "
        f"osrm-extract -p /opt/car.lua {pbf_name}; "
        "f=$(ls *.osrm | head -n1); "
        'osrm-partition "$f"; osrm-customize "$f"; '
        "echo OSRM_PREPROCESS_DONE"
    )
    log("docker: osrm-extract / partition / customize")
    run([
        "docker", "run", "--rm",
        *_docker_user_args(),
        "-v", f"{region}:/data",
        "--entrypoint", "sh",
        image, "-lc", script,
    ])
    graph = find_osrm_graph(region)
    if graph is None:
        raise RuntimeError("OSRM pre-processing finished but no *.osrm file was produced")
    log(f"graph: {graph.name} ({graph.stat().st_size / 1e6:.1f} MB)")
    return graph


def start_osrm(region: Path, graph: Path, image: str, port: int, max_table_size: int) -> str:
    name = f"osrm-approx-{hashlib.sha1(str(region).encode()).hexdigest()[:10]}"
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    run([
        "docker", "run", "-d", "--rm", "--name", name,
        *_docker_user_args(),
        "-p", f"{port}:{OSRM_INTERNAL_PORT}",
        "-v", f"{region}:/data",
        "--entrypoint", "osrm-routed",
        image,
        "--algorithm", "mld",
        "--max-table-size", str(max_table_size),
        "--port", str(OSRM_INTERNAL_PORT),
        f"/data/{graph.name}",
    ])
    log(f"osrm: container {name} listening on :{port}")
    return name


def stop_osrm(name: str) -> None:
    log(f"osrm: stopping {name}")
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)


def wait_for_osrm(url: str, centre: tuple[float, float], timeout_s: float = 180.0) -> None:
    lat, lon = centre
    probe = f"{url}/nearest/v1/driving/{lon:.6f},{lat:.6f}?number=1"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(probe, timeout=5)
            if response.ok and response.json().get("code") == "Ok":
                log(f"osrm: healthy at {url}")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"OSRM did not become healthy at {url} within {timeout_s:.0f}s")


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pbf_url", help="URL of the .osm.pbf extract to download and learn from")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help=f"cache root (default: {DEFAULT_CACHE})")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "server" / "models" / "model.bin",
                        help="where to copy the compiled model (default: server/models/model.bin)")
    parser.add_argument("--bbox", default=None,
                        help="'min_lat,min_lon,max_lat,max_lon'; derived from the PBF header when omitted")
    parser.add_argument("--osrm-url", default=None,
                        help="use an already-running OSRM instead of building/serving one (skips Docker)")
    parser.add_argument("--osrm-port", type=int, default=DEFAULT_OSRM_PORT, help="host port for the OSRM container")
    parser.add_argument("--osrm-image", default="osrm/osrm-backend:latest")

    parser.add_argument("--grid-points", type=int, default=3000, help="sampled grid nodes (all-pairs source)")
    parser.add_argument("--matrix-max-coords", type=int, default=1200, help="sources+destinations per /table request")
    parser.add_argument("--max-snap-m", type=float, default=250.0, help="drop grid points further than this from a road")
    parser.add_argument("--off-points", type=int, default=400, help="random off-network coordinates (0 disables)")
    parser.add_argument("--off-block", type=int, default=25, help="origins x destinations per off-network /table block")

    parser.add_argument("--num-leaves", type=int, default=511)
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--weighting", choices=["none", "inv_freq"], default="inv_freq")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0, help="0 lets LightGBM/NumPy choose")

    parser.add_argument("--force", action="store_true", help="ignore cached data/model and rebuild (keeps the download)")
    parser.add_argument("--redownload", action="store_true", help="also re-download the extract")
    parser.add_argument("--keep-osrm-running", action="store_true", help="leave the OSRM container up after the run")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.bbox:
        try:
            override = tuple(float(v) for v in args.bbox.split(","))
        except ValueError as exc:
            raise SystemExit(f"--bbox must be 'min_lat,min_lon,max_lat,max_lon': {exc}") from None
        if len(override) != 4 or not (override[0] < override[2] and override[1] < override[3]):
            raise SystemExit("--bbox must be 'min_lat,min_lon,max_lat,max_lon' with min < max")
    else:
        override = None

    region = region_dir_for(args.cache_dir, args.pbf_url)
    region.mkdir(parents=True, exist_ok=True)
    pbf_name = pbf_filename_for(args.pbf_url)
    pbf = region / pbf_name
    grid = region / "grid.parquet"
    samples = region / "samples.parquet"
    offnetwork = region / "offnetwork.parquet"
    onnx = region / "model.onnx"
    binary = region / "model.bin"
    metadata = region / "model_metadata.json"
    manifest_path = region / "manifest.json"

    log(f"cache: {region}")

    # ---- decide what to reuse ---------------------------------------------------------
    wanted_data = {
        "url": args.pbf_url,
        "bbox_override": list(override) if override else None,
        "grid_points": args.grid_points,
        "max_snap_m": args.max_snap_m,
        "off_points": args.off_points,
    }
    wanted_train = {
        "num_leaves": args.num_leaves,
        "estimators": args.estimators,
        "learning_rate": args.learning_rate,
        "weighting": args.weighting,
        "seed": args.seed,
    }
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    data_match = manifest.get("data") == wanted_data
    train_match = manifest.get("train") == wanted_train

    if args.force or args.redownload:
        reset_region(region, pbf_name, keep_pbf=not args.redownload)

    if not args.force and not args.redownload and data_match and train_match and binary.exists():
        log("reuse: cached model matches this URL/configuration — nothing to do")
        _copy_outputs(region, args.out)
        return 0

    if args.force or args.redownload:
        data_match = train_match = False
    if not data_match:
        # The labels changed, so the model trained on the old ones is stale too.
        train_match = False
        for stale in (grid, samples, offnetwork):
            stale.unlink(missing_ok=True)
    if not train_match:
        for stale in (onnx, binary, metadata):
            stale.unlink(missing_ok=True)

    # ---- step 1: download -------------------------------------------------------------
    (region / "source.url").write_text(args.pbf_url + "\n")
    download_pbf(args.pbf_url, pbf, redownload=args.redownload)

    # ---- step 2: region bbox + grid sizing --------------------------------------------
    bbox = derive_bbox(pbf, override)
    rings, centre = compute_rings(bbox)
    patch_region_globals(bbox, centre)

    # ---- data stages (need a live OSRM) ----------------------------------------------
    need_data = any(not path.exists() for path in (grid, samples, offnetwork))
    container: str | None = None
    osrm_url = args.osrm_url

    if need_data:
        if osrm_url is None:
            graph = find_osrm_graph(region)
            if graph is None or args.redownload:
                graph = build_osrm_graph(region, pbf, args.osrm_image)
            port = find_free_port(args.osrm_port)
            container = start_osrm(region, graph, args.osrm_image, port, MAX_TABLE_SIZE)
            osrm_url = f"http://localhost:{port}"
        wait_for_osrm(osrm_url, centre)

    try:
        if not grid.exists():
            log("stage: multi-resolution grid")
            call_main(generate_grid, [
                "--out", str(grid),
                "--max-points", str(args.grid_points),
                "--seed", str(args.seed),
                "--rings", ",".join(f"{r:g}:{step:g}" for r, step in rings),
            ])

        if not samples.exists():
            log("stage: OSRM /table ground truth")
            rc = call_main(fetch_osrm_matrix, [
                "--url", osrm_url,
                "--grid", str(grid),
                "--out", str(samples),
                "--max-coords", str(args.matrix_max_coords),
                "--max-snap-m", str(args.max_snap_m),
            ])
            if rc != 0:
                raise RuntimeError(f"fetch_osrm_matrix failed (exit {rc})")

        if not offnetwork.exists():
            if args.off_points > 0:
                log("stage: off-network samples")
                rc = call_main(gen_offnetwork, [
                    "--url", osrm_url,
                    "--points", str(args.off_points),
                    "--block", str(args.off_block),
                    "--seed", str(args.seed),
                    "--out", str(offnetwork),
                ])
                if rc != 0:
                    raise RuntimeError(f"gen_offnetwork failed (exit {rc})")
            else:
                log("stage: off-network samples disabled — writing an empty table")
                import pandas as pd

                pd.DataFrame(columns=[
                    "orig_lat", "orig_lon", "dest_lat", "dest_lon",
                    "osrm_distance_m", "osrm_duration_s",
                ]).to_parquet(offnetwork, index=False)
    finally:
        if container is not None and not args.keep_osrm_running:
            stop_osrm(container)

    # ---- step 4: train ----------------------------------------------------------------
    if not (onnx.exists() and binary.exists()):
        log("stage: LightGBM training + ONNX export")
        rc = call_main(train_export_onnx, [
            "--samples", str(samples),
            "--extra-samples", str(offnetwork),
            "--out", str(onnx),
            "--num-leaves", str(args.num_leaves),
            "--estimators", str(args.estimators),
            "--learning-rate", str(args.learning_rate),
            "--weighting", args.weighting,
            "--seed", str(args.seed),
            "--threads", str(args.threads),
        ])
        if rc != 0:
            raise RuntimeError(f"train_export_onnx failed (exit {rc})")

    # ---- step 5: compile to the server's binary format --------------------------------
    if not binary.exists():
        log("stage: compile ONNX trees -> model.bin")
        rc = call_main(export_binary, ["--onnx", str(onnx), "--out", str(binary)])
        if rc != 0:
            raise RuntimeError(f"export_binary failed (exit {rc})")

    # ---- record the manifest and publish ---------------------------------------------
    manifest = {
        "data": wanted_data,
        "train": wanted_train,
        "auto_bbox": list(bbox),
        "rings": [[r, step] for r, step in rings],
        "url": args.pbf_url,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _copy_outputs(region, args.out)
    log(f"done: {binary.stat().st_size / 1e6:.2f} MB model.bin")
    return 0


def _copy_outputs(region: Path, out: Path) -> None:
    """Publish model.bin (and its onnx/metadata siblings) next to ``--out``."""
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, target in (
        ("model.bin", out),
        ("model.onnx", out.with_name("model.onnx")),
        ("model_metadata.json", out.with_name("model_metadata.json")),
    ):
        source = region / name
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
            log(f"copy: {source.name} -> {target}")


if __name__ == "__main__":
    raise SystemExit(main())
