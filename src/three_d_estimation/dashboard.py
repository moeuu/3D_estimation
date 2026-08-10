"""Browser dashboard and URL serving for online surface MLE progress."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .types import MLEEstimate


DASHBOARD_DATA_FILENAME = "dashboard_data.json"
DASHBOARD_INDEX_FILENAME = "index.html"
DEFAULT_DASHBOARD_PORT = 8878
OVERVIEW_IMAGE_FILENAME = "latest_experiment_overview.png"
ROBOT_IMAGE_FILENAME = "latest_robot_2d.png"
MLE_IMAGE_FILENAME = "latest_mle_3d.png"
SPECTRUM_IMAGE_FILENAME = "latest_spectrum.png"

_HTTP_SERVERS: dict[tuple[str, int], ThreadingHTTPServer] = {}
_HTTP_SERVER_ROOTS: dict[tuple[str, int], Path] = {}
_HTTP_THREADS: list[threading.Thread] = []


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve static dashboard files without per-request terminal noise."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress the standard request log."""
        del format, args


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Durably replace one dashboard artifact."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Dashboard staging file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tcp_port_is_open(host: str, port: int) -> bool:
    """Return whether a local TCP endpoint is already accepting connections."""
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((connect_host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _default_public_host() -> str:
    """Return a likely browser-reachable host for dashboard URLs."""
    configured = os.environ.get(
        "MLE_DASHBOARD_PUBLIC_HOST",
        os.environ.get("CUI_SPLIT_VIEW_PUBLIC_HOST"),
    )
    if configured:
        return configured
    try:
        output = subprocess.check_output(
            ["hostname", "-I"],
            text=True,
            timeout=0.2,
        )
        candidates = [value for value in output.split() if value]
        for candidate in candidates:
            if candidate.startswith("100."):
                return candidate
        for candidate in candidates:
            if not candidate.startswith(("127.", "172.")):
                return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.2)
            probe.connect(("8.8.8.8", 80))
            candidate = str(probe.getsockname()[0])
            if candidate and not candidate.startswith("127."):
                return candidate
    except OSError:
        pass
    return "127.0.0.1"


def _available_dashboard_port(host: str, requested_port: int) -> int:
    """Return the first locally available dashboard port at or above a request."""
    for candidate in range(int(requested_port), min(int(requested_port) + 100, 65536)):
        if (host, candidate) not in _HTTP_SERVERS and not _tcp_port_is_open(
            host, candidate
        ):
            return candidate
    raise OSError(
        f"No dashboard port is available in {requested_port}.."
        f"{min(int(requested_port) + 99, 65535)}."
    )


def _dashboard_browser_url(public_host: str, port: int) -> str:
    """Return one explicit browser URL, including IPv6 brackets and index path."""
    display_host = str(public_host).strip()
    if not display_host:
        raise ValueError("Dashboard public host must be nonempty.")
    if "://" in display_host or "/" in display_host:
        raise ValueError("Dashboard public host must be a host name or IP address.")
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{int(port)}/index.html"


def ensure_dashboard_server(
    output_dir: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_DASHBOARD_PORT,
    public_host: str | None = None,
) -> str:
    """Start or reuse a persistent static server and return its browser URL."""
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dashboard output directory does not exist: {root}")
    parsed_port = int(port)
    if not 1 <= parsed_port <= 65535:
        raise ValueError("Dashboard port must lie in 1..65535.")
    display_host = (
        _default_public_host()
        if public_host is None and host in {"0.0.0.0", "::"}
        else str(public_host or host)
    )
    requested_key = (host, parsed_port)
    if requested_key in _HTTP_SERVERS and _HTTP_SERVER_ROOTS.get(requested_key) == root:
        return _dashboard_browser_url(display_host, parsed_port)
    selected_port = _available_dashboard_port(host, parsed_port)
    url = _dashboard_browser_url(display_host, selected_port)
    key = (host, selected_port)

    log_path = root / f"dashboard_server_{selected_port}.log"
    pid_path = root / f"dashboard_server_{selected_port}.pid"
    try:
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "http.server",
                    str(selected_port),
                    "--bind",
                    host,
                    "--directory",
                    root.as_posix(),
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        for _ in range(20):
            if _tcp_port_is_open(host, selected_port):
                return url
            if process.poll() is not None:
                break
            time.sleep(0.05)
    except OSError:
        pass

    handler = partial(_QuietHandler, directory=root.as_posix())
    server = ThreadingHTTPServer((host, selected_port), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="online-mle-dashboard",
        daemon=True,
    )
    thread.start()
    _HTTP_SERVERS[key] = server
    _HTTP_SERVER_ROOTS[key] = root
    _HTTP_THREADS.append(thread)
    return url


def _finite_float(value: object, *, fallback: float = 0.0) -> float:
    """Return a finite JSON float for dashboard display."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return parsed if np.isfinite(parsed) else float(fallback)


def _hotspot_payload(estimate: MLEEstimate) -> list[dict[str, object]]:
    """Return compact finite hotspot rows from estimate diagnostics."""
    raw = estimate.diagnostics.get("hotspot_clusters", [])
    if not isinstance(raw, list):
        return []
    hotspots: list[dict[str, object]] = []
    for value in raw:
        if not isinstance(value, Mapping):
            continue
        centroid = np.asarray(value.get("centroid_xyz", ()), dtype=float)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            continue
        hotspots.append(
            {
                "isotope": str(value.get("isotope", "unknown")),
                "cluster_id": int(value.get("cluster_id", len(hotspots))),
                "centroid_xyz": centroid.tolist(),
                "integrated_strength_cps_1m": _finite_float(
                    value.get("integrated_strength_cps_1m", 0.0)
                ),
                "peak_density_cps_1m_m2": _finite_float(
                    value.get("peak_density_cps_1m_m2", 0.0)
                ),
                "surface_kinds": [
                    str(item) for item in (value.get("surface_kinds") or [])
                ],
            }
        )
    return hotspots


_ISOTOPE_COLORS = {
    "Cs-137": "#d62728",
    "Co-60": "#1f77b4",
    "Eu-154": "#2ca02c",
}


def _environment_bounds(
    environment: Mapping[str, object],
    estimate: MLEEstimate | None,
) -> tuple[float, float, float]:
    """Return positive xyz plotting bounds from runtime-owned scene metadata."""
    x_max = _finite_float(environment.get("size_x"), fallback=0.0)
    y_max = _finite_float(environment.get("size_y"), fallback=0.0)
    z_max = _finite_float(environment.get("size_z"), fallback=0.0)
    if estimate is not None:
        points = np.asarray(
            [patch.centroid_xyz for patch in estimate.patches],
            dtype=np.float64,
        )
        x_max = max(x_max, float(np.max(points[:, 0], initial=1.0)))
        y_max = max(y_max, float(np.max(points[:, 1], initial=1.0)))
        z_max = max(z_max, float(np.max(points[:, 2], initial=1.0)))
    return max(x_max, 1.0), max(y_max, 1.0), max(z_max, 1.0)


def _draw_obstacles(
    axis: object,
    environment: Mapping[str, object],
) -> None:
    """Draw the runtime-owned obstacle grid on one top-down Matplotlib axis."""
    from matplotlib.patches import Rectangle

    raw_grid = environment.get("obstacle_grid", {})
    grid = raw_grid if isinstance(raw_grid, Mapping) else {}
    cell_size = max(_finite_float(grid.get("cell_size"), fallback=1.0), 1.0e-9)
    raw_origin = grid.get("origin", (0.0, 0.0))
    origin = np.asarray(raw_origin, dtype=np.float64).reshape(-1)
    if origin.size < 2:
        origin = np.zeros(2, dtype=np.float64)
    for raw_cell in grid.get("blocked_cells", []):
        cell = np.asarray(raw_cell, dtype=np.float64).reshape(-1)
        if cell.size < 2 or np.any(~np.isfinite(cell[:2])):
            continue
        axis.add_patch(
            Rectangle(
                (
                    float(origin[0] + cell[0] * cell_size),
                    float(origin[1] + cell[1] * cell_size),
                ),
                cell_size,
                cell_size,
                facecolor="black",
                edgecolor="none",
                alpha=0.75,
                zorder=0,
            )
        )


def _draw_obstacles_3d(
    axis: object,
    environment: Mapping[str, object],
) -> None:
    """Draw runtime obstacle cells as flat floor patches on a 3-D axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    raw_grid = environment.get("obstacle_grid", {})
    grid = raw_grid if isinstance(raw_grid, Mapping) else {}
    cell_size = max(_finite_float(grid.get("cell_size"), fallback=1.0), 1.0e-9)
    origin = np.asarray(grid.get("origin", (0.0, 0.0)), dtype=np.float64).reshape(-1)
    if origin.size < 2:
        origin = np.zeros(2, dtype=np.float64)
    patches: list[list[tuple[float, float, float]]] = []
    for raw_cell in grid.get("blocked_cells", []):
        cell = np.asarray(raw_cell, dtype=np.float64).reshape(-1)
        if cell.size < 2 or np.any(~np.isfinite(cell[:2])):
            continue
        x0 = float(origin[0] + cell[0] * cell_size)
        y0 = float(origin[1] + cell[1] * cell_size)
        patches.append(
            [
                (x0, y0, 0.0),
                (x0 + cell_size, y0, 0.0),
                (x0 + cell_size, y0 + cell_size, 0.0),
                (x0, y0 + cell_size, 0.0),
            ]
        )
    if patches:
        axis.add_collection3d(
            Poly3DCollection(
                patches,
                facecolor="black",
                edgecolor="none",
                alpha=0.25,
            )
        )


def _truth_arrays(
    cui_overlay: Mapping[str, object] | None,
    isotope: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return evaluation-only truth positions and strengths for one isotope."""
    truth = None if cui_overlay is None else cui_overlay.get("truth")
    if not isinstance(truth, Mapping):
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    sources = truth.get("true_sources", {})
    strengths = truth.get("true_strengths", {})
    raw_positions = sources.get(isotope, []) if isinstance(sources, Mapping) else []
    raw_strengths = strengths.get(isotope, []) if isinstance(strengths, Mapping) else []
    positions = np.asarray(raw_positions, dtype=np.float64)
    if positions.size == 0:
        positions = np.zeros((0, 3), dtype=np.float64)
    elif positions.ndim != 2 or positions.shape[1] != 3:
        positions = np.zeros((0, 3), dtype=np.float64)
    values = np.asarray(raw_strengths, dtype=np.float64).reshape(-1)
    if values.size != positions.shape[0]:
        values = np.zeros(positions.shape[0], dtype=np.float64)
    return positions, values


def _hotspot_arrays(
    estimate: MLEEstimate | None,
    isotope: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return MLE hotspot centroids and integrated strengths for one isotope."""
    if estimate is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    rows = [
        row for row in _hotspot_payload(estimate) if str(row.get("isotope")) == isotope
    ]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return (
        np.asarray([row["centroid_xyz"] for row in rows], dtype=np.float64),
        np.asarray(
            [row["integrated_strength_cps_1m"] for row in rows],
            dtype=np.float64,
        ),
    )


def _xyz_rows(value: object) -> NDArray[np.float64]:
    """Return finite xyz rows or an empty array for malformed display input."""
    rows = np.asarray(value, dtype=np.float64)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 3 or np.any(~np.isfinite(rows)):
        return np.zeros((0, 3), dtype=np.float64)
    return rows


def _path_segments(payload: Mapping[str, object]) -> list[NDArray[np.float64]]:
    """Return validated obstacle-aware runtime path segments."""
    segments: list[NDArray[np.float64]] = []
    raw_segments = payload.get("travel_path_segments_xyz", [])
    if not isinstance(raw_segments, list):
        return segments
    for raw_segment in raw_segments:
        segment = _xyz_rows(raw_segment)
        if segment.shape[0] >= 2:
            segments.append(segment)
    return segments


def _measurement_stations(
    payload: Mapping[str, object],
) -> tuple[NDArray[np.float64], list[str]]:
    """Return measurement positions and PF-style station visit labels."""
    points: list[NDArray[np.float64]] = []
    labels: list[str] = []
    raw_stations = payload.get("measurement_stations", [])
    if isinstance(raw_stations, list):
        for index, raw_station in enumerate(raw_stations):
            if not isinstance(raw_station, Mapping):
                continue
            point = np.asarray(raw_station.get("position_xyz", ()), dtype=np.float64)
            if point.shape != (3,) or np.any(~np.isfinite(point)):
                continue
            visits = max(int(raw_station.get("visit_count", 1)), 1)
            station_id = int(raw_station.get("station_id", index))
            labels.append(str(station_id) if visits <= 1 else f"{station_id}({visits})")
            points.append(point)
    if points:
        return np.vstack(points), labels

    positions = _xyz_rows(payload.get("detector_positions_xyz", []))
    if not positions.size:
        return positions, []
    unique_points: list[NDArray[np.float64]] = []
    counts: list[int] = []
    for point in positions:
        if unique_points and float(np.linalg.norm(point - unique_points[-1])) <= 1.0e-6:
            counts[-1] += 1
        else:
            unique_points.append(point.copy())
            counts.append(1)
    labels = [
        str(index) if count <= 1 else f"{index}({count})"
        for index, count in enumerate(counts)
    ]
    return np.vstack(unique_points), labels


def _unique_path_waypoints(
    segments: list[NDArray[np.float64]],
    stations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return distinct intermediate waypoints that are not station positions."""
    waypoints: list[NDArray[np.float64]] = []
    for segment in segments:
        for point in segment[1:-1]:
            if (
                stations.size
                and float(np.min(np.linalg.norm(stations - point, axis=1))) <= 1.0e-6
            ):
                continue
            if any(
                float(np.linalg.norm(point - prior)) <= 1.0e-6 for prior in waypoints
            ):
                continue
            waypoints.append(point.copy())
    return np.vstack(waypoints) if waypoints else np.zeros((0, 3), dtype=np.float64)


def _draw_path(
    axis: object,
    payload: Mapping[str, object],
    *,
    three_d: bool,
    show_station_labels: bool = False,
    show_legend_context: bool = True,
) -> None:
    """Draw the runtime route using the same visual contract as the PF CUI."""
    from matplotlib import patheffects

    segments = _path_segments(payload)
    stations, station_labels = _measurement_stations(payload)
    for index, segment in enumerate(segments):
        coordinates = (
            (segment[:, 0], segment[:, 1], segment[:, 2])
            if three_d
            else (segment[:, 0], segment[:, 1])
        )
        axis.plot(
            *coordinates,
            "-",
            color="cyan",
            linewidth=1.6 if three_d else 2.0,
            alpha=0.68 if three_d else 0.75,
            label=("traversed path" if index == 0 and show_legend_context else None),
            zorder=4,
        )
    waypoints = _unique_path_waypoints(segments, stations)
    if waypoints.size:
        coordinates = (
            (waypoints[:, 0], waypoints[:, 1], waypoints[:, 2])
            if three_d
            else (waypoints[:, 0], waypoints[:, 1])
        )
        axis.scatter(
            *coordinates,
            s=9 if three_d else 18,
            color="cyan",
            edgecolor="black",
            linewidth=0.25 if three_d else 0.3,
            alpha=0.28 if three_d else 0.55,
            marker=".",
            label="path waypoint" if show_legend_context else None,
            zorder=6,
        )
    if stations.size:
        coordinates = (
            (stations[:, 0], stations[:, 1], stations[:, 2])
            if three_d
            else (stations[:, 0], stations[:, 1])
        )
        axis.scatter(
            *coordinates,
            s=34 if three_d else 55,
            color="white",
            edgecolor="cyan",
            linewidth=0.8 if three_d else 1.0,
            label="measurement station" if show_legend_context else None,
            zorder=9,
        )
        if show_station_labels and not three_d:
            for point, label in zip(stations, station_labels):
                text = axis.text(
                    point[0],
                    point[1],
                    label,
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=10,
                )
                text.set_path_effects(
                    [patheffects.withStroke(linewidth=1.8, foreground="white")]
                )
    current = np.asarray(
        payload.get("current_detector_position_xyz", ()), dtype=np.float64
    )
    if current.shape != (3,) or np.any(~np.isfinite(current)):
        current = stations[-1] if stations.size else np.zeros(0, dtype=np.float64)
    if current.size:
        coordinates = (
            (current[0:1], current[1:2], current[2:3])
            if three_d
            else (current[0:1], current[1:2])
        )
        axis.scatter(
            *coordinates,
            s=70 if three_d else 130,
            color="cyan",
            edgecolor="black",
            linewidth=0.7 if three_d else 1.0,
            label="robot" if show_legend_context else None,
            zorder=12,
        )


def _apply_metric_ticks_2d(
    axis: object,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Apply PF-style two-metre major ticks to a 2-D scene axis."""
    axis.set_xticks(np.arange(xlim[0], xlim[1] + 1.0e-9, 2.0))
    axis.set_yticks(np.arange(ylim[0], ylim[1] + 1.0e-9, 2.0))


def _format_3d_axis(
    axis: object,
    *,
    bounds: tuple[float, float, float],
    title: str,
) -> None:
    """Apply PF-style world bounds, ticks, aspect, labels, and camera."""
    x_max, y_max, z_max = bounds
    axis.set(xlim=(0.0, x_max), ylim=(0.0, y_max), zlim=(0.0, z_max))
    axis.set_xticks(np.arange(0.0, x_max + 1.0e-9, 2.0))
    axis.set_yticks(np.arange(0.0, y_max + 1.0e-9, 2.0))
    axis.set_zticks(np.arange(0.0, z_max + 1.0e-9, 2.0))
    axis.set_box_aspect((x_max, y_max, z_max))
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(title, fontsize=10)
    axis.view_init(elev=26.0, azim=-58.0)


def _save_figure_atomic(figure: object, path: Path) -> None:
    """Atomically publish one Matplotlib PNG without partial browser reads."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Dashboard image staging file exists: {temporary}")
    try:
        figure.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _render_dashboard_images(
    estimate: MLEEstimate | None,
    payload: Mapping[str, object],
    environment: Mapping[str, object],
    cui_overlay: Mapping[str, object] | None,
    output_dir: Path,
) -> None:
    """Render the PF-style scientific PNG set for the browser CUI."""
    isotopes = tuple(str(value) for value in payload.get("isotopes", []))
    x_max, y_max, z_max = _environment_bounds(environment, estimate)
    progress = (
        f"records={int(payload.get('record_count', 0))} "
        f"station={payload.get('latest_station_id', '—')} "
        f"step={payload.get('latest_step_id', '—')}"
    )

    overview = plt.figure(figsize=(11.2, 8.0))
    overview_grid = overview.add_gridspec(2, 2)
    top_axis = overview.add_subplot(overview_grid[:, 0])
    elevation_axis = overview.add_subplot(overview_grid[0, 1])
    info_axis = overview.add_subplot(overview_grid[1, 1])
    info_axis.axis("off")
    _draw_obstacles(top_axis, environment)
    _draw_path(top_axis, payload, three_d=False)
    for isotope in isotopes:
        color = _ISOTOPE_COLORS.get(isotope, "#9467bd")
        truth_positions, _ = _truth_arrays(cui_overlay, isotope)
        hotspots, _ = _hotspot_arrays(estimate, isotope)
        if truth_positions.size:
            top_axis.scatter(
                truth_positions[:, 0],
                truth_positions[:, 1],
                marker="*",
                s=85,
                color=color,
                label=f"true {isotope}",
                zorder=7,
            )
            elevation_axis.scatter(
                truth_positions[:, 0],
                truth_positions[:, 2],
                marker="*",
                s=85,
                color=color,
                label=f"true {isotope}",
                zorder=7,
            )
        if hotspots.size:
            top_axis.scatter(
                hotspots[:, 0],
                hotspots[:, 1],
                marker="x",
                s=145,
                linewidths=2.2,
                color=color,
                label=f"MLE {isotope}",
                zorder=8,
            )
            elevation_axis.scatter(
                hotspots[:, 0],
                hotspots[:, 2],
                marker="x",
                s=145,
                linewidths=2.2,
                color=color,
                label=f"MLE {isotope}",
                zorder=8,
            )
    top_axis.set(xlim=(0.0, x_max), ylim=(0.0, y_max), xlabel="x [m]", ylabel="y [m]")
    elevation_axis.set(
        xlim=(0.0, x_max), ylim=(0.0, z_max), xlabel="x [m]", ylabel="z [m]"
    )
    stations, _ = _measurement_stations(payload)
    if stations.size:
        elevation_axis.scatter(
            stations[:, 0],
            stations[:, 2],
            s=28,
            color="cyan",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.55,
            label="station height",
            zorder=6,
        )
    elevation_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    elevation_axis.axhline(z_max, color="black", linewidth=0.8, alpha=0.25)
    top_axis.set_title("Top-down map: obstacles, path, truth, and MLE")
    elevation_axis.set_title("Elevation projection: height ambiguity")
    for axis, x_limits, y_limits in (
        (top_axis, (0.0, x_max), (0.0, y_max)),
        (elevation_axis, (0.0, x_max), (0.0, z_max)),
    ):
        _apply_metric_ticks_2d(axis, xlim=x_limits, ylim=y_limits)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    handles, labels = top_axis.get_legend_handles_labels()
    elevation_handles, elevation_labels = elevation_axis.get_legend_handles_labels()
    legend_by_label = dict(zip(labels + elevation_labels, handles + elevation_handles))
    if legend_by_label:
        info_axis.legend(
            legend_by_label.values(),
            legend_by_label.keys(),
            loc="upper left",
            fontsize=7,
            frameon=True,
        )
    summary = payload.get("summary")
    summary_mapping = summary if isinstance(summary, Mapping) else {}
    info_axis.text(
        0.0,
        0.02,
        "\n".join(
            (
                progress,
                f"status: {payload.get('status', 'starting')}",
                f"converged: {summary_mapping.get('converged', '—')}",
                f"iterations: {summary_mapping.get('iterations', '—')}",
                f"Poisson deviance: {summary_mapping.get('poisson_deviance', '—')}",
                f"surface patches: {summary_mapping.get('patch_count', '—')}",
                "truth markers are evaluation overlay only",
            )
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        transform=info_axis.transAxes,
    )
    overview.suptitle("RA-L experiment overview", fontsize=13, fontweight="bold")
    overview.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.08,
        wspace=0.25,
        hspace=0.32,
    )
    _save_figure_atomic(overview, output_dir / OVERVIEW_IMAGE_FILENAME)

    robot, robot_axis = plt.subplots(figsize=(8.4, 7.2))
    _draw_obstacles(robot_axis, environment)
    _draw_path(
        robot_axis,
        payload,
        three_d=False,
        show_station_labels=True,
    )
    for isotope in isotopes:
        color = _ISOTOPE_COLORS.get(isotope, "#9467bd")
        truth_positions, _ = _truth_arrays(cui_overlay, isotope)
        hotspots, _ = _hotspot_arrays(estimate, isotope)
        if truth_positions.size:
            robot_axis.scatter(
                truth_positions[:, 0],
                truth_positions[:, 1],
                marker="*",
                s=95,
                color=color,
                label=f"true {isotope}",
                zorder=7,
            )
        if hotspots.size:
            robot_axis.scatter(
                hotspots[:, 0],
                hotspots[:, 1],
                marker="x",
                s=170,
                linewidths=2.4,
                color=color,
                label=f"MLE {isotope}",
                zorder=8,
            )
    robot_axis.set(xlim=(0.0, x_max), ylim=(0.0, y_max), xlabel="x [m]", ylabel="y [m]")
    _apply_metric_ticks_2d(
        robot_axis,
        xlim=(0.0, x_max),
        ylim=(0.0, y_max),
    )
    robot_axis.set_aspect("equal", adjustable="box")
    robot_axis.grid(alpha=0.25)
    robot_axis.set_title(f"Robot 2D position — {progress}")
    handles, labels = robot_axis.get_legend_handles_labels()
    if handles:
        robot_axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    robot.tight_layout()
    _save_figure_atomic(robot, output_dir / ROBOT_IMAGE_FILENAME)

    mle_figure = plt.figure(figsize=(14.0, 6.2))
    density_axis = mle_figure.add_subplot(1, 2, 1, projection="3d")
    hotspot_axis = mle_figure.add_subplot(1, 2, 2, projection="3d")
    _draw_obstacles_3d(density_axis, environment)
    _draw_obstacles_3d(hotspot_axis, environment)
    _draw_path(density_axis, payload, three_d=True)
    _draw_path(
        hotspot_axis,
        payload,
        three_d=True,
        show_legend_context=False,
    )
    if estimate is not None:
        patch_points = np.asarray(
            [patch.centroid_xyz for patch in estimate.patches], dtype=np.float64
        )
        for isotope_index, isotope in enumerate(isotopes):
            if isotope_index >= estimate.density_by_isotope.shape[0]:
                continue
            density = np.asarray(
                estimate.density_by_isotope[isotope_index], dtype=float
            )
            peak = max(float(np.max(density, initial=0.0)), 1.0e-12)
            active = density > peak * 1.0e-4
            if np.any(active):
                density_axis.scatter(
                    patch_points[active, 0],
                    patch_points[active, 1],
                    patch_points[active, 2],
                    s=4.0 + 34.0 * np.sqrt(density[active] / peak),
                    alpha=0.38,
                    color=_ISOTOPE_COLORS.get(isotope, "#9467bd"),
                    label=isotope,
                )
            hotspots, strengths = _hotspot_arrays(estimate, isotope)
            if hotspots.size:
                sizes = 90.0 + 120.0 * strengths / max(
                    float(np.max(strengths)), 1.0e-12
                )
                hotspot_axis.scatter(
                    hotspots[:, 0],
                    hotspots[:, 1],
                    hotspots[:, 2],
                    marker="x",
                    s=sizes,
                    linewidths=2.5,
                    color=_ISOTOPE_COLORS.get(isotope, "#9467bd"),
                    label=f"MLE {isotope}",
                )
    for isotope in isotopes:
        truth_positions, _ = _truth_arrays(cui_overlay, isotope)
        if truth_positions.size:
            for axis in (density_axis, hotspot_axis):
                axis.scatter(
                    truth_positions[:, 0],
                    truth_positions[:, 1],
                    truth_positions[:, 2],
                    marker="*",
                    s=100,
                    color=_ISOTOPE_COLORS.get(isotope, "#9467bd"),
                    label=f"true {isotope}",
                )
    for axis, title in (
        (density_axis, "Surface-patch intensity"),
        (hotspot_axis, "MLE hotspot centroids"),
    ):
        _format_3d_axis(
            axis,
            bounds=(x_max, y_max, z_max),
            title=title,
        )
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    mle_figure.suptitle(f"Surface MLE 3D — {progress}")
    mle_figure.tight_layout()
    _save_figure_atomic(mle_figure, output_dir / MLE_IMAGE_FILENAME)

    spectrum, spectrum_axis = plt.subplots(figsize=(11.0, 4.8))
    observed = np.asarray(
        payload.get("latest_observed_spectrum_counts", []),
        dtype=np.float64,
    ).reshape(-1)
    energy_edges = np.asarray(
        payload.get("energy_bin_edges_keV", []),
        dtype=np.float64,
    ).reshape(-1)
    energy_axis = (
        0.5 * (energy_edges[:-1] + energy_edges[1:])
        if energy_edges.size == observed.size + 1
        else np.arange(observed.size, dtype=np.float64)
    )
    if observed.size:
        spectrum_axis.step(
            energy_axis,
            observed,
            where="mid",
            color="#202020",
            linewidth=0.8,
            alpha=0.75,
            label="observed",
        )
    predicted_available = False
    if estimate is not None and estimate.predicted_spectra is not None:
        predicted = np.asarray(estimate.predicted_spectra, dtype=np.float64)
        if predicted.ndim == 2 and predicted.shape[0]:
            prediction = predicted[-1]
            prediction_axis = (
                energy_axis
                if energy_axis.size == prediction.size
                else np.arange(prediction.size, dtype=np.float64)
            )
            spectrum_axis.plot(
                prediction_axis,
                prediction,
                color="#1f77b4",
                linewidth=1.0,
                label="MLE prediction",
            )
            predicted_available = True
    if observed.size or predicted_available:
        spectrum_axis.set_yscale("symlog", linthresh=1.0)
        spectrum_axis.set_ylabel("counts per measurement")
        spectrum_axis.set_xlabel(
            "energy [keV]" if energy_edges.size == observed.size + 1 else "spectrum bin"
        )
        spectrum_axis.legend()
    else:
        spectrum_axis.text(
            0.5,
            0.5,
            "Predicted spectrum appears after the first completed fit",
            ha="center",
            va="center",
            transform=spectrum_axis.transAxes,
        )
    spectrum_axis.grid(alpha=0.25)
    spectrum_axis.set_title(f"Latest observed and predicted spectrum — {progress}")
    spectrum.tight_layout()
    _save_figure_atomic(spectrum, output_dir / SPECTRUM_IMAGE_FILENAME)


def _dashboard_payload(
    estimate: MLEEstimate | None,
    state: Mapping[str, object],
    *,
    environment: Mapping[str, object] | None = None,
    cui_overlay: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the browser data contract with a display-only CUI overlay."""
    truth = None if cui_overlay is None else cui_overlay.get("truth")
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": str(state.get("status", "starting")),
        "run_id": state.get("run_id"),
        "mode": state.get("mode"),
        "isotopes": list(state.get("isotopes", [])),
        "record_count": int(state.get("record_count", 0)),
        "latest_step_id": state.get("latest_step_id"),
        "latest_station_id": state.get("latest_station_id"),
        "station_reports": list(state.get("station_reports", [])),
        "planning": state.get("latest_planning"),
        "summary": None,
        "patches": [],
        "density_by_isotope": {},
        "detector_positions_xyz": [],
        "travel_path_segments_xyz": list(state.get("travel_path_segments_xyz", [])),
        "measurement_stations": list(state.get("measurement_stations", [])),
        "current_detector_position_xyz": state.get("current_detector_position_xyz"),
        "hotspots": [],
        "latest_observed_spectrum_counts": list(
            state.get("latest_observed_spectrum_counts", [])
        ),
        "energy_bin_edges_keV": list(state.get("energy_bin_edges_keV", [])),
        "cui": {
            "environment": dict(environment or {}),
            "truth": truth,
        },
    }
    if estimate is None:
        return payload
    diagnostics = estimate.diagnostics
    payload["summary"] = {
        "objective": _finite_float(estimate.objective_value),
        "poisson_deviance": _finite_float(estimate.poisson_deviance),
        "iterations": int(estimate.iterations),
        "converged": bool(estimate.converged),
        "patch_count": len(estimate.patches),
        "residual_l2": _finite_float(diagnostics.get("residual_l2", 0.0)),
    }
    payload["patches"] = [
        {
            "patch_id": patch.patch_id,
            "centroid_xyz": patch.centroid_xyz.tolist(),
            "area_m2": float(patch.area_m2),
            "surface_kind": patch.surface_kind,
            "object_id": patch.object_id,
        }
        for patch in estimate.patches
    ]
    payload["density_by_isotope"] = {
        isotope: estimate.density_by_isotope[index].tolist()
        for index, isotope in enumerate(estimate.isotope_names)
    }
    positions = diagnostics.get("detector_positions_xyz", [])
    position_array = np.asarray(positions, dtype=float)
    if (
        position_array.ndim == 2
        and position_array.shape[1:] == (3,)
        and np.all(np.isfinite(position_array))
    ):
        payload["detector_positions_xyz"] = position_array.tolist()
    payload["hotspots"] = _hotspot_payload(estimate)
    return payload


class OnlineMLEDashboard:
    """Publish a self-refreshing estimator and private-CUI browser workspace."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        environment: Mapping[str, object] | None = None,
        cui_overlay: Mapping[str, object] | None = None,
    ) -> None:
        """Create static dashboard assets in an online result directory."""
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / DASHBOARD_INDEX_FILENAME
        self.data_path = self.output_dir / DASHBOARD_DATA_FILENAME
        self.environment = dict(environment or {})
        self.cui_overlay = None if cui_overlay is None else dict(cui_overlay)
        _write_bytes_atomic(self.index_path, _DASHBOARD_HTML.encode("utf-8"))

    def publish(
        self,
        estimate: MLEEstimate | None,
        state: Mapping[str, object],
    ) -> None:
        """Atomically publish the latest browser data snapshot."""
        payload = _dashboard_payload(
            estimate,
            state,
            environment=self.environment,
            cui_overlay=self.cui_overlay,
        )
        _render_dashboard_images(
            estimate,
            payload,
            self.environment,
            self.cui_overlay,
            self.output_dir,
        )
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        _write_bytes_atomic(self.data_path, encoded)


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Rotating Shield MLE CUI View</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    header { padding: 10px 16px; background: #1d1d1d; border-bottom: 1px solid #333; display: flex; justify-content: space-between; gap: 16px; }
    header span:last-child { color: #aaa; }
    main { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px; }
    section { background: #181818; border: 1px solid #333; padding: 8px; }
    h2 { margin: 0 0 8px; font-size: 16px; font-weight: 600; }
    img { width: 100%; height: calc(50vh - 70px); object-fit: contain; background: #fff; }
    .wide { grid-column: 1 / span 2; }
    .overview img { height: min(78vh, 980px); }
    @media (max-width: 860px) { main { grid-template-columns: 1fr; } .wide { grid-column: auto; } img, .overview img { height: auto; } }
  </style>
</head>
<body>
  <header><span>Rotating Shield MLE CUI View — auto refresh every 2 s — truth: evaluation overlay only</span><span id="status">loading</span></header>
  <main>
    <section class="wide overview"><h2>RA-L experiment overview</h2><img id="overview" src="latest_experiment_overview.png" alt="MLE experiment overview"></section>
    <section><h2>Robot position 2D</h2><img id="robot" src="latest_robot_2d.png" alt="Robot path and MLE estimates"></section>
    <section><h2>Surface MLE 3D</h2><img id="mle" src="latest_mle_3d.png" alt="Three-dimensional MLE surface estimate"></section>
    <section class="wide"><h2>Latest observed and predicted full spectrum</h2><img id="spectrum" src="latest_spectrum.png" alt="Latest observed and predicted spectrum"></section>
  </main>
  <script>
    async function refresh() {
      const t = Date.now();
      document.getElementById("overview").src = "latest_experiment_overview.png?t=" + t;
      document.getElementById("robot").src = "latest_robot_2d.png?t=" + t;
      document.getElementById("mle").src = "latest_mle_3d.png?t=" + t;
      document.getElementById("spectrum").src = "latest_spectrum.png?t=" + t;
      try {
        const response = await fetch("dashboard_data.json?t=" + t, {cache: "no-store"});
        const data = await response.json();
        document.getElementById("status").textContent = `${String(data.status || "starting").toUpperCase()} · records ${data.record_count ?? 0} · station ${data.latest_station_id ?? "—"}`;
      } catch (_) {
        document.getElementById("status").textContent = "RECONNECTING";
      }
    }
    refresh(); setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


__all__ = [
    "DASHBOARD_DATA_FILENAME",
    "DASHBOARD_INDEX_FILENAME",
    "DEFAULT_DASHBOARD_PORT",
    "OnlineMLEDashboard",
    "ensure_dashboard_server",
]
