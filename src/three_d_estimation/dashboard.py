"""Browser dashboard and URL serving for online surface MLE progress."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import threading
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from runtime.artifacts import atomic_write_bytes
from runtime.cui import (
    CUIDashboardConfig,
    CUIRoute,
    CUIServerHandle,
    cui_browser_url,
    start_cui_server,
)
from runtime.cui_components import (
    CUIPanelSpec,
    CUIScene,
    shared_cui_panel_specs,
    write_cui_index,
)
from runtime.defaults import (
    DEFAULT_CUI_SPLIT_VIEW_HOST,
    DEFAULT_CUI_SPLIT_VIEW_PORT,
)

from .types import MLEEstimate


DASHBOARD_DATA_FILENAME = "dashboard_data.json"
DASHBOARD_INDEX_FILENAME = "index.html"
DEFAULT_DASHBOARD_HOST = DEFAULT_CUI_SPLIT_VIEW_HOST
DEFAULT_DASHBOARD_PORT = DEFAULT_CUI_SPLIT_VIEW_PORT
OVERVIEW_IMAGE_FILENAME = "latest_experiment_overview.png"
ROBOT_IMAGE_FILENAME = "latest_robot_2d.png"
MLE_IMAGE_FILENAME = "latest_mle_3d.png"
MLE_LABELED_IMAGE_FILENAME = "latest_mle_3d_labeled.png"
SPECTRUM_IMAGE_FILENAME = "latest_spectrum.png"
MLE_RESULT_PANEL_SPECS = (
    CUIPanelSpec(
        "mle-surface-map",
        "Surface-MLE patch grid and hotspots",
        MLE_IMAGE_FILENAME,
    ),
    CUIPanelSpec(
        "mle-surface-map-labeled",
        "Surface-MLE patch grid with hotspot labels",
        MLE_LABELED_IMAGE_FILENAME,
        2,
    ),
)

_CUI_SERVER_HANDLES: dict[tuple[Path, str, int, str | None], CUIServerHandle] = {}
_CUI_SERVER_LOCK = threading.Lock()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Durably replace one dashboard artifact."""
    atomic_write_bytes(path, payload)


def ensure_dashboard_server(
    output_dir: str | Path,
    *,
    host: str = DEFAULT_DASHBOARD_HOST,
    port: int = DEFAULT_DASHBOARD_PORT,
    public_host: str | None = None,
) -> str:
    """Start or reuse the shared runtime CUI server and return its browser URL."""
    root = Path(output_dir).resolve()
    config = CUIDashboardConfig(
        host=host,
        port=port,
        public_host=public_host,
    )
    key = (root, config.host, config.port, config.public_host)
    with _CUI_SERVER_LOCK:
        handle = _CUI_SERVER_HANDLES.get(key)
        if handle is None:
            handle = start_cui_server(
                root,
                index_path=DASHBOARD_INDEX_FILENAME,
                config=config,
            )
            handle.persistent = True
            _CUI_SERVER_HANDLES[key] = handle
    if handle.url is None:
        raise RuntimeError("Shared runtime did not start the MLE CUI server.")
    return handle.url


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
    scene: CUIScene,
    estimate: MLEEstimate | None,
) -> tuple[float, float, float]:
    """Return positive xyz plotting bounds from the canonical runtime scene."""
    x_max, y_max, z_max = map(float, scene.bounds_max_xyz)
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
    scene: CUIScene,
) -> None:
    """Draw canonical runtime obstacle footprints on a top-down axis."""
    from matplotlib.patches import Polygon

    for footprint in scene.obstacle_footprints_xy:
        axis.add_patch(
            Polygon(
                footprint,
                closed=True,
                facecolor="black",
                edgecolor="none",
                alpha=0.75,
                zorder=0,
            )
        )


def _draw_obstacles_3d(
    axis: object,
    scene: CUIScene,
) -> None:
    """Draw canonical runtime obstacles as flat floor patches on a 3-D axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    patches: list[list[tuple[float, float, float]]] = []
    for footprint in scene.obstacle_footprints_xy:
        patches.append([(float(x), float(y), 0.0) for x, y in footprint])
    if patches:
        axis.add_collection3d(
            Poly3DCollection(
                patches,
                facecolor="black",
                edgecolor="none",
                alpha=0.25,
            )
        )


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


def _measurement_stations(
    route: CUIRoute,
) -> tuple[NDArray[np.float64], list[str]]:
    """Return canonical runtime station positions and PF-style visit labels."""
    labels = [
        str(station_id) if visits <= 1 else f"{station_id}({visits})"
        for station_id, visits in zip(
            route.measurement_station_ids,
            route.measurement_visit_counts,
            strict=True,
        )
    ]
    return route.measurement_stations_xyz, labels


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
    route: CUIRoute,
    *,
    three_d: bool,
    show_station_labels: bool = False,
    show_legend_context: bool = True,
) -> None:
    """Draw the runtime route using the same visual contract as the PF CUI."""
    from matplotlib import patheffects

    segments = list(route.travel_path_segments_xyz)
    stations, station_labels = _measurement_stations(route)
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
    current = route.current_detector_position_xyz
    if current is None:
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
    try:
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        atomic_write_bytes(path, buffer.getvalue())
    finally:
        plt.close(figure)


def _annotate_3d_sources(
    axis: object,
    positions: NDArray[np.float64],
    labels: list[str],
    *,
    color: str,
) -> None:
    """Draw compact source identifiers beside 3-D points."""
    for point, label in zip(positions, labels, strict=True):
        axis.text(
            float(point[0]),
            float(point[1]),
            float(point[2]),
            label,
            color=color,
            fontsize=8,
            fontweight="bold",
        )


def _build_mle_3d_figure(
    estimate: MLEEstimate | None,
    *,
    isotopes: tuple[str, ...],
    route: CUIRoute,
    scene: CUIScene,
    bounds: tuple[float, float, float],
    progress: str,
    labeled: bool,
) -> object:
    """Build one PF-aligned Surface MLE 3-D figure."""
    figure = plt.figure(figsize=(14.2, 6.6))
    density_axis = figure.add_subplot(1, 2, 1, projection="3d")
    hotspot_axis = figure.add_subplot(1, 2, 2, projection="3d")
    _draw_obstacles_3d(density_axis, scene)
    _draw_obstacles_3d(hotspot_axis, scene)
    _draw_path(density_axis, route, three_d=True)
    _draw_path(
        hotspot_axis,
        route,
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
            color = _ISOTOPE_COLORS.get(isotope, "#9467bd")
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
                    color=color,
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
                    color=color,
                    label=f"MLE {isotope}",
                )
                if labeled:
                    _annotate_3d_sources(
                        hotspot_axis,
                        hotspots,
                        [
                            f"{isotope} E{index + 1}"
                            for index in range(hotspots.shape[0])
                        ],
                        color=color,
                    )
    for axis, title in (
        (density_axis, "Surface-patch intensity"),
        (hotspot_axis, "MLE hotspot centroids"),
    ):
        _format_3d_axis(axis, bounds=bounds, title=title)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    label_suffix = " with source labels" if labeled else ""
    figure.suptitle(f"Surface MLE 3D{label_suffix} - {progress}")
    figure.tight_layout()
    return figure


def _render_dashboard_images(
    estimate: MLEEstimate | None,
    payload: Mapping[str, object],
    route: CUIRoute,
    scene: CUIScene,
    output_dir: Path,
) -> None:
    """Render the PF-style scientific PNG set for the browser CUI."""
    isotopes = tuple(str(value) for value in payload.get("isotopes", []))
    x_max, y_max, z_max = _environment_bounds(scene, estimate)
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
    _draw_obstacles(top_axis, scene)
    _draw_path(top_axis, route, three_d=False)
    for isotope in isotopes:
        color = _ISOTOPE_COLORS.get(isotope, "#9467bd")
        hotspots, _ = _hotspot_arrays(estimate, isotope)
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
    stations, _ = _measurement_stations(route)
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
    top_axis.set_title("Top-down map: obstacles, path, and MLE")
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
    summary_lines = [
        progress,
        f"status: {payload.get('status', 'starting')}",
        f"converged: {summary_mapping.get('converged', '—')}",
        f"iterations: {summary_mapping.get('iterations', '—')}",
        f"Poisson deviance: {summary_mapping.get('poisson_deviance', '—')}",
        f"surface patches: {summary_mapping.get('patch_count', '—')}",
    ]
    info_axis.text(
        0.0,
        0.02,
        "\n".join(summary_lines),
        ha="left",
        va="bottom",
        fontsize=8,
        transform=info_axis.transAxes,
    )
    overview.suptitle("Live experiment overview", fontsize=13, fontweight="bold")
    overview.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.08,
        wspace=0.25,
        hspace=0.32,
    )
    _save_figure_atomic(overview, output_dir / OVERVIEW_IMAGE_FILENAME)

    robot, robot_axis = plt.subplots(figsize=(7.0, 6.0))
    _draw_obstacles(robot_axis, scene)
    _draw_path(
        robot_axis,
        route,
        three_d=False,
        show_station_labels=True,
    )
    for isotope in isotopes:
        color = _ISOTOPE_COLORS.get(isotope, "#9467bd")
        hotspots, _ = _hotspot_arrays(estimate, isotope)
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
    robot_axis.set_title(f"Robot 2D position - {progress}")
    handles, labels = robot_axis.get_legend_handles_labels()
    if handles:
        robot_axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    robot.tight_layout()
    _save_figure_atomic(robot, output_dir / ROBOT_IMAGE_FILENAME)

    mle_figure = _build_mle_3d_figure(
        estimate,
        isotopes=isotopes,
        route=route,
        scene=scene,
        bounds=(x_max, y_max, z_max),
        progress=progress,
        labeled=False,
    )
    _save_figure_atomic(mle_figure, output_dir / MLE_IMAGE_FILENAME)
    mle_labeled_figure = _build_mle_3d_figure(
        estimate,
        isotopes=isotopes,
        route=route,
        scene=scene,
        bounds=(x_max, y_max, z_max),
        progress=progress,
        labeled=True,
    )
    _save_figure_atomic(
        mle_labeled_figure,
        output_dir / MLE_LABELED_IMAGE_FILENAME,
    )

    spectrum, spectrum_axis = plt.subplots(figsize=(10.0, 4.8))
    observed = (
        np.zeros(0, dtype=np.float64)
        if route.latest_spectrum_counts is None
        else route.latest_spectrum_counts.astype(np.float64)
    )
    energy_edges = (
        np.zeros(0, dtype=np.float64)
        if route.energy_bin_edges_keV is None
        else route.energy_bin_edges_keV
    )
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
    spectrum_axis.set_title(f"Full spectrum - {progress}")
    spectrum.tight_layout()
    _save_figure_atomic(spectrum, output_dir / SPECTRUM_IMAGE_FILENAME)


def _dashboard_payload(
    estimate: MLEEstimate | None,
    state: Mapping[str, object],
    *,
    route: CUIRoute | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the truth-free browser data contract."""
    resolved_route = CUIRoute() if route is None else route
    if not isinstance(resolved_route, CUIRoute):
        raise TypeError("route must be a runtime CUIRoute.")
    route_payload = resolved_route.to_payload()
    payload: dict[str, object] = {
        **route_payload,
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
        "hotspots": [],
        "latest_observed_spectrum_counts": (
            []
            if resolved_route.latest_spectrum_counts is None
            else resolved_route.latest_spectrum_counts.tolist()
        ),
        "cui": {
            "environment": dict(environment or {}),
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
    """Publish a self-refreshing truth-free estimator browser workspace."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        scene: CUIScene,
        environment: Mapping[str, object] | None = None,
    ) -> None:
        """Create static dashboard assets in an online result directory."""
        if not isinstance(scene, CUIScene):
            raise TypeError("scene must be a runtime CUIScene.")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / DASHBOARD_INDEX_FILENAME
        self.data_path = self.output_dir / DASHBOARD_DATA_FILENAME
        self.environment = dict(environment or {})
        self.scene = scene
        self._write_index()

    def _write_index(self) -> None:
        """Publish MLE-owned result panels in the shared runtime shell."""
        self.index_path = write_cui_index(
            self.output_dir,
            shared_cui_panel_specs(MLE_RESULT_PANEL_SPECS),
            title="Rotating Shield MLE CUI View",
            refresh_interval_ms=2000,
            index_filename=DASHBOARD_INDEX_FILENAME,
        )

    def publish(
        self,
        estimate: MLEEstimate | None,
        state: Mapping[str, object],
        *,
        route: CUIRoute | None = None,
    ) -> None:
        """Atomically publish the latest browser data snapshot."""
        payload = _dashboard_payload(
            estimate,
            state,
            route=route,
            environment=self.environment,
        )
        _render_dashboard_images(
            estimate,
            payload,
            CUIRoute() if route is None else route,
            self.scene,
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


__all__ = [
    "DASHBOARD_DATA_FILENAME",
    "DASHBOARD_INDEX_FILENAME",
    "DEFAULT_DASHBOARD_HOST",
    "DEFAULT_DASHBOARD_PORT",
    "MLE_IMAGE_FILENAME",
    "MLE_LABELED_IMAGE_FILENAME",
    "MLE_RESULT_PANEL_SPECS",
    "OnlineMLEDashboard",
    "cui_browser_url",
    "ensure_dashboard_server",
]
