"""Tests for browser-reachable CUI dashboard URL publication."""

from __future__ import annotations

from pathlib import Path
import socket
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import pytest
from runtime.cui import CUIRoute, CUIServerHandle
from runtime.cui_components import CUIScene, pf_reference_panel_specs, write_cui_index

from three_d_estimation import dashboard
from three_d_estimation import cli
from three_d_estimation.cli import _print_cui_dashboard_url, build_argument_parser
from three_d_estimation.closed_loop import MLEStopConfig


def _scene(
    *,
    bounds: tuple[float, float, float] = (2.0, 2.0, 1.5),
    obstacle_boxes: object = (),
) -> CUIScene:
    """Return one explicit runtime CUI scene for dashboard unit tests."""
    return CUIScene(
        bounds_min_xyz=np.zeros(3, dtype=np.float64),
        bounds_max_xyz=np.asarray(bounds, dtype=np.float64),
        obstacle_boxes_xyz=np.asarray(obstacle_boxes, dtype=np.float64).reshape(-1, 6),
    )


def _unused_local_port() -> int:
    """Return one recently released IPv4 loopback port for a managed-server test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _managed_dashboard_handle(port: int) -> CUIServerHandle:
    """Return the shared runtime handle bound to one test port."""
    with dashboard._CUI_SERVER_LOCK:
        for handle in dashboard._CUI_SERVER_HANDLES.values():
            if handle.port == int(port):
                return handle
    raise KeyError(f"No managed dashboard uses port {port}.")


def _close_managed_dashboard(port: int) -> None:
    """Stop and forget one process-managed dashboard created by a test."""
    with dashboard._CUI_SERVER_LOCK:
        key, handle = next(
            (key, handle)
            for key, handle in dashboard._CUI_SERVER_HANDLES.items()
            if handle.port == int(port)
        )
        dashboard._CUI_SERVER_HANDLES.pop(key)
    handle.close()


def _fetch_text(url: str) -> str:
    """Return one browser-served dashboard artifact as UTF-8 text."""
    with urlopen(url, timeout=2.0) as response:
        return response.read().decode("utf-8")


def test_dashboard_url_has_explicit_index_and_ipv6_brackets() -> None:
    """CUI URLs must be directly clickable for IPv4, names, and IPv6 hosts."""
    assert (
        dashboard.cui_browser_url("100.127.159.83", 8877)
        == "http://100.127.159.83:8877/index.html"
    )
    assert (
        dashboard.cui_browser_url("fd7a:115c:a1e0::1", 8877)
        == "http://[fd7a:115c:a1e0::1]:8877/index.html"
    )
    with pytest.raises(ValueError, match="without URL syntax"):
        dashboard.cui_browser_url("https://example.test", 8877)


def test_dashboard_binds_next_port_when_requested_port_is_occupied(
    tmp_path: Path,
) -> None:
    """An unknown listener must be skipped using the managed bind result."""
    root = tmp_path / "dashboard"
    root.mkdir()
    (root / "index.html").write_text("current-dashboard", encoding="utf-8")
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    requested_port = int(occupied.getsockname()[1])
    url: str | None = None
    try:
        url = dashboard.ensure_dashboard_server(
            root,
            host="127.0.0.1",
            port=requested_port,
            public_host="127.0.0.1",
        )
        selected_port = urlsplit(url).port
        assert selected_port is not None
        assert selected_port > requested_port
        assert _fetch_text(url) == "current-dashboard"
    finally:
        occupied.close()
        if url is not None:
            selected_port = urlsplit(url).port
            assert selected_port is not None
            _close_managed_dashboard(selected_port)


def test_dashboard_reuses_shared_runtime_handle_for_same_root(
    tmp_path: Path,
) -> None:
    """Repeated URL requests must reuse one managed runtime CUI handle."""
    root = tmp_path / "dashboard"
    root.mkdir()
    (root / "index.html").write_text("shared-dashboard", encoding="utf-8")
    port = _unused_local_port()

    first_url = dashboard.ensure_dashboard_server(
        root,
        host="127.0.0.1",
        port=port,
    )
    second_url = dashboard.ensure_dashboard_server(
        root,
        host="127.0.0.1",
        port=port,
    )

    selected_port = urlsplit(first_url).port
    assert selected_port is not None
    try:
        assert second_url == first_url
        assert _fetch_text(first_url) == "shared-dashboard"
        with dashboard._CUI_SERVER_LOCK:
            assert (
                sum(
                    handle.port == selected_port
                    for handle in dashboard._CUI_SERVER_HANDLES.values()
                )
                == 1
            )
    finally:
        _close_managed_dashboard(selected_port)


def test_dashboard_http_server_rejects_symlinks_hidden_files_and_listings(
    tmp_path: Path,
) -> None:
    """The public dashboard must expose only named regular root artifacts."""
    root = tmp_path / "dashboard"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "index.html").write_text("dashboard-index", encoding="utf-8")
    (root / ".private").write_text("hidden", encoding="utf-8")
    listing = root / "frames"
    listing.mkdir()
    (listing / "frame.txt").write_text("frame", encoding="utf-8")
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    port = _unused_local_port()
    url = dashboard.ensure_dashboard_server(
        root,
        host="127.0.0.1",
        port=port,
        public_host="127.0.0.1",
    )
    base_url = url.rsplit("/", 1)[0]
    try:
        assert _fetch_text(url) == "dashboard-index"
        for path in ("/linked/secret.txt", "/.private", "/frames/"):
            with pytest.raises(HTTPError) as error:
                _fetch_text(base_url + path)
            assert error.value.code == 404
    finally:
        selected_port = urlsplit(url).port
        assert selected_port is not None
        _close_managed_dashboard(selected_port)


def test_dashboard_server_stays_in_process_without_child_or_pid_file(
    tmp_path: Path,
) -> None:
    """Managed serving must leave no detached child after the process exits."""
    root = tmp_path / "dashboard"
    root.mkdir()
    (root / "index.html").write_text("managed", encoding="utf-8")

    port = _unused_local_port()
    url = dashboard.ensure_dashboard_server(
        root,
        host="127.0.0.1",
        port=port,
        public_host="127.0.0.1",
    )
    selected_port = urlsplit(url).port
    assert selected_port is not None
    handle = _managed_dashboard_handle(selected_port)
    try:
        assert _fetch_text(url) == "managed"
        assert handle.managed
        assert handle.persistent
        assert handle.process_id is None
        assert not tuple(root.glob("*server*.pid"))
        assert not tuple(root.glob("*server*.log"))
    finally:
        _close_managed_dashboard(selected_port)


def test_cui_url_is_visible_without_corrupting_json_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Human output uses stdout while JSON mode relays the immediate URL on stderr."""
    url = "http://100.127.159.83:8877/index.html"
    _print_cui_dashboard_url(url, json_output=False)
    human = capsys.readouterr()
    assert human.out == f"CUI split visualization URL: {url}\n"
    assert human.err == ""

    _print_cui_dashboard_url(url, json_output=True)
    structured = capsys.readouterr()
    assert structured.out == ""
    assert structured.err == f"CUI split visualization URL: {url}\n"


def test_ral_command_announces_cui_url_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The live RA-L command must not repeat its immediate CUI URL at exit."""
    url = "http://127.0.0.1:8877/index.html"
    monkeypatch.setattr(
        cli,
        "preflight_ral_full_simulation",
        lambda **kwargs: SimpleNamespace(
            ready=True,
            errors=(),
            runtime_root=tmp_path,
        ),
    )
    monkeypatch.setattr(
        cli.MLEStopConfig,
        "load",
        lambda path: MLEStopConfig(),
    )

    def fake_closed_loop(*args: object, **kwargs: object) -> SimpleNamespace:
        """Announce the server URL and return one completed run shell."""
        del args
        kwargs["dashboard_url_hook"](url)
        return SimpleNamespace(
            to_dict=lambda: {"status": "complete"},
            run_id="test-run",
            record_count=1,
            measurement_log_path=tmp_path / "log",
            mle_output_dir=tmp_path / "mle",
            stop_reason="test",
            dashboard_url=url,
        )

    monkeypatch.setattr(cli, "run_ral_closed_loop", fake_closed_loop)
    args = build_argument_parser().parse_args(
        [
            "ral-full-simulation",
            "--scenario",
            str(tmp_path / "scenario.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert cli._run_ral_full_simulation(args) == 0
    captured = capsys.readouterr()
    assert captured.out.count(f"CUI split visualization URL: {url}\n") == 1
    assert captured.err == ""


def test_dashboard_publishes_pf_style_scientific_images(tmp_path: Path) -> None:
    """The browser work surface must be the same PNG-first form as the PF CUI."""
    publisher = dashboard.OnlineMLEDashboard(
        tmp_path,
        scene=_scene(bounds=(10.0, 20.0, 10.0)),
    )
    publisher.publish(
        None,
        {
            "status": "starting",
            "run_id": "test-run",
            "mode": "spectral",
            "isotopes": ["Co-60", "Cs-137", "Eu-154"],
            "record_count": 0,
            "latest_step_id": None,
            "latest_station_id": None,
        },
    )

    expected = (
        dashboard.OVERVIEW_IMAGE_FILENAME,
        dashboard.ROBOT_IMAGE_FILENAME,
        dashboard.MLE_IMAGE_FILENAME,
        dashboard.MLE_LABELED_IMAGE_FILENAME,
        dashboard.SPECTRUM_IMAGE_FILENAME,
    )
    html = (tmp_path / dashboard.DASHBOARD_INDEX_FILENAME).read_text(encoding="utf-8")
    for filename in expected:
        payload = (tmp_path / filename).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert filename in html
    assert [html.index(filename) for filename in expected] == sorted(
        html.index(filename) for filename in expected
    )
    assert "truth" not in html.lower()
    reference_index = write_cui_index(
        tmp_path / "reference-shell",
        pf_reference_panel_specs(
            estimator_title="Surface MLE 3D",
            estimator_filename=dashboard.MLE_IMAGE_FILENAME,
            labeled_estimator_title="Surface MLE 3D with source labels",
            labeled_estimator_filename=dashboard.MLE_LABELED_IMAGE_FILENAME,
        ),
        title="Rotating Shield MLE CUI View",
        refresh_interval_ms=2000,
    )
    assert html == reference_index.read_text(encoding="utf-8")


def test_dashboard_exposes_no_truth_overlay_channel(
    tmp_path: Path,
) -> None:
    """The estimator-owned dashboard must have no truth input or output path."""
    publisher = dashboard.OnlineMLEDashboard(tmp_path, scene=_scene())
    publisher.publish(
        None,
        {
            "status": "starting",
            "run_id": "test-run",
            "mode": "spectral",
            "isotopes": ["Cs-137"],
            "record_count": 0,
        },
    )

    html = publisher.index_path.read_text(encoding="utf-8")
    payload = publisher.data_path.read_text(encoding="utf-8")
    assert not hasattr(publisher, "set_cui_overlay")
    assert "truth" not in html.lower()
    assert "truth" not in payload.lower()


def test_dashboard_draws_runtime_waypoints_stations_and_robot() -> None:
    """Scene rendering must not replace an obstacle-aware route with a chord."""
    route = CUIRoute(
        travel_path_segments_xyz=(
            np.asarray(
                [
                    [0.5, 0.5, 1.0],
                    [0.5, 1.5, 0.25],
                    [1.5, 1.5, 0.25],
                    [1.5, 2.5, 1.0],
                ]
            ),
        ),
        measurement_stations_xyz=np.asarray(
            [
                [0.5, 0.5, 1.0],
                [1.5, 2.5, 1.0],
                [1.5, 2.5, 1.0],
            ]
        ),
        measurement_station_ids=np.asarray([0, 1, 2], dtype=np.int64),
        measurement_step_ids=np.asarray([0, 1, 9], dtype=np.int64),
        measurement_visit_counts=np.asarray([1, 8, 1], dtype=np.int64),
        current_detector_position_xyz=np.asarray([1.5, 2.5, 1.0]),
        latest_step_id=9,
    )
    figure, axis = dashboard.plt.subplots()
    try:
        dashboard._draw_path(
            axis,
            route,
            three_d=False,
            show_station_labels=True,
        )

        assert len(axis.lines) == 1
        assert np.array_equal(axis.lines[0].get_xdata(), [0.5, 0.5, 1.5, 1.5])
        assert np.array_equal(axis.lines[0].get_ydata(), [0.5, 1.5, 1.5, 2.5])
        assert [text.get_text() for text in axis.texts] == ["0", "1(8)", "2"]
    finally:
        dashboard.plt.close(figure)


def test_dashboard_draws_runtime_obstacle_grid_in_2d_and_3d() -> None:
    """Canonical asymmetric obstacle XY must remain untransposed in both views."""
    scene = _scene(
        bounds=(10.0, 20.0, 3.0),
        obstacle_boxes=((3.25, 3.0, 0.0, 3.75, 3.5, 2.0),),
    )
    expected_xy = np.asarray([[3.25, 3.0], [3.75, 3.0], [3.75, 3.5], [3.25, 3.5]])
    figure = dashboard.plt.figure()
    axis_2d = figure.add_subplot(1, 2, 1)
    axis_3d = figure.add_subplot(1, 2, 2, projection="3d")
    try:
        dashboard._draw_obstacles(axis_2d, scene)
        dashboard._draw_obstacles_3d(axis_3d, scene)

        np.testing.assert_array_equal(
            scene.obstacle_footprints_xy[0],
            expected_xy,
        )
        np.testing.assert_array_equal(
            axis_2d.patches[0].get_xy()[:4],
            expected_xy,
        )
        assert len(axis_2d.patches) == 1
        assert len(axis_3d.collections) == 1
    finally:
        dashboard.plt.close(figure)
