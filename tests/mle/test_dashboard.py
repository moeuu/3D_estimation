"""Tests for browser-reachable CUI dashboard URL publication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from three_d_estimation import dashboard
from three_d_estimation.cli import _print_cui_dashboard_url


def test_dashboard_url_has_explicit_index_and_ipv6_brackets() -> None:
    """CUI URLs must be directly clickable for IPv4, names, and IPv6 hosts."""
    assert (
        dashboard._dashboard_browser_url("100.127.159.83", 8878)
        == "http://100.127.159.83:8878/index.html"
    )
    assert (
        dashboard._dashboard_browser_url("fd7a:115c:a1e0::1", 8878)
        == "http://[fd7a:115c:a1e0::1]:8878/index.html"
    )
    with pytest.raises(ValueError, match="host name or IP"):
        dashboard._dashboard_browser_url("https://example.test", 8878)


def test_dashboard_selects_next_port_instead_of_reusing_stale_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An occupied prior-run port must not silently serve stale dashboard files."""
    monkeypatch.setattr(
        dashboard,
        "_tcp_port_is_open",
        lambda _host, port: int(port) == 8878,
    )

    assert dashboard._available_dashboard_port("0.0.0.0", 8878) == 8879


def test_cui_url_is_visible_without_corrupting_json_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Human output uses stdout while JSON mode relays the immediate URL on stderr."""
    url = "http://100.127.159.83:8878/index.html"
    _print_cui_dashboard_url(url, json_output=False)
    human = capsys.readouterr()
    assert human.out == f"CUI dashboard URL: {url}\n"
    assert human.err == ""

    _print_cui_dashboard_url(url, json_output=True)
    structured = capsys.readouterr()
    assert structured.out == ""
    assert structured.err == f"CUI dashboard URL: {url}\n"


def test_dashboard_publishes_pf_style_scientific_images(tmp_path: Path) -> None:
    """The browser work surface must be the same PNG-first form as the PF CUI."""
    publisher = dashboard.OnlineMLEDashboard(
        tmp_path,
        environment={"size_x": 10.0, "size_y": 20.0, "size_z": 10.0},
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
        dashboard.SPECTRUM_IMAGE_FILENAME,
    )
    html = (tmp_path / dashboard.DASHBOARD_INDEX_FILENAME).read_text(encoding="utf-8")
    for filename in expected:
        payload = (tmp_path / filename).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert filename in html


def test_dashboard_draws_runtime_waypoints_stations_and_robot() -> None:
    """Scene rendering must not replace an obstacle-aware route with a chord."""
    payload = {
        "travel_path_segments_xyz": [
            [
                [0.5, 0.5, 1.0],
                [0.5, 1.5, 0.25],
                [1.5, 1.5, 0.25],
                [1.5, 2.5, 1.0],
            ]
        ],
        "measurement_stations": [
            {
                "station_id": 0,
                "position_xyz": [0.5, 0.5, 1.0],
                "visit_count": 1,
            },
            {
                "station_id": 1,
                "position_xyz": [1.5, 2.5, 1.0],
                "visit_count": 8,
            },
        ],
        "current_detector_position_xyz": [1.5, 2.5, 1.0],
        "detector_positions_xyz": [
            [0.5, 0.5, 1.0],
            [1.5, 2.5, 1.0],
        ],
    }
    figure, axis = dashboard.plt.subplots()
    try:
        dashboard._draw_path(
            axis,
            payload,
            three_d=False,
            show_station_labels=True,
        )

        assert len(axis.lines) == 1
        assert np.array_equal(axis.lines[0].get_xdata(), [0.5, 0.5, 1.5, 1.5])
        assert np.array_equal(axis.lines[0].get_ydata(), [0.5, 1.5, 1.5, 2.5])
        assert [text.get_text() for text in axis.texts] == ["0", "1(8)"]
    finally:
        dashboard.plt.close(figure)


def test_dashboard_draws_runtime_obstacle_grid_in_2d_and_3d() -> None:
    """Blocked runtime cells must be visible in every spatial scene."""
    environment = {
        "obstacle_grid": {
            "cell_size": 1.0,
            "origin": [0.0, 0.0],
            "blocked_cells": [[2, 3], [3, 3]],
        }
    }
    figure = dashboard.plt.figure()
    axis_2d = figure.add_subplot(1, 2, 1)
    axis_3d = figure.add_subplot(1, 2, 2, projection="3d")
    try:
        dashboard._draw_obstacles(axis_2d, environment)
        dashboard._draw_obstacles_3d(axis_3d, environment)

        assert len(axis_2d.patches) == 2
        assert len(axis_3d.collections) == 1
    finally:
        dashboard.plt.close(figure)
