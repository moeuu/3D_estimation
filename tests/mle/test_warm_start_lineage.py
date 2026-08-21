"""Regression tests for refined surface-MLE warm starts."""

from __future__ import annotations

import numpy as np

from measurement.model import EnvironmentConfig
from three_d_estimation.estimator import _initial_density_for_patches
from three_d_estimation.surface_patches import (
    build_surface_patches,
    refine_surface_patches,
)
from three_d_estimation.types import MLEEstimate


def test_rebuilt_base_warm_start_aggregates_all_active_descendant_levels() -> None:
    """Grandchild strength must reach its base patch without an active parent."""
    base = build_surface_patches(
        EnvironmentConfig(size_x=2.0, size_y=3.0, size_z=4.0),
        None,
        spacing=10.0,
        quadrature_points_per_patch=1,
    )
    floor = next(patch for patch in base.patches if patch.surface_kind == "floor")
    ceiling = next(
        patch for patch in base.patches if patch.surface_kind == "ceiling"
    )
    first_level = refine_surface_patches(
        base,
        [floor.patch_id, ceiling.patch_id],
    )
    floor_children = tuple(
        patch
        for patch in first_level.patches
        if patch.parent_patch_id == floor.patch_id
    )
    active = refine_surface_patches(
        first_level,
        [patch.patch_id for patch in floor_children],
    )

    isotope_names = ("Cs-137", "Co-60")
    densities = np.zeros((len(isotope_names), active.patch_count), dtype=float)
    floor_descendant_indices = [
        index
        for index, patch in enumerate(active.patches)
        if patch.surface_kind == "floor" and patch.refinement_level == 2
    ]
    ceiling_child_indices = [
        index
        for index, patch in enumerate(active.patches)
        if patch.parent_patch_id == ceiling.patch_id
    ]
    unchanged_index = next(
        index
        for index, patch in enumerate(active.patches)
        if patch.refinement_level == 0 and patch.surface_kind == "wall"
    )
    densities[0, floor_descendant_indices] = np.arange(1.0, 17.0)
    densities[1, floor_descendant_indices] = np.arange(101.0, 117.0)
    densities[0, ceiling_child_indices] = [20.0, 24.0, 28.0, 32.0]
    densities[1, ceiling_child_indices] = [3.0, 5.0, 7.0, 9.0]
    densities[:, unchanged_index] = [7.5, 11.5]
    areas = active.areas_m2
    prior = MLEEstimate(
        isotope_names=isotope_names,
        patches=active.patches,
        density_by_isotope=densities,
        patch_strength_by_isotope=densities * areas[None, :],
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=0.0,
        poisson_deviance=0.0,
        iterations=0,
        converged=True,
        diagnostics={},
    )

    mapped = _initial_density_for_patches(prior, base, isotope_names)

    assert mapped is not None
    floor_base_index = next(
        index
        for index, patch in enumerate(base.patches)
        if patch.patch_id == floor.patch_id
    )
    ceiling_base_index = next(
        index
        for index, patch in enumerate(base.patches)
        if patch.patch_id == ceiling.patch_id
    )
    unchanged_patch_id = active.patches[unchanged_index].patch_id
    unchanged_base_index = next(
        index
        for index, patch in enumerate(base.patches)
        if patch.patch_id == unchanged_patch_id
    )
    np.testing.assert_allclose(mapped[floor_base_index], [8.5, 108.5])
    np.testing.assert_allclose(mapped[ceiling_base_index], [26.0, 6.0])
    np.testing.assert_allclose(mapped[unchanged_base_index], [7.5, 11.5])
    np.testing.assert_allclose(
        mapped[floor_base_index] * floor.area_m2,
        np.sum(
            densities[:, floor_descendant_indices]
            * areas[np.asarray(floor_descendant_indices)][None, :],
            axis=1,
        ),
    )


def test_coarse_online_density_seeds_independently_built_fine_dictionary() -> None:
    """A cold final grid inherits density from containing online patches."""
    environment = EnvironmentConfig(size_x=2.0, size_y=3.0, size_z=4.0)
    coarse = build_surface_patches(
        environment,
        None,
        spacing=10.0,
        quadrature_points_per_patch=1,
    )
    fine = build_surface_patches(
        environment,
        None,
        spacing=1.0,
        quadrature_points_per_patch=1,
    )
    isotope_names = ("Cs-137", "Co-60")
    coarse_density = np.zeros((len(isotope_names), coarse.patch_count), dtype=float)
    coarse_floor_indices = [
        index
        for index, patch in enumerate(coarse.patches)
        if patch.surface_kind == "floor"
    ]
    coarse_density[:, coarse_floor_indices] = np.asarray([[4.0], [7.0]])
    prior = MLEEstimate(
        isotope_names=isotope_names,
        patches=coarse.patches,
        density_by_isotope=coarse_density,
        patch_strength_by_isotope=coarse_density * coarse.areas_m2[None, :],
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=0.0,
        poisson_deviance=0.0,
        iterations=0,
        converged=True,
        diagnostics={},
    )

    mapped = _initial_density_for_patches(prior, fine, isotope_names)

    assert mapped is not None
    fine_floor_indices = [
        index
        for index, patch in enumerate(fine.patches)
        if patch.surface_kind == "floor"
    ]
    np.testing.assert_allclose(
        mapped[np.asarray(fine_floor_indices)],
        np.broadcast_to([4.0, 7.0], (len(fine_floor_indices), 2)),
    )
    np.testing.assert_allclose(
        np.sum(
            mapped[np.asarray(fine_floor_indices)]
            * fine.areas_m2[np.asarray(fine_floor_indices), None],
            axis=0,
        ),
        np.sum(
            coarse_density[:, np.asarray(coarse_floor_indices)]
            * coarse.areas_m2[np.asarray(coarse_floor_indices)][None, :],
            axis=1,
        ),
    )
