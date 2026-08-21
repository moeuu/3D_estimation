"""Laplace and grouped-bootstrap uncertainty for standalone surface MLE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .response_operator import ResponseOperator, weighted_response_gram
from .types import MLEEstimate, ObservationBatch, SurfacePatch


@dataclass(frozen=True, slots=True)
class LaplaceSupportResult:
    """Store a bounded active-support conditional covariance approximation."""

    active_source_indices: NDArray[np.int64]
    covariance: NDArray[np.float64]
    standard_deviations: NDArray[np.float64]
    condition_number: float

    def to_dict(
        self,
        *,
        patch_ids: Sequence[int],
        isotope_names: Sequence[str],
    ) -> dict[str, object]:
        """Return support-indexed JSON diagnostics with physical labels."""
        isotope_count = len(tuple(isotope_names))
        entries = []
        for local_index, source_index in enumerate(self.active_source_indices):
            patch_index, isotope_index = divmod(int(source_index), isotope_count)
            entries.append(
                {
                    "source_index": int(source_index),
                    "patch_id": int(patch_ids[patch_index]),
                    "isotope": str(tuple(isotope_names)[isotope_index]),
                    "density_standard_deviation": float(
                        self.standard_deviations[local_index]
                    ),
                }
            )
        return {
            "method": "active_support_laplace_fisher",
            "active_parameter_count": len(entries),
            "condition_number": float(self.condition_number),
            "entries": entries,
            "covariance": self.covariance.tolist(),
        }


def active_support_laplace(
    response: NDArray[np.float64] | ResponseOperator,
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    densities: NDArray[np.float64],
    patch_areas_m2: NDArray[np.float64],
    fit_indices: NDArray[np.int64],
    *,
    support_threshold_fraction: float,
    maximum_active_parameters: int,
    ridge: float,
    overdispersion_alpha_by_bin: NDArray[np.float64] | None = None,
) -> LaplaceSupportResult:
    """Compute a conditional Fisher covariance on selected density columns."""
    del observed
    density = np.asarray(densities, dtype=float)
    maximum = max(float(np.max(density)), 1.0e-30)
    active = np.flatnonzero(
        density.reshape(-1) >= maximum * float(support_threshold_fraction)
    )
    if active.size == 0:
        active = np.asarray([int(np.argmax(density.reshape(-1)))], dtype=np.int64)
    if active.size > int(maximum_active_parameters):
        values = density.reshape(-1)[active]
        active = active[
            np.argsort(values, kind="stable")[-int(maximum_active_parameters) :]
        ]
    selected_mean = np.maximum(
        np.asarray(predicted, dtype=float)[fit_indices],
        1.0e-12,
    )
    if (
        overdispersion_alpha_by_bin is None
        or not np.asarray(overdispersion_alpha_by_bin).size
    ):
        variance = selected_mean
    else:
        alpha = np.asarray(overdispersion_alpha_by_bin, dtype=float)
        if selected_mean.ndim != 2 or alpha.shape != (selected_mean.shape[1],):
            raise ValueError("Laplace overdispersion alpha must match spectrum bins.")
        variance = selected_mean + alpha[None, :] * selected_mean**2
    weights = 1.0 / np.maximum(variance.reshape(-1), 1.0e-12)
    if isinstance(response, ResponseOperator):
        selected = response.select_measurements(fit_indices.tolist())
        gram = weighted_response_gram(selected, active, weights)
    else:
        values = np.asarray(response, dtype=float)[fit_indices]
        isotope_count = density.shape[1]
        area_scale = np.repeat(
            np.asarray(patch_areas_m2, dtype=float),
            isotope_count,
        )
        design = values.reshape(-1, density.size) * area_scale[None, :]
        selected_design = design[:, active]
        gram = selected_design.T @ (weights[:, None] * selected_design)
    diagonal_scale = max(float(np.max(np.diag(gram))), 1.0)
    precision = gram + float(ridge) * diagonal_scale * np.eye(active.size)
    condition = float(np.linalg.cond(precision))
    if not np.isfinite(condition):
        raise np.linalg.LinAlgError("Active-support Laplace precision is singular.")
    covariance = np.linalg.inv(precision)
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return LaplaceSupportResult(
        active_source_indices=active.astype(np.int64),
        covariance=covariance,
        standard_deviations=standard_deviations,
        condition_number=condition,
    )


def _station_bootstrap_sample(
    batch: ObservationBatch,
    rng: np.random.Generator,
) -> tuple[ObservationBatch, NDArray[np.int64]]:
    """Return one station-block resample and its original row indices."""
    station_ids = np.unique(batch.station_ids)
    sampled = rng.choice(station_ids, size=station_ids.size, replace=True)
    rows: list[int] = []
    new_station_ids: list[int] = []
    new_blocks: list[str] = []
    for new_station, old_station in enumerate(sampled):
        selected = np.flatnonzero(batch.station_ids == old_station)
        rows.extend(int(index) for index in selected)
        new_station_ids.extend([new_station] * selected.size)
        new_blocks.extend([f"bootstrap-station:{new_station}"] * selected.size)
    indices = np.asarray(rows, dtype=np.int64)
    measurement_count = indices.size
    resampled = ObservationBatch(
        detector_positions_xyz=batch.detector_positions_xyz[indices],
        detector_quaternions_wxyz=batch.detector_quaternions_wxyz[indices],
        fe_indices=batch.fe_indices[indices],
        pb_indices=batch.pb_indices[indices],
        live_times_s=batch.live_times_s[indices],
        spectrum_counts=batch.spectrum_counts[indices],
        spectrum_variances=(
            None
            if batch.spectrum_variances is None
            else batch.spectrum_variances[indices]
        ),
        energy_bin_edges_keV=batch.energy_bin_edges_keV,
        isotope_counts=(
            None if batch.isotope_counts is None else batch.isotope_counts[indices]
        ),
        isotope_covariances=(
            None
            if batch.isotope_covariances is None
            else batch.isotope_covariances[indices]
        ),
        station_ids=np.asarray(new_station_ids, dtype=np.int64),
        isotope_names=batch.isotope_names,
        step_ids=np.arange(measurement_count, dtype=np.int64),
        action_ids=np.arange(measurement_count, dtype=np.int64),
        travel_times_s=batch.travel_times_s[indices],
        shield_actuation_times_s=batch.shield_actuation_times_s[indices],
        shield_program_block_ids=tuple(new_blocks),
    )
    indices.setflags(write=False)
    return resampled, indices


def station_bootstrap_batch(
    batch: ObservationBatch,
    rng: np.random.Generator,
) -> ObservationBatch:
    """Resample whole station blocks with replacement and renumber causally."""
    return _station_bootstrap_sample(batch, rng)[0]


def augment_clusters_with_laplace(
    estimate: MLEEstimate,
    laplace: LaplaceSupportResult,
    *,
    confidence_level: float,
) -> list[dict[str, object]]:
    """Attach delta-method centroid covariance and strength intervals."""
    from scipy.stats import norm

    active_lookup = {
        int(source_index): local_index
        for local_index, source_index in enumerate(laplace.active_source_indices)
    }
    patch_by_id = {
        int(patch.patch_id): index for index, patch in enumerate(estimate.patches)
    }
    isotope_by_name = {
        isotope: index for index, isotope in enumerate(estimate.isotope_names)
    }
    isotope_count = len(estimate.isotope_names)
    critical = float(norm.ppf(0.5 + 0.5 * float(confidence_level)))
    result = []
    for cluster in _clusters(estimate):
        enriched = dict(cluster)
        isotope_index = isotope_by_name.get(str(cluster.get("isotope", "")))
        patch_ids = cluster.get("patch_ids", [])
        if isotope_index is None or not isinstance(patch_ids, Sequence):
            result.append(enriched)
            continue
        centroid = np.asarray(cluster.get("centroid_xyz"), dtype=float)
        total_strength = float(cluster.get("integrated_strength_cps_1m", 0.0))
        centroid_jacobian = np.zeros(
            (laplace.active_source_indices.size, 3),
            dtype=float,
        )
        strength_gradient = np.zeros(
            laplace.active_source_indices.size,
            dtype=float,
        )
        cluster_source_indices: list[int] = []
        for patch_id in patch_ids:
            patch_index = patch_by_id.get(int(patch_id))
            if patch_index is None:
                continue
            source_index = patch_index * isotope_count + isotope_index
            cluster_source_indices.append(source_index)
            local_index = active_lookup.get(source_index)
            if local_index is None:
                continue
            patch = estimate.patches[patch_index]
            area = float(patch.area_m2)
            strength_gradient[local_index] = area
            if total_strength > 0.0:
                centroid_jacobian[local_index] = (
                    area * (patch.centroid_xyz - centroid) / total_strength
                )
        unsupported = [
            source_index
            for source_index in cluster_source_indices
            if source_index not in active_lookup
        ]
        if not cluster_source_indices or unsupported:
            enriched["laplace_interval_status"] = "skipped_partial_support"
            enriched["laplace_unsupported_source_indices"] = unsupported
            result.append(enriched)
            continue
        centroid_covariance = (
            centroid_jacobian.T @ laplace.covariance @ centroid_jacobian
        )
        strength_variance = float(
            strength_gradient @ laplace.covariance @ strength_gradient
        )
        centroid_sd = np.sqrt(np.maximum(np.diag(centroid_covariance), 0.0))
        strength_sd = np.sqrt(max(strength_variance, 0.0))
        enriched["centroid_covariance_xyz_m2"] = centroid_covariance.tolist()
        enriched["centroid_interval_xyz_m"] = np.column_stack(
            (centroid - critical * centroid_sd, centroid + critical * centroid_sd)
        ).tolist()
        enriched["integrated_strength_interval_cps_1m"] = [
            max(0.0, total_strength - critical * strength_sd),
            total_strength + critical * strength_sd,
        ]
        enriched["uncertainty_method"] = "active_support_laplace_delta"
        enriched["laplace_interval_status"] = "complete_support"
        result.append(enriched)
    return result


def _clusters(estimate: MLEEstimate) -> list[dict[str, object]]:
    """Return copied cluster diagnostics from one estimate."""
    raw = estimate.diagnostics.get("hotspot_clusters", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [dict(value) for value in raw if isinstance(value, Mapping)]


def _project_patch_strengths_to_base(
    base_patches: Sequence[SurfacePatch],
    replicate_patches: Sequence[SurfacePatch],
) -> NDArray[np.float64]:
    """Return a mass-preserving rectangle-overlap projection onto base patches."""
    base = tuple(base_patches)
    replicate = tuple(replicate_patches)
    projection = np.zeros((len(base), len(replicate)), dtype=np.float64)
    base_groups: dict[tuple[str, str], list[int]] = {}
    replicate_groups: dict[tuple[str, str], list[int]] = {}
    for index, patch in enumerate(base):
        base_groups.setdefault((patch.surface_kind, patch.object_id), []).append(index)
    for index, patch in enumerate(replicate):
        replicate_groups.setdefault((patch.surface_kind, patch.object_id), []).append(
            index
        )
    if set(base_groups) != set(replicate_groups):
        raise ValueError("Bootstrap and base estimates cover different surfaces.")
    for group_key, base_indices_list in base_groups.items():
        replicate_indices_list = replicate_groups[group_key]
        base_indices = np.asarray(base_indices_list, dtype=np.int64)
        replicate_indices = np.asarray(replicate_indices_list, dtype=np.int64)
        reference = base[int(base_indices[0])]
        origin = reference.vertices_xyz[0]
        u_axis = reference.vertices_xyz[1] - origin
        v_axis = reference.vertices_xyz[3] - origin
        u_axis /= np.linalg.norm(u_axis)
        v_axis /= np.linalg.norm(v_axis)
        normal = reference.normal_xyz
        scale = max(
            1.0,
            *(float(patch.area_m2) ** 0.5 for patch in base),
            *(float(patch.area_m2) ** 0.5 for patch in replicate),
        )
        tolerance = 1.0e-8 * scale

        def bounds(
            indices: NDArray[np.int64], patches: tuple[SurfacePatch, ...]
        ) -> tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]:
            """Project one coplanar patch group into the reference face frame."""
            vertices = np.stack(
                [patches[int(index)].vertices_xyz for index in indices],
                axis=0,
            )
            normals = np.stack(
                [patches[int(index)].normal_xyz for index in indices],
                axis=0,
            )
            if np.any(normals @ normal < 1.0 - 1.0e-8):
                raise ValueError("Bootstrap surface orientations do not match base.")
            offsets = vertices - origin[None, None, :]
            if np.max(np.abs(offsets @ normal)) > tolerance:
                raise ValueError("Bootstrap surface planes do not match base.")
            u_values = offsets @ u_axis
            v_values = offsets @ v_axis
            return (
                np.min(u_values, axis=1),
                np.max(u_values, axis=1),
                np.min(v_values, axis=1),
                np.max(v_values, axis=1),
            )

        base_u0, base_u1, base_v0, base_v1 = bounds(base_indices, base)
        rep_u0, rep_u1, rep_v0, rep_v1 = bounds(replicate_indices, replicate)
        overlap_u = np.maximum(
            np.minimum(base_u1[:, None], rep_u1[None, :])
            - np.maximum(base_u0[:, None], rep_u0[None, :]),
            0.0,
        )
        overlap_v = np.maximum(
            np.minimum(base_v1[:, None], rep_v1[None, :])
            - np.maximum(base_v0[:, None], rep_v0[None, :]),
            0.0,
        )
        overlap = overlap_u * overlap_v
        covered_area = np.sum(overlap, axis=0)
        expected_area = np.asarray(
            [replicate[int(index)].area_m2 for index in replicate_indices],
            dtype=np.float64,
        )
        if not np.allclose(
            covered_area,
            expected_area,
            rtol=1.0e-8,
            atol=max(1.0e-12, tolerance**2),
        ):
            raise ValueError(
                "Bootstrap patch projection does not preserve surface area."
            )
        projection[np.ix_(base_indices, replicate_indices)] = overlap / np.maximum(
            covered_area[None, :],
            np.finfo(np.float64).tiny,
        )
    return projection


def _cluster_match_metadata(
    estimate: MLEEstimate,
    cluster: Mapping[str, object],
) -> tuple[set[str], set[str], float, bool]:
    """Return physical support gates for one reported hotspot cluster."""
    patch_ids = cluster.get("patch_ids", ())
    if not isinstance(patch_ids, Sequence) or isinstance(patch_ids, (str, bytes)):
        return set(), set(), 0.0, False
    by_id = {int(patch.patch_id): patch for patch in estimate.patches}
    patches = tuple(by_id.get(int(patch_id)) for patch_id in patch_ids)
    selected = tuple(patch for patch in patches if patch is not None)
    centroid = np.asarray(cluster.get("centroid_xyz"), dtype=np.float64)
    if not selected or centroid.shape != (3,) or np.any(~np.isfinite(centroid)):
        return set(), set(), 0.0, False
    vertices = np.concatenate([patch.vertices_xyz for patch in selected], axis=0)
    support_center = np.mean(vertices, axis=0)
    support_radius = float(np.max(np.linalg.norm(vertices - support_center, axis=1)))
    centroid_valid = bool(
        np.linalg.norm(centroid - support_center) <= support_radius + 1.0e-8
    )
    return (
        {str(patch.surface_kind) for patch in selected},
        {str(patch.object_id) for patch in selected},
        support_radius,
        centroid_valid,
    )


def bootstrap_uncertainty_summary(
    base: MLEEstimate,
    replicates: Sequence[MLEEstimate],
    *,
    confidence_level: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Aggregate station-bootstrap map, surface, z, and cluster uncertainty."""
    estimates = tuple(replicates)
    if not estimates:
        return {"replicate_count": 0}, _clusters(base)
    alpha = 0.5 * (1.0 - float(confidence_level))
    quantiles = (alpha, 1.0 - alpha)
    base_patch_ids = [int(patch.patch_id) for patch in base.patches]
    base_kinds = [str(patch.surface_kind) for patch in base.patches]
    projected_strengths = np.zeros(
        (len(estimates), len(base.isotope_names), len(base.patches)),
        dtype=np.float64,
    )
    for replicate_index, estimate in enumerate(estimates):
        projection = _project_patch_strengths_to_base(
            base.patches,
            estimate.patches,
        )
        isotope_lookup = {
            name: index for index, name in enumerate(estimate.isotope_names)
        }
        for base_isotope_index, isotope in enumerate(base.isotope_names):
            replicate_isotope_index = isotope_lookup.get(isotope)
            if replicate_isotope_index is None:
                raise ValueError(
                    "Bootstrap isotope names do not match the base estimate."
                )
            projected_strengths[replicate_index, base_isotope_index] = (
                projection
                @ np.asarray(
                    estimate.patch_strength_by_isotope[replicate_isotope_index],
                    dtype=np.float64,
                )
            )
    isotope_summaries: dict[str, object] = {}
    for isotope_index, isotope in enumerate(base.isotope_names):
        strength_samples = projected_strengths[:, isotope_index]
        surface_fraction_samples: dict[str, list[float]] = {
            "floor": [],
            "wall": [],
            "ceiling": [],
            "obstacle_top": [],
            "obstacle_side": [],
        }
        z_samples: list[float] = []
        ceiling_dominant: list[float] = []
        for replicate_index, estimate in enumerate(estimates):
            replicate_isotope_index = tuple(estimate.isotope_names).index(isotope)
            all_strengths = np.asarray(
                estimate.patch_strength_by_isotope[replicate_isotope_index],
                dtype=float,
            )
            total = max(float(np.sum(all_strengths)), 1.0e-30)
            for kind in surface_fraction_samples:
                mass = sum(
                    float(all_strengths[index])
                    for index, patch in enumerate(estimate.patches)
                    if patch.surface_kind == kind
                )
                surface_fraction_samples[kind].append(mass / total)
            z_samples.append(
                float(
                    np.average(
                        [patch.centroid_xyz[2] for patch in estimate.patches],
                        weights=np.maximum(all_strengths, 0.0),
                    )
                )
                if np.any(all_strengths > 0.0)
                else float("nan")
            )
            ceiling_dominant.append(
                float(surface_fraction_samples["ceiling"][-1] >= 0.5)
            )
        maximum_by_replicate = np.max(strength_samples, axis=1, keepdims=True)
        selected = (maximum_by_replicate > 0.0) & (
            strength_samples >= 1.0e-3 * maximum_by_replicate
        )
        finite_z = np.asarray(z_samples, dtype=np.float64)
        finite_z = finite_z[np.isfinite(finite_z)]
        isotope_summaries[isotope] = {
            "patch_ids": base_patch_ids,
            "patch_surface_kinds": base_kinds,
            "patch_selection_frequency": np.mean(selected, axis=0).tolist(),
            "patch_strength_interval_cps_1m": np.quantile(
                strength_samples,
                quantiles,
                axis=0,
            ).T.tolist(),
            "surface_mass_probability": {
                kind: {
                    "mean": float(np.mean(values)),
                    "interval": np.quantile(values, quantiles).tolist(),
                }
                for kind, values in surface_fraction_samples.items()
            },
            "z_interval_m": (
                np.quantile(finite_z, quantiles).tolist() if finite_z.size else None
            ),
            "z_interval_status": (
                "available" if finite_z.size else "unavailable_zero_strength"
            ),
            "ceiling_source_probability": float(np.mean(ceiling_dominant)),
        }

    base_clusters = _clusters(base)
    base_cluster_metadata = tuple(
        _cluster_match_metadata(base, cluster) for cluster in base_clusters
    )
    centroid_samples: list[list[NDArray[np.float64]]] = [
        [] for _cluster in base_clusters
    ]
    strength_samples_by_cluster: list[list[float]] = [[] for _cluster in base_clusters]
    from scipy.optimize import linear_sum_assignment

    for estimate in estimates:
        replicate_clusters = _clusters(estimate)
        replicate_cluster_metadata = tuple(
            _cluster_match_metadata(estimate, cluster) for cluster in replicate_clusters
        )
        isotopes = {str(cluster.get("isotope", "")) for cluster in base_clusters}
        for isotope in isotopes:
            base_indices = [
                index
                for index, cluster in enumerate(base_clusters)
                if str(cluster.get("isotope", "")) == isotope
            ]
            candidate_indices = [
                index
                for index, cluster in enumerate(replicate_clusters)
                if str(cluster.get("isotope", "")) == isotope
            ]
            if not base_indices or not candidate_indices:
                continue
            base_points = np.vstack(
                [
                    np.asarray(base_clusters[index].get("centroid_xyz"), dtype=float)
                    for index in base_indices
                ]
            )
            candidate_points = np.vstack(
                [
                    np.asarray(
                        replicate_clusters[index].get("centroid_xyz"),
                        dtype=float,
                    )
                    for index in candidate_indices
                ]
            )
            distances = np.linalg.norm(
                base_points[:, None, :] - candidate_points[None, :, :],
                axis=2,
            )
            valid = np.zeros_like(distances, dtype=bool)
            for base_local, base_index in enumerate(base_indices):
                base_kinds_set, base_objects, base_radius, base_valid = (
                    base_cluster_metadata[base_index]
                )
                for candidate_local, candidate_index in enumerate(candidate_indices):
                    (
                        candidate_kinds,
                        candidate_objects,
                        candidate_radius,
                        candidate_valid,
                    ) = replicate_cluster_metadata[candidate_index]
                    valid[base_local, candidate_local] = bool(
                        base_valid
                        and candidate_valid
                        and base_kinds_set.intersection(candidate_kinds)
                        and base_objects.intersection(candidate_objects)
                        and distances[base_local, candidate_local]
                        <= base_radius + candidate_radius + 1.0e-8
                    )
            if not np.any(valid):
                continue
            invalid_cost = max(float(np.max(distances)), 1.0) * 1.0e6
            matched_base, matched_candidate = linear_sum_assignment(
                np.where(valid, distances, invalid_cost)
            )
            for base_local, candidate_local in zip(
                matched_base,
                matched_candidate,
                strict=True,
            ):
                if not valid[int(base_local), int(candidate_local)]:
                    continue
                base_index = base_indices[int(base_local)]
                candidate_index = candidate_indices[int(candidate_local)]
                centroid_samples[base_index].append(
                    candidate_points[int(candidate_local)]
                )
                strength_samples_by_cluster[base_index].append(
                    float(
                        replicate_clusters[candidate_index].get(
                            "integrated_strength_cps_1m",
                            0.0,
                        )
                    )
                )

    augmented_clusters: list[dict[str, object]] = []
    for cluster_index, base_cluster in enumerate(base_clusters):
        matched_centroids = centroid_samples[cluster_index]
        matched_strengths = strength_samples_by_cluster[cluster_index]
        enriched = dict(base_cluster)
        enriched["bootstrap_selection_frequency"] = len(matched_centroids) / len(
            estimates
        )
        if matched_centroids:
            points = np.vstack(matched_centroids)
            enriched["centroid_interval_xyz_m"] = np.quantile(
                points,
                quantiles,
                axis=0,
            ).T.tolist()
            enriched["centroid_covariance_xyz_m2"] = (
                np.cov(points, rowvar=False).tolist()
                if len(points) > 1
                else np.zeros((3, 3), dtype=float).tolist()
            )
            enriched["integrated_strength_interval_cps_1m"] = np.quantile(
                matched_strengths,
                quantiles,
            ).tolist()
        augmented_clusters.append(enriched)
    return (
        {
            "method": "station_block_bootstrap",
            "replicate_count": len(estimates),
            "confidence_level": float(confidence_level),
            "isotopes": isotope_summaries,
        },
        augmented_clusters,
    )


__all__ = [
    "LaplaceSupportResult",
    "active_support_laplace",
    "augment_clusters_with_laplace",
    "bootstrap_uncertainty_summary",
    "station_bootstrap_batch",
]
