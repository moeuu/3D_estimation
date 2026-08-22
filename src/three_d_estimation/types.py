"""Validated data contracts for surface maximum-likelihood estimation.

The MLE unknown is an isotope-specific surface density measured in detector
``cps@1m / m^2``.  Multiplying a density by :attr:`SurfacePatch.area_m2`
therefore produces the integrated detector ``cps@1m`` strength of that patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


SurfaceKind = Literal[
    "floor",
    "ceiling",
    "wall",
    "obstacle_top",
    "obstacle_side",
]

SURFACE_DENSITY_UNIT = "detector_cps_1m_per_m2"
PATCH_STRENGTH_UNIT = "detector_cps_1m"
_SURFACE_KINDS = frozenset(
    {"floor", "ceiling", "wall", "obstacle_top", "obstacle_side"}
)


def _readonly_float_array(
    values: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    """Return an owned, finite, read-only float array with an optional shape."""
    array = np.array(values, dtype=float, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _readonly_integer_vector(
    values: ArrayLike,
    *,
    name: str,
    length: int,
    non_negative: bool = True,
) -> NDArray[np.int64]:
    """Return a validated, read-only integer vector without lossy coercion."""
    raw = np.asarray(values)
    if raw.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.number) or np.any(~np.isfinite(raw)):
        raise ValueError(f"{name} must contain finite integer values.")
    integers = np.asarray(raw, dtype=np.int64)
    if not np.array_equal(raw, integers):
        raise ValueError(f"{name} must contain integer values.")
    if non_negative and np.any(integers < 0):
        raise ValueError(f"{name} must be non-negative.")
    integers = np.array(integers, dtype=np.int64, copy=True)
    integers.setflags(write=False)
    return integers


def _validated_isotope_names(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return non-empty, unique isotope names in their declared order."""
    names = tuple(str(value).strip() for value in values)
    if not names or any(not value for value in names):
        raise ValueError("isotope_names must contain at least one non-empty name.")
    if len(set(names)) != len(names):
        raise ValueError("isotope_names must be unique.")
    return names


@dataclass(frozen=True, slots=True)
class SurfacePatch:
    """Describe one oriented rectangular surface element.

    ``quadrature_weights`` are normalized to one.  Area is deliberately kept
    separate so response construction has the unambiguous form
    ``live_time * area * sum(weight * kernel)``.
    """

    patch_id: int
    centroid_xyz: NDArray[np.float64]
    normal_xyz: NDArray[np.float64]
    area_m2: float
    surface_kind: SurfaceKind
    object_id: str
    vertices_xyz: NDArray[np.float64]
    quadrature_points_xyz: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]
    neighbor_patch_ids: tuple[int, ...] = ()
    neighbor_shared_edge_lengths_m: tuple[float, ...] = ()
    parent_patch_id: int | None = None
    refinement_level: int = 0

    def __post_init__(self) -> None:
        """Validate geometry, quadrature, graph metadata, and lineage."""
        patch_id = int(self.patch_id)
        if patch_id < 0 or patch_id != self.patch_id:
            raise ValueError("patch_id must be a non-negative integer.")
        area = float(self.area_m2)
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("area_m2 must be finite and positive.")
        kind = str(self.surface_kind)
        if kind not in _SURFACE_KINDS:
            raise ValueError(f"Unsupported surface_kind: {kind!r}.")
        object_id = str(self.object_id).strip()
        if not object_id:
            raise ValueError("object_id must be non-empty.")

        centroid = _readonly_float_array(
            self.centroid_xyz,
            name="centroid_xyz",
            shape=(3,),
        )
        normal = _readonly_float_array(
            self.normal_xyz,
            name="normal_xyz",
            shape=(3,),
        )
        normal_norm = float(np.linalg.norm(normal))
        if not np.isclose(normal_norm, 1.0, rtol=1.0e-9, atol=1.0e-9):
            raise ValueError("normal_xyz must be a unit vector.")
        vertices = _readonly_float_array(
            self.vertices_xyz,
            name="vertices_xyz",
            shape=(4, 3),
        )
        u_vector = vertices[1] - vertices[0]
        v_vector = vertices[3] - vertices[0]
        u_length = float(np.linalg.norm(u_vector))
        v_length = float(np.linalg.norm(v_vector))
        scale = max(u_length, v_length, 1.0)
        tolerance = 1.0e-9 * scale
        if u_length <= tolerance or v_length <= tolerance:
            raise ValueError("vertices_xyz must define a non-degenerate rectangle.")
        if not np.allclose(
            vertices[2],
            vertices[0] + u_vector + v_vector,
            rtol=1.0e-9,
            atol=tolerance,
        ):
            raise ValueError("vertices_xyz must be ordered around a parallelogram.")
        if not np.isclose(
            float(np.dot(u_vector, v_vector)),
            0.0,
            atol=1.0e-9 * u_length * v_length,
        ):
            raise ValueError("vertices_xyz must define a rectangle, not a skew face.")
        oriented_area = np.cross(u_vector, v_vector)
        computed_area = float(np.linalg.norm(oriented_area))
        if not np.isclose(computed_area, area, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError("area_m2 must equal the exact rectangular vertex area.")
        if not np.allclose(
            oriented_area / computed_area,
            normal,
            rtol=1.0e-9,
            atol=1.0e-9,
        ):
            raise ValueError(
                "vertices_xyz winding must agree with the declared normal_xyz."
            )
        expected_centroid = vertices[0] + 0.5 * (u_vector + v_vector)
        if not np.allclose(
            centroid,
            expected_centroid,
            rtol=1.0e-9,
            atol=tolerance,
        ):
            raise ValueError("centroid_xyz must be the exact rectangle centroid.")

        quadrature_points = _readonly_float_array(
            self.quadrature_points_xyz,
            name="quadrature_points_xyz",
        )
        if quadrature_points.ndim != 2 or quadrature_points.shape[1] != 3:
            raise ValueError("quadrature_points_xyz must have shape (Q, 3).")
        if quadrature_points.shape[0] not in {1, 4}:
            raise ValueError(
                "Each rectangular patch must use 1 or 4 quadrature points."
            )
        quadrature_weights = _readonly_float_array(
            self.quadrature_weights,
            name="quadrature_weights",
            shape=(quadrature_points.shape[0],),
        )
        if np.any(quadrature_weights <= 0.0):
            raise ValueError("quadrature_weights must be strictly positive.")
        if not np.isclose(
            float(np.sum(quadrature_weights)),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("quadrature_weights must sum to one.")
        relative = quadrature_points - vertices[0]
        alpha = relative @ u_vector / float(np.dot(u_vector, u_vector))
        beta = relative @ v_vector / float(np.dot(v_vector, v_vector))
        projected = (
            vertices[0]
            + alpha[:, None] * u_vector[None, :]
            + beta[:, None] * v_vector[None, :]
        )
        if not np.allclose(projected, quadrature_points, rtol=0.0, atol=tolerance):
            raise ValueError("quadrature points must lie in the patch plane.")
        if np.any(alpha < -1.0e-10) or np.any(alpha > 1.0 + 1.0e-10):
            raise ValueError("quadrature points must lie inside the patch rectangle.")
        if np.any(beta < -1.0e-10) or np.any(beta > 1.0 + 1.0e-10):
            raise ValueError("quadrature points must lie inside the patch rectangle.")

        neighbor_ids = tuple(int(value) for value in self.neighbor_patch_ids)
        edge_lengths = tuple(
            float(value) for value in self.neighbor_shared_edge_lengths_m
        )
        if len(neighbor_ids) != len(edge_lengths):
            raise ValueError(
                "neighbor_patch_ids and neighbor_shared_edge_lengths_m must align."
            )
        if any(value < 0 or value == patch_id for value in neighbor_ids):
            raise ValueError("Neighbors must be distinct, non-negative, non-self IDs.")
        if len(set(neighbor_ids)) != len(neighbor_ids):
            raise ValueError("neighbor_patch_ids must not contain duplicates.")
        if any(not np.isfinite(value) or value <= 0.0 for value in edge_lengths):
            raise ValueError("Shared-edge lengths must be finite and positive.")
        sorted_neighbors = sorted(
            zip(neighbor_ids, edge_lengths), key=lambda item: item[0]
        )
        neighbor_ids = tuple(item[0] for item in sorted_neighbors)
        edge_lengths = tuple(item[1] for item in sorted_neighbors)

        parent_id = self.parent_patch_id
        if parent_id is not None:
            parent_id = int(parent_id)
            if parent_id < 0 or parent_id == patch_id:
                raise ValueError("parent_patch_id must be a distinct non-negative ID.")
        refinement_level = int(self.refinement_level)
        if refinement_level < 0 or refinement_level != self.refinement_level:
            raise ValueError("refinement_level must be a non-negative integer.")
        if refinement_level > 0 and parent_id is None:
            raise ValueError("A refined patch must retain its parent_patch_id.")

        object.__setattr__(self, "patch_id", patch_id)
        object.__setattr__(self, "centroid_xyz", centroid)
        object.__setattr__(self, "normal_xyz", normal)
        object.__setattr__(self, "area_m2", area)
        object.__setattr__(self, "surface_kind", kind)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "vertices_xyz", vertices)
        object.__setattr__(self, "quadrature_points_xyz", quadrature_points)
        object.__setattr__(self, "quadrature_weights", quadrature_weights)
        object.__setattr__(self, "neighbor_patch_ids", neighbor_ids)
        object.__setattr__(self, "neighbor_shared_edge_lengths_m", edge_lengths)
        object.__setattr__(self, "parent_patch_id", parent_id)
        object.__setattr__(self, "refinement_level", refinement_level)

    @property
    def density_unit(self) -> str:
        """Return the physical unit of the MLE unknown on this patch."""
        return SURFACE_DENSITY_UNIT

    @property
    def quadrature_count(self) -> int:
        """Return the number of normalized integration points."""
        return int(self.quadrature_weights.size)

    def integrated_strength_cps_1m(self, density_cps_1m_m2: float) -> float:
        """Convert one surface density to integrated patch strength."""
        density = float(density_cps_1m_m2)
        if not np.isfinite(density) or density < 0.0:
            raise ValueError("density_cps_1m_m2 must be finite and non-negative.")
        return density * self.area_m2


@dataclass(frozen=True, slots=True)
class SurfacePatchSet:
    """Store ordered patches and their physical shared-edge graph.

    ``adjacency_edges`` contains dense indices into :attr:`patches`, as expected
    by sparse numerical solvers.  Individual patches expose stable neighbor
    IDs, and :attr:`adjacency_patch_id_edges` provides the corresponding stable
    edge representation after refinement leaves gaps in the ID sequence.
    """

    patches: tuple[SurfacePatch, ...]
    adjacency_edges: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )
    shared_edge_lengths_m: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    def __post_init__(self) -> None:
        """Validate stable IDs and exact agreement with per-patch neighbors."""
        patches = tuple(self.patches)
        if not patches:
            raise ValueError("SurfacePatchSet must contain at least one patch.")
        if any(not isinstance(patch, SurfacePatch) for patch in patches):
            raise TypeError("patches must contain only SurfacePatch instances.")
        patch_ids = tuple(patch.patch_id for patch in patches)
        if len(set(patch_ids)) != len(patch_ids):
            raise ValueError("SurfacePatch patch_id values must be unique.")
        edges = np.asarray(self.adjacency_edges, dtype=np.int64)
        if edges.size == 0:
            edges = np.zeros((0, 2), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("adjacency_edges must have shape (E, 2).")
        lengths = np.asarray(self.shared_edge_lengths_m, dtype=float).reshape(-1)
        if lengths.size != edges.shape[0]:
            raise ValueError("shared_edge_lengths_m must contain one value per edge.")
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
            raise ValueError("shared_edge_lengths_m must be finite and positive.")
        if edges.size:
            if np.any(edges < 0) or np.any(edges >= len(patches)):
                raise ValueError(
                    "adjacency_edges must reference existing patch indices."
                )
            if np.any(edges[:, 0] == edges[:, 1]):
                raise ValueError("adjacency_edges cannot contain self edges.")
            canonical = np.sort(edges, axis=1)
            order = np.lexsort((canonical[:, 1], canonical[:, 0]))
            canonical = canonical[order]
            lengths = lengths[order]
            if canonical.shape[0] > 1 and np.any(
                np.all(canonical[1:] == canonical[:-1], axis=1)
            ):
                raise ValueError("adjacency_edges must be unique.")
            edges = canonical

        expected_neighbors: dict[int, dict[int, float]] = {
            patch_id: {} for patch_id in patch_ids
        }
        for edge, length in zip(edges, lengths):
            first_index, second_index = int(edge[0]), int(edge[1])
            first = patches[first_index].patch_id
            second = patches[second_index].patch_id
            expected_neighbors[first][second] = float(length)
            expected_neighbors[second][first] = float(length)
        for patch in patches:
            expected = expected_neighbors[patch.patch_id]
            actual = dict(
                zip(
                    patch.neighbor_patch_ids,
                    patch.neighbor_shared_edge_lengths_m,
                )
            )
            if set(actual) != set(expected):
                raise ValueError(
                    f"Patch {patch.patch_id} neighbor IDs do not match adjacency_edges."
                )
            if any(
                not np.isclose(actual[key], expected[key], rtol=1.0e-9, atol=1.0e-12)
                for key in actual
            ):
                raise ValueError(
                    f"Patch {patch.patch_id} shared-edge lengths do not match the graph."
                )

        edges = np.array(edges, dtype=np.int64, copy=True)
        lengths = np.array(lengths, dtype=float, copy=True)
        edges.setflags(write=False)
        lengths.setflags(write=False)
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "adjacency_edges", edges)
        object.__setattr__(self, "shared_edge_lengths_m", lengths)

    @property
    def patch_count(self) -> int:
        """Return the number of active patches."""
        return len(self.patches)

    @property
    def density_unit(self) -> str:
        """Return the unknown's physical unit."""
        return SURFACE_DENSITY_UNIT

    @property
    def patch_ids(self) -> NDArray[np.int64]:
        """Return stable patch IDs in numerical array order."""
        result = np.asarray([patch.patch_id for patch in self.patches], dtype=np.int64)
        result.setflags(write=False)
        return result

    @property
    def centroids_xyz(self) -> NDArray[np.float64]:
        """Return patch centroids in numerical array order."""
        result = np.vstack([patch.centroid_xyz for patch in self.patches])
        result.setflags(write=False)
        return result

    @property
    def normals_xyz(self) -> NDArray[np.float64]:
        """Return patch normals in numerical array order."""
        result = np.vstack([patch.normal_xyz for patch in self.patches])
        result.setflags(write=False)
        return result

    @property
    def kinds(self) -> tuple[str, ...]:
        """Return surface kinds in numerical array order."""
        return tuple(patch.surface_kind for patch in self.patches)

    @property
    def surface_kinds(self) -> tuple[str, ...]:
        """Return a descriptive alias for :attr:`kinds`."""
        return self.kinds

    @property
    def object_ids(self) -> tuple[str, ...]:
        """Return physical object/face identifiers in array order."""
        return tuple(patch.object_id for patch in self.patches)

    @property
    def face_ids(self) -> tuple[str, ...]:
        """Return the common-runtime alias for object identifiers."""
        return self.object_ids

    @property
    def areas_m2(self) -> NDArray[np.float64]:
        """Return exact patch areas in numerical array order."""
        result = np.asarray([patch.area_m2 for patch in self.patches], dtype=float)
        result.setflags(write=False)
        return result

    @property
    def adjacency_weights(self) -> NDArray[np.float64]:
        """Return physical graph-TV weights (shared-edge lengths in metres)."""
        return self.shared_edge_lengths_m

    @property
    def adjacency_index_edges(self) -> NDArray[np.int64]:
        """Return adjacency endpoints as dense indices into :attr:`patches`."""
        result = np.array(self.adjacency_edges, dtype=np.int64, copy=True).reshape(
            -1, 2
        )
        result.setflags(write=False)
        return result

    @property
    def adjacency_patch_id_edges(self) -> NDArray[np.int64]:
        """Return adjacency endpoints as stable patch IDs."""
        patch_ids = self.patch_ids
        result = patch_ids[self.adjacency_edges]
        result.setflags(write=False)
        return result

    @property
    def quadrature_points_xyz(self) -> NDArray[np.float64]:
        """Return dense padded quadrature points with shape ``(G, Q, 3)``."""
        maximum_count = max(patch.quadrature_count for patch in self.patches)
        result = np.empty((self.patch_count, maximum_count, 3), dtype=float)
        for index, patch in enumerate(self.patches):
            count = patch.quadrature_count
            result[index, :count] = patch.quadrature_points_xyz
            result[index, count:] = patch.quadrature_points_xyz[-1]
        result.setflags(write=False)
        return result

    @property
    def quadrature_weights(self) -> NDArray[np.float64]:
        """Return dense padded normalized weights with shape ``(G, Q)``."""
        maximum_count = max(patch.quadrature_count for patch in self.patches)
        result = np.zeros((self.patch_count, maximum_count), dtype=float)
        for index, patch in enumerate(self.patches):
            result[index, : patch.quadrature_count] = patch.quadrature_weights
        result.setflags(write=False)
        return result

    @property
    def total_area_m2(self) -> float:
        """Return total active surface area."""
        return float(np.sum(self.areas_m2))

    def patch_by_id(self, patch_id: int) -> SurfacePatch:
        """Return one patch by stable ID."""
        requested = int(patch_id)
        for patch in self.patches:
            if patch.patch_id == requested:
                return patch
        raise KeyError(f"Unknown patch_id: {requested}")

    def integrated_strengths_cps_1m(
        self,
        densities_cps_1m_m2: ArrayLike,
    ) -> NDArray[np.float64]:
        """Convert arrays whose final axis is patch density to strengths."""
        densities = np.asarray(densities_cps_1m_m2, dtype=float)
        if densities.ndim < 1 or densities.shape[-1] != self.patch_count:
            raise ValueError("Density arrays must have one final entry per patch.")
        if np.any(~np.isfinite(densities)) or np.any(densities < 0.0):
            raise ValueError("Surface densities must be finite and non-negative.")
        return densities * self.areas_m2


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Store estimator-independent observations for an incremental update."""

    detector_positions_xyz: NDArray[np.float64]
    detector_quaternions_wxyz: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    spectrum_counts: NDArray[np.float64]
    spectrum_variances: NDArray[np.float64] | None
    energy_bin_edges_keV: NDArray[np.float64]
    isotope_counts: NDArray[np.float64] | None
    isotope_covariances: NDArray[np.float64] | None
    station_ids: NDArray[np.int64]
    isotope_names: tuple[str, ...]
    step_ids: NDArray[np.int64] | None = None
    action_ids: NDArray[np.int64] | None = None
    travel_times_s: NDArray[np.float64] | None = None
    shield_actuation_times_s: NDArray[np.float64] | None = None
    shield_program_block_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate row alignment, spectra, isotope channels, and covariance."""
        positions = np.asarray(self.detector_positions_xyz, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
            raise ValueError("detector_positions_xyz must have shape (M, 3), M >= 1.")
        measurement_count = int(positions.shape[0])
        positions = _readonly_float_array(
            positions,
            name="detector_positions_xyz",
            shape=(measurement_count, 3),
        )
        quaternions = _readonly_float_array(
            self.detector_quaternions_wxyz,
            name="detector_quaternions_wxyz",
            shape=(measurement_count, 4),
        )
        quaternion_norms = np.linalg.norm(quaternions, axis=1)
        if not np.allclose(quaternion_norms, 1.0, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("detector quaternions must be normalized.")
        fe_indices = _readonly_integer_vector(
            self.fe_indices,
            name="fe_indices",
            length=measurement_count,
        )
        pb_indices = _readonly_integer_vector(
            self.pb_indices,
            name="pb_indices",
            length=measurement_count,
        )
        if np.any(fe_indices > 7) or np.any(pb_indices > 7):
            raise ValueError("Fe/Pb orientation indices must lie in [0, 7].")
        station_ids = _readonly_integer_vector(
            self.station_ids,
            name="station_ids",
            length=measurement_count,
        )
        live_times = _readonly_float_array(
            self.live_times_s,
            name="live_times_s",
            shape=(measurement_count,),
        )
        if np.any(live_times <= 0.0):
            raise ValueError("live_times_s must be strictly positive.")
        step_ids = _readonly_integer_vector(
            (
                np.arange(measurement_count, dtype=np.int64)
                if self.step_ids is None
                else self.step_ids
            ),
            name="step_ids",
            length=measurement_count,
        )
        if measurement_count > 1 and np.any(np.diff(step_ids) <= 0):
            raise ValueError(
                "step_ids must be strictly increasing in causal row order."
            )
        action_ids = _readonly_integer_vector(
            step_ids if self.action_ids is None else self.action_ids,
            name="action_ids",
            length=measurement_count,
        )
        if np.unique(action_ids).size != measurement_count:
            raise ValueError("action_ids must be unique measurement-action IDs.")
        if measurement_count > 1 and np.any(np.diff(station_ids) < 0):
            raise ValueError("station_ids must be nondecreasing in causal row order.")
        travel_times = _readonly_float_array(
            (
                np.zeros(measurement_count, dtype=float)
                if self.travel_times_s is None
                else self.travel_times_s
            ),
            name="travel_times_s",
            shape=(measurement_count,),
        )
        shield_actuation_times = _readonly_float_array(
            (
                np.zeros(measurement_count, dtype=float)
                if self.shield_actuation_times_s is None
                else self.shield_actuation_times_s
            ),
            name="shield_actuation_times_s",
            shape=(measurement_count,),
        )
        if np.any(travel_times < 0.0) or np.any(shield_actuation_times < 0.0):
            raise ValueError("Travel and shield-actuation times must be non-negative.")
        if self.shield_program_block_ids is None:
            block_ids = tuple(f"station:{int(value)}" for value in station_ids)
        else:
            block_ids = tuple(
                str(value).strip() for value in self.shield_program_block_ids
            )
        if len(block_ids) != measurement_count or any(not value for value in block_ids):
            raise ValueError(
                "shield_program_block_ids must contain one non-empty ID per row."
            )

        spectrum = np.asarray(self.spectrum_counts, dtype=float)
        if spectrum.ndim != 2 or spectrum.shape[0] != measurement_count:
            raise ValueError("spectrum_counts must have shape (M, B).")
        if spectrum.shape[1] < 1:
            raise ValueError("spectrum_counts must contain at least one energy bin.")
        spectrum = _readonly_float_array(
            spectrum,
            name="spectrum_counts",
            shape=(measurement_count, int(spectrum.shape[1])),
        )
        if np.any(spectrum < 0.0):
            raise ValueError("spectrum_counts must be non-negative.")
        bin_count = int(spectrum.shape[1])
        edges = _readonly_float_array(
            self.energy_bin_edges_keV,
            name="energy_bin_edges_keV",
            shape=(bin_count + 1,),
        )
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("energy_bin_edges_keV must be strictly increasing.")

        variances = None
        if self.spectrum_variances is not None:
            variances = _readonly_float_array(
                self.spectrum_variances,
                name="spectrum_variances",
                shape=(measurement_count, bin_count),
            )
            if np.any(variances < 0.0):
                raise ValueError("spectrum_variances must be non-negative.")

        isotope_names = _validated_isotope_names(self.isotope_names)
        isotope_count = len(isotope_names)
        counts = None
        if self.isotope_counts is not None:
            counts = _readonly_float_array(
                self.isotope_counts,
                name="isotope_counts",
                shape=(measurement_count, isotope_count),
            )
            if np.any(counts < 0.0):
                raise ValueError("isotope_counts must be non-negative.")
        covariances = None
        if self.isotope_covariances is not None:
            if counts is None:
                raise ValueError(
                    "isotope_covariances require corresponding isotope_counts."
                )
            covariances = _readonly_float_array(
                self.isotope_covariances,
                name="isotope_covariances",
                shape=(measurement_count, isotope_count, isotope_count),
            )
            if not np.allclose(
                covariances,
                np.swapaxes(covariances, 1, 2),
                rtol=1.0e-8,
                atol=1.0e-10,
            ):
                raise ValueError("isotope_covariances must be symmetric.")
            eigenvalues = np.linalg.eigvalsh(covariances)
            covariance_scale = np.maximum(
                np.max(np.abs(eigenvalues), axis=1, keepdims=True),
                1.0,
            )
            if np.any(eigenvalues < -1.0e-9 * covariance_scale):
                raise ValueError("isotope_covariances must be positive semidefinite.")

        object.__setattr__(self, "detector_positions_xyz", positions)
        object.__setattr__(self, "detector_quaternions_wxyz", quaternions)
        object.__setattr__(self, "fe_indices", fe_indices)
        object.__setattr__(self, "pb_indices", pb_indices)
        object.__setattr__(self, "live_times_s", live_times)
        object.__setattr__(self, "spectrum_counts", spectrum)
        object.__setattr__(self, "spectrum_variances", variances)
        object.__setattr__(self, "energy_bin_edges_keV", edges)
        object.__setattr__(self, "isotope_counts", counts)
        object.__setattr__(self, "isotope_covariances", covariances)
        object.__setattr__(self, "station_ids", station_ids)
        object.__setattr__(self, "isotope_names", isotope_names)
        object.__setattr__(self, "step_ids", step_ids)
        object.__setattr__(self, "action_ids", action_ids)
        object.__setattr__(self, "travel_times_s", travel_times)
        object.__setattr__(
            self,
            "shield_actuation_times_s",
            shield_actuation_times,
        )
        object.__setattr__(self, "shield_program_block_ids", block_ids)

    @property
    def measurement_count(self) -> int:
        """Return the number of finalized measurement rows."""
        return int(self.detector_positions_xyz.shape[0])

    @property
    def energy_bin_count(self) -> int:
        """Return the number of spectrum bins."""
        return int(self.spectrum_counts.shape[1])

    @property
    def isotope_count(self) -> int:
        """Return the number of declared isotope channels."""
        return len(self.isotope_names)


@dataclass(frozen=True, slots=True)
class MLEEstimate:
    """Store an isotope-wise surface estimate and fit diagnostics."""

    isotope_names: tuple[str, ...]
    patches: tuple[SurfacePatch, ...]
    density_by_isotope: NDArray[np.float64]
    patch_strength_by_isotope: NDArray[np.float64]
    predicted_spectra: NDArray[np.float64] | None
    predicted_isotope_counts: NDArray[np.float64] | None
    background_parameters: NDArray[np.float64]
    nuisance_parameters: NDArray[np.float64]
    objective_value: float
    poisson_deviance: float
    iterations: int
    converged: bool
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        """Validate physical units, shapes, non-negativity, and convergence data."""
        isotope_names = _validated_isotope_names(self.isotope_names)
        patches = tuple(self.patches)
        if not patches or any(not isinstance(patch, SurfacePatch) for patch in patches):
            raise ValueError("patches must contain at least one SurfacePatch.")
        if len({patch.patch_id for patch in patches}) != len(patches):
            raise ValueError("MLEEstimate patches must have unique patch IDs.")
        isotope_count = len(isotope_names)
        patch_count = len(patches)
        density = _readonly_float_array(
            self.density_by_isotope,
            name="density_by_isotope",
            shape=(isotope_count, patch_count),
        )
        strength = _readonly_float_array(
            self.patch_strength_by_isotope,
            name="patch_strength_by_isotope",
            shape=(isotope_count, patch_count),
        )
        if np.any(density < 0.0) or np.any(strength < 0.0):
            raise ValueError(
                "Estimated densities and patch strengths must be non-negative."
            )
        areas = np.asarray([patch.area_m2 for patch in patches], dtype=float)
        if not np.allclose(
            strength,
            density * areas[None, :],
            rtol=1.0e-7,
            atol=1.0e-10,
        ):
            raise ValueError(
                "patch_strength_by_isotope must equal density_by_isotope * patch area."
            )

        predicted_spectra = None
        measurement_count: int | None = None
        if self.predicted_spectra is not None:
            predicted_spectra = _readonly_float_array(
                self.predicted_spectra,
                name="predicted_spectra",
            )
            if predicted_spectra.ndim != 2:
                raise ValueError("predicted_spectra must have shape (M, B).")
            if np.any(predicted_spectra < 0.0):
                raise ValueError("predicted_spectra must be non-negative.")
            measurement_count = int(predicted_spectra.shape[0])
        predicted_counts = None
        if self.predicted_isotope_counts is not None:
            raw_counts = np.asarray(self.predicted_isotope_counts)
            if raw_counts.ndim != 2 or raw_counts.shape[1] != isotope_count:
                raise ValueError("predicted_isotope_counts must have shape (M, I).")
            if (
                measurement_count is not None
                and raw_counts.shape[0] != measurement_count
            ):
                raise ValueError("Predicted spectrum and isotope rows must align.")
            predicted_counts = _readonly_float_array(
                raw_counts,
                name="predicted_isotope_counts",
                shape=(int(raw_counts.shape[0]), isotope_count),
            )
            if np.any(predicted_counts < 0.0):
                raise ValueError("predicted_isotope_counts must be non-negative.")
        background = _readonly_float_array(
            self.background_parameters,
            name="background_parameters",
        ).reshape(-1)
        if np.any(background < 0.0):
            raise ValueError("background_parameters must be non-negative.")
        background.setflags(write=False)
        nuisance = _readonly_float_array(
            self.nuisance_parameters,
            name="nuisance_parameters",
        ).reshape(-1)
        nuisance.setflags(write=False)
        objective = float(self.objective_value)
        deviance = float(self.poisson_deviance)
        if not np.isfinite(objective):
            raise ValueError("objective_value must be finite.")
        if not np.isfinite(deviance) or deviance < 0.0:
            raise ValueError("poisson_deviance must be finite and non-negative.")
        iterations = int(self.iterations)
        if iterations < 0 or iterations != self.iterations:
            raise ValueError("iterations must be a non-negative integer.")
        if not isinstance(self.converged, (bool, np.bool_)):
            raise ValueError("converged must be boolean.")
        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary.")

        object.__setattr__(self, "isotope_names", isotope_names)
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "density_by_isotope", density)
        object.__setattr__(self, "patch_strength_by_isotope", strength)
        object.__setattr__(self, "predicted_spectra", predicted_spectra)
        object.__setattr__(self, "predicted_isotope_counts", predicted_counts)
        object.__setattr__(self, "background_parameters", background)
        object.__setattr__(self, "nuisance_parameters", nuisance)
        object.__setattr__(self, "objective_value", objective)
        object.__setattr__(self, "poisson_deviance", deviance)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


__all__ = [
    "MLEEstimate",
    "ObservationBatch",
    "PATCH_STRENGTH_UNIT",
    "SURFACE_DENSITY_UNIT",
    "SurfaceKind",
    "SurfacePatch",
    "SurfacePatchSet",
]
