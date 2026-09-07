import numpy as np

from radar_gs.ply_evaluation import (
    PlyTable,
    filter_by_support,
    ground_truth_metrics,
    intrinsic_point_metrics,
    make_reference_voxel_grid,
    read_ascii_ply,
    voxel_overlap_metrics,
    voxelize_gaussian_support,
    voxelize_gaussians,
    write_voxel_surface_ply,
    write_ascii_ply,
)


def _table() -> PlyTable:
    properties = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "target_support_probability",
    ]
    values = np.asarray(
        [
            [0.0, 0.0, 0.0, 0, 0, 1, 0.05],
            [0.1, 0.0, 0.0, 0, 0, 1, 0.20],
            [0.2, 0.0, 0.0, 0, 0, 1, 0.40],
            [0.3, 0.0, 0.0, 0, 0, 1, 0.80],
        ],
        dtype=np.float64,
    )
    return PlyTable(properties, values)


def test_q_threshold_filter_and_roundtrip(tmp_path):
    filtered = filter_by_support(_table(), 0.3)
    assert len(filtered.values) == 2
    path = tmp_path / "filtered.ply"
    write_ascii_ply(path, filtered)
    loaded = read_ascii_ply(path)
    np.testing.assert_allclose(loaded.values, filtered.values)
    assert loaded.properties == filtered.properties


def test_intrinsic_and_gt_metrics_are_finite():
    table = filter_by_support(_table(), 0.1)
    points = table.values[:, :3]
    support = table.column("target_support_probability")
    intrinsic = intrinsic_point_metrics(points, support, graph_neighbors=2)
    assert intrinsic["point_count"] == 3
    assert 0.0 <= intrinsic["largest_component_ratio"] <= 1.0
    reference = points + np.asarray([0.01, 0.0, 0.0])
    gt = ground_truth_metrics(points, reference, [0.02])
    assert gt["available"]
    assert gt["distance_threshold_scores"]["0.02"]["f_score"] == 1.0
    assert gt["accuracy_p95_m"] > 0.0
    assert gt["completeness_p95_m"] > 0.0
    assert gt["hausdorff_p95_m"] > 0.0
    assert gt["normalized_chamfer_l1"] > 0.0


def test_voxel_overlap_counts_outside_prediction_as_false_positive():
    reference_points = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    spec = make_reference_voxel_grid(reference_points, resolution=16, padding_fraction=0.1)
    reference = np.zeros((16, 16, 16), dtype=bool)
    reference[4:6, 4:6, 4:6] = True
    prediction = reference.copy()
    metrics = voxel_overlap_metrics(
        prediction, reference, spec, prediction_outside_count=1
    )
    assert metrics["intersection_voxel_count"] == 8
    assert metrics["reference_voxel_count"] == 8
    assert metrics["prediction_voxel_count"] == 9
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 8.0 / 9.0
    assert metrics["iou_3d"] == 8.0 / 9.0


def test_oriented_gaussian_voxelization_is_nonempty():
    reference_points = np.asarray([[-1, -1, -1], [1, 1, 1]], dtype=np.float64)
    spec = make_reference_voxel_grid(reference_points, resolution=24, padding_fraction=0.0)
    grid, outside = voxelize_gaussians(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 1.0]]),
        np.asarray([[0.3, 0.2, 0.1]]),
        spec,
        sigma_extent=2.0,
        fill_interior=False,
    )
    assert np.count_nonzero(grid) > 0
    assert outside == 0


def test_noisy_or_support_requires_overlap_for_low_q_gaussians():
    reference_points = np.asarray([[-1, -1, -1], [1, 1, 1]], dtype=np.float64)
    spec = make_reference_voxel_grid(reference_points, resolution=24, padding_fraction=0.0)
    mean = np.asarray([[0.0, 0.0, 0.0]])
    normal = np.asarray([[0.0, 0.0, 1.0]])
    scale = np.asarray([[0.3, 0.3, 0.3]])
    single = voxelize_gaussian_support(
        mean,
        normal,
        scale,
        np.asarray([0.4]),
        spec,
        sigma_extent=2.0,
        occupancy_probability_threshold=0.5,
        fill_interior=False,
    )
    assert not np.any(single.occupied_grid)

    double = voxelize_gaussian_support(
        np.repeat(mean, 2, axis=0),
        np.repeat(normal, 2, axis=0),
        np.repeat(scale, 2, axis=0),
        np.asarray([0.4, 0.4]),
        spec,
        sigma_extent=2.0,
        occupancy_probability_threshold=0.5,
        fill_interior=False,
    )
    assert np.any(double.occupied_grid)
    assert double.overlap_count_grid.max() == 2
    assert double.summary()["multi_gaussian_overlap_voxel_count"] > 0


def test_fused_voxel_surface_ply_contains_faces(tmp_path):
    reference_points = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    spec = make_reference_voxel_grid(reference_points, resolution=8, padding_fraction=0.0)
    occupied = np.zeros((8, 8, 8), dtype=bool)
    occupied[3:5, 3:5, 3:5] = True
    path = tmp_path / "fused.ply"
    write_voxel_surface_ply(path, occupied, spec)
    text = path.read_text(encoding="utf-8")
    assert "element face 24" in text
    assert "property list uchar int vertex_indices" in text
