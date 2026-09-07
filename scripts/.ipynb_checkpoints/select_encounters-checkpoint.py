from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank equal-cost encounters using acquisition geometry only; "
            "target structure and ISAR intensity are never used"
        )
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        help="Encounter counts to optimize independently; default: every K from 1 to N",
    )
    parser.add_argument("--frames_per_encounter", type=int, default=120)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument(
        "--images_per_encounter",
        type=int,
        required=True,
        help="Equal number of aperture-spanning images retained per encounter",
    )
    parser.add_argument("--strict_equal_available_frames", action="store_true", default=True)
    parser.add_argument(
        "--allow_unequal_available_frames",
        action="store_false",
        dest="strict_equal_available_frames",
    )
    parser.add_argument("--sphere_samples", type=int, default=20000)
    parser.add_argument("--direction_cap_deg", type=float, default=7.5)
    parser.add_argument("--direction_weight", type=float, default=1.0)
    parser.add_argument("--observability_weight", type=float, default=1.0)
    parser.add_argument("--baseline_weight", type=float, default=0.25)
    parser.add_argument("--max_eval_views", type=int, default=200)
    parser.add_argument(
        "--max_exact_combinations",
        type=int,
        default=50_000,
        help="Safety limit for exhaustive fixed-K subset evaluation",
    )
    parser.add_argument(
        "--search_restarts",
        type=int,
        default=8,
        help="Independent starts for approximate fixed-K search when exhaustive search is too large",
    )
    parser.add_argument("--search_max_passes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _numeric_key(path: Path) -> tuple:
    values = re.findall(r"\d+", path.name)
    return (0, *(int(value) for value in values)) if values else (1, path.name)


def _view_angle_path(encounter: Path) -> Path:
    for candidate in (encounter / "view_angle.npy", encounter / "Es" / "view_angle.npy"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No view_angle.npy found for {encounter}")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-12)


def _fixed_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("images_per_encounter must be positive")
    if count > len(indices):
        raise ValueError(f"Requested {count} images from only {len(indices)} available frames")
    if count == len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    chosen = [indices[int(position)] for position in positions]
    if len(set(chosen)) != count:
        raise RuntimeError("Fixed-count selection generated duplicate indices")
    return chosen


def _fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _aperture_metrics(kr: np.ndarray) -> dict[str, float]:
    increments = _angle_deg(kr[1:], kr[:-1]) if len(kr) > 1 else np.zeros(0)
    pairwise = np.clip(kr @ kr.T, -1.0, 1.0)
    return {
        "endpoint_angle_deg": float(_angle_deg(kr[0], kr[-1])),
        "cumulative_angle_deg": float(increments.sum()),
        "maximum_pairwise_angle_deg": float(np.rad2deg(np.arccos(pairwise.min()))),
    }


def _projection_information(kr: np.ndarray, kd: np.ndarray) -> np.ndarray:
    # Every image constrains range and cross-range projections. Equal image
    # counts make the mean information matrix directly comparable.
    information = np.einsum("ni,nj->ij", kr, kr) + np.einsum("ni,nj->ij", kd, kd)
    return information / len(kr)


def _observability_metrics(information: np.ndarray) -> dict[str, float]:
    eigenvalues = np.maximum(np.linalg.eigvalsh(information), 0.0)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    trace = float(eigenvalues.sum())
    return {
        "normalized_minimum_eigenvalue": smallest / trace if trace > 0 else 0.0,
        "observability_isotropy": smallest / largest if largest > 0 else 0.0,
        "information_condition_number": largest / max(smallest, 1.0e-12),
    }


def _mean_direction(kr: np.ndarray) -> np.ndarray:
    vector = kr.mean(axis=0)
    if np.linalg.norm(vector) < 1.0e-12:
        vector = kr[0]
    return _normalize(vector[None])[0]


def _pairwise_center_metrics(selected: list[int], centers: np.ndarray) -> dict[str, float]:
    if len(selected) < 2:
        return {
            "minimum_center_separation_deg": 0.0,
            "mean_center_separation_deg": 0.0,
            "maximum_center_separation_deg": 0.0,
        }
    angles = [
        float(_angle_deg(centers[first], centers[second]))
        for first, second in combinations(selected, 2)
    ]
    return {
        "minimum_center_separation_deg": min(angles),
        "mean_center_separation_deg": float(np.mean(angles)),
        "maximum_center_separation_deg": max(angles),
    }


def _selection_state(
    selected: list[int],
    direction_masks: np.ndarray,
    information_matrices: np.ndarray,
    centers: np.ndarray,
    weights: dict[str, float],
) -> dict[str, float]:
    direction_fraction = float(direction_masks[selected].any(axis=0).mean())
    information = information_matrices[selected].mean(axis=0)
    observability = _observability_metrics(information)
    baseline = _pairwise_center_metrics(selected, centers)
    objective = (
        weights["direction"] * direction_fraction
        + weights["observability"] * observability["normalized_minimum_eigenvalue"]
        + weights["baseline"] * baseline["mean_center_separation_deg"] / 180.0
    )
    return {
        "direction_union_fraction": direction_fraction,
        **observability,
        **baseline,
        "objective": objective,
    }


def _best_fixed_k(
    count: int,
    names: list[str],
    direction_masks: np.ndarray,
    information_matrices: np.ndarray,
    centers: np.ndarray,
    weights: dict[str, float],
) -> tuple[tuple[int, ...], dict[str, float], int]:
    best_subset: tuple[int, ...] | None = None
    best_state: dict[str, float] | None = None
    evaluated = 0
    tolerance = 1.0e-15
    for subset in combinations(range(len(names)), count):
        state = _selection_state(
            list(subset), direction_masks, information_matrices, centers, weights
        )
        evaluated += 1
        subset_names = tuple(names[index] for index in subset)
        best_names = (
            tuple(names[index] for index in best_subset) if best_subset is not None else None
        )
        if (
            best_state is None
            or state["objective"] > best_state["objective"] + tolerance
            or (
                abs(state["objective"] - best_state["objective"]) <= tolerance
                and (best_names is None or subset_names < best_names)
            )
        ):
            best_subset = subset
            best_state = state
    if best_subset is None or best_state is None:
        raise RuntimeError(f"No subset generated for K={count}")
    return best_subset, best_state, evaluated


def _greedy_fixed_k_initialization(
    count: int,
    names: list[str],
    direction_masks: np.ndarray,
    information_matrices: np.ndarray,
    centers: np.ndarray,
    weights: dict[str, float],
) -> tuple[int, ...]:
    selected: list[int] = []
    remaining = set(range(len(names)))
    while len(selected) < count:
        candidates = []
        for candidate in remaining:
            state = _selection_state(
                selected + [candidate],
                direction_masks,
                information_matrices,
                centers,
                weights,
            )
            candidates.append((state["objective"], names[candidate], candidate))
        _, _, chosen = max(candidates, key=lambda item: (item[0], item[1]))
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(sorted(selected))


def _objective_from_swap_statistics(
    coverage_counts: np.ndarray,
    information_sum: np.ndarray,
    pairwise_angle_sum: float,
    count: int,
    weights: dict[str, float],
) -> float:
    direction_fraction = float(np.count_nonzero(coverage_counts)) / len(coverage_counts)
    observability = _observability_metrics(information_sum / count)
    mean_separation = (
        pairwise_angle_sum / math.comb(count, 2) if count >= 2 else 0.0
    )
    return (
        weights["direction"] * direction_fraction
        + weights["observability"] * observability["normalized_minimum_eigenvalue"]
        + weights["baseline"] * mean_separation / 180.0
    )


def _local_search_fixed_k(
    initial: tuple[int, ...],
    direction_masks: np.ndarray,
    information_matrices: np.ndarray,
    center_angles: np.ndarray,
    weights: dict[str, float],
    max_passes: int,
) -> tuple[tuple[int, ...], float, int]:
    selected = set(initial)
    coverage_counts = direction_masks[list(selected)].sum(axis=0, dtype=np.int16)
    information_sum = information_matrices[list(selected)].sum(axis=0)
    pairwise_sum = sum(
        center_angles[first, second] for first, second in combinations(selected, 2)
    )
    objective = _objective_from_swap_statistics(
        coverage_counts, information_sum, pairwise_sum, len(selected), weights
    )
    evaluated = 1
    tolerance = 1.0e-15
    for _ in range(max_passes):
        best_move = None
        best_objective = objective
        outside = sorted(set(range(len(direction_masks))) - selected)
        for removed in sorted(selected):
            retained = selected - {removed}
            removed_pair_sum = sum(center_angles[removed, item] for item in retained)
            base_pair_sum = pairwise_sum - removed_pair_sum
            base_coverage = coverage_counts - direction_masks[removed]
            base_information = information_sum - information_matrices[removed]
            for added in outside:
                trial_coverage = base_coverage + direction_masks[added]
                trial_information = base_information + information_matrices[added]
                trial_pair_sum = base_pair_sum + sum(
                    center_angles[added, item] for item in retained
                )
                trial_objective = _objective_from_swap_statistics(
                    trial_coverage,
                    trial_information,
                    trial_pair_sum,
                    len(selected),
                    weights,
                )
                evaluated += 1
                move = (removed, added)
                if (
                    trial_objective > best_objective + tolerance
                    or (
                        abs(trial_objective - best_objective) <= tolerance
                        and best_move is not None
                        and move < best_move
                    )
                ):
                    best_objective = trial_objective
                    best_move = move
        if best_move is None or best_objective <= objective + tolerance:
            break
        removed, added = best_move
        retained = selected - {removed}
        pairwise_sum = (
            pairwise_sum
            - sum(center_angles[removed, item] for item in retained)
            + sum(center_angles[added, item] for item in retained)
        )
        coverage_counts = coverage_counts - direction_masks[removed] + direction_masks[added]
        information_sum = (
            information_sum - information_matrices[removed] + information_matrices[added]
        )
        selected.remove(removed)
        selected.add(added)
        objective = best_objective
    return tuple(sorted(selected)), objective, evaluated


def _best_fixed_k_multistart(
    count: int,
    names: list[str],
    direction_masks: np.ndarray,
    information_matrices: np.ndarray,
    centers: np.ndarray,
    weights: dict[str, float],
    restarts: int,
    max_passes: int,
    seed: int,
) -> tuple[tuple[int, ...], dict[str, float], int]:
    if restarts < 1:
        raise ValueError("search_restarts must be at least 1")
    center_angles = np.rad2deg(
        np.arccos(np.clip(centers @ centers.T, -1.0, 1.0))
    )
    generator = np.random.default_rng(seed + 1009 * count)
    initial_sets = [
        _greedy_fixed_k_initialization(
            count, names, direction_masks, information_matrices, centers, weights
        )
    ]
    seen = {initial_sets[0]}
    attempts = 0
    while len(initial_sets) < restarts and attempts < restarts * 20:
        candidate = tuple(sorted(generator.choice(len(names), count, replace=False).tolist()))
        attempts += 1
        if candidate not in seen:
            seen.add(candidate)
            initial_sets.append(candidate)

    best_subset = None
    best_state = None
    evaluated = 0
    for initial in initial_sets:
        subset, _, local_evaluated = _local_search_fixed_k(
            initial,
            direction_masks,
            information_matrices,
            center_angles,
            weights,
            max_passes,
        )
        evaluated += local_evaluated
        state = _selection_state(
            list(subset), direction_masks, information_matrices, centers, weights
        )
        subset_names = tuple(names[index] for index in subset)
        best_names = (
            tuple(names[index] for index in best_subset) if best_subset is not None else None
        )
        if (
            best_state is None
            or state["objective"] > best_state["objective"] + 1.0e-15
            or (
                abs(state["objective"] - best_state["objective"]) <= 1.0e-15
                and (best_names is None or subset_names < best_names)
            )
        ):
            best_subset = subset
            best_state = state
    if best_subset is None or best_state is None:
        raise RuntimeError(f"Approximate search failed for K={count}")
    return best_subset, best_state, evaluated


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _balanced_eval_frames(heldout: dict[str, list[int]], maximum: int) -> list[dict[str, int | str]]:
    names = sorted(heldout)
    total = sum(len(values) for values in heldout.values())
    if maximum <= 0 or total <= maximum:
        return [
            {"encounter": name, "frame_index": index}
            for name in names
            for index in heldout[name]
        ]
    base, remainder = divmod(maximum, len(names))
    selected = []
    for position, name in enumerate(names):
        values = heldout[name]
        quota = min(len(values), base + (1 if position < remainder else 0))
        if quota:
            chosen = np.linspace(0, len(values) - 1, quota).round().astype(int)
            selected.extend(
                {"encounter": name, "frame_index": values[int(item)]} for item in chosen
            )
    return selected


def _save_plots(output_dir: Path, individual: list[dict], fixed_k: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV and JSON outputs were still generated")
        return
    ordered = sorted(individual, key=lambda row: row["direction_coverage_pct"], reverse=True)
    figure, axis = plt.subplots(figsize=(max(8, len(ordered) * 0.55), 4.5))
    axis.bar(
        [row["encounter"] for row in ordered],
        [row["direction_coverage_pct"] for row in ordered],
    )
    axis.set_ylabel("observation-direction coverage (%)")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "individual_direction_coverage.png", dpi=180)
    plt.close(figure)

    figure, left = plt.subplots(figsize=(7, 4.5))
    counts = [row["encounter_count"] for row in fixed_k]
    left.plot(
        counts,
        [row["direction_union_pct"] for row in fixed_k],
        marker="o",
        label="direction coverage",
    )
    left.set_xlabel("number of selected encounters")
    left.set_ylabel("observation-direction coverage (%)")
    left.grid(alpha=0.25)
    right = left.twinx()
    right.plot(
        counts,
        [row["normalized_minimum_eigenvalue"] for row in fixed_k],
        color="tab:orange",
        marker="s",
        label="weakest-direction observability",
    )
    right.set_ylabel("normalized minimum eigenvalue")
    figure.tight_layout()
    figure.savefig(output_dir / "fixed_k_geometry_curve.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encounters = sorted(
        [
            path
            for path in data_root.iterdir()
            if path.is_dir() and (path / "isar_images_noaxis").is_dir()
        ],
        key=_numeric_key,
    )
    if not encounters:
        raise FileNotFoundError(f"No encounter directories found under {data_root}")

    sphere = _fibonacci_sphere(args.sphere_samples)
    cap_cosine = math.cos(math.radians(args.direction_cap_deg))
    names: list[str] = []
    selected_frames: dict[str, list[int]] = {}
    heldout_frames: dict[str, list[int]] = {}
    kr_sets: list[np.ndarray] = []
    kd_sets: list[np.ndarray] = []
    available_counts: list[int] = []

    for encounter in encounters:
        angles = np.asarray(np.load(_view_angle_path(encounter)), dtype=np.float64)
        if angles.ndim != 2 or angles.shape[1] < 6:
            raise ValueError(f"Expected [N,6+] view_angle.npy for {encounter}")
        stop = min(len(angles), args.frame_start + args.frames_per_encounter)
        available = list(range(args.frame_start, stop))
        chosen = _fixed_indices(available, args.images_per_encounter)
        kr = _normalize(angles[chosen, :3])
        kd_raw = angles[chosen, 3:6]
        kd = _normalize(kd_raw - np.sum(kd_raw * kr, axis=1, keepdims=True) * kr)
        names.append(encounter.name)
        selected_frames[encounter.name] = chosen
        heldout_frames[encounter.name] = sorted(set(available) - set(chosen))
        kr_sets.append(kr)
        kd_sets.append(kd)
        available_counts.append(len(available))

    if args.strict_equal_available_frames and len(set(available_counts)) != 1:
        raise ValueError(
            "Available frame counts differ between encounters. Fix the dataset/range or pass "
            "--allow_unequal_available_frames; selected training counts remain equal either way."
        )

    direction_masks = np.asarray(
        [(sphere @ kr.T >= cap_cosine).any(axis=1) for kr in kr_sets], dtype=bool
    )
    information_matrices = np.asarray(
        [_projection_information(kr, kd) for kr, kd in zip(kr_sets, kd_sets)]
    )
    centers = np.asarray([_mean_direction(kr) for kr in kr_sets])

    individual = []
    for index, name in enumerate(names):
        observability = _observability_metrics(information_matrices[index])
        individual.append(
            {
                "encounter": name,
                "available_frames": available_counts[index],
                "selected_images": args.images_per_encounter,
                "direction_coverage_pct": 100.0 * float(direction_masks[index].mean()),
                **observability,
                **_aperture_metrics(kr_sets[index]),
            }
        )
    _write_csv(output_dir / "individual_encounter_geometry.csv", individual)

    weights = {
        "direction": args.direction_weight,
        "observability": args.observability_weight,
        "baseline": args.baseline_weight,
    }
    counts = sorted(set(args.counts or range(1, len(names) + 1)))
    if not counts or counts[0] < 1 or counts[-1] > len(names):
        raise ValueError(f"Counts must be between 1 and {len(names)}")

    subsets_dir = output_dir / "subsets"
    subsets_dir.mkdir(exist_ok=True)
    fixed_k = []
    subsets: dict[str, list[str]] = {}
    for count in counts:
        combination_count = math.comb(len(names), count)
        if combination_count <= args.max_exact_combinations:
            subset, state, evaluated = _best_fixed_k(
                count, names, direction_masks, information_matrices, centers, weights
            )
            search_method = "exact_exhaustive"
        else:
            print(
                f"K={count}: {combination_count:,} possible subsets; using independent "
                f"{args.search_restarts}-restart 1-swap local search",
                flush=True,
            )
            subset, state, evaluated = _best_fixed_k_multistart(
                count,
                names,
                direction_masks,
                information_matrices,
                centers,
                weights,
                args.search_restarts,
                args.search_max_passes,
                args.seed,
            )
            search_method = "multi_start_1swap_local_search"
        subset_names = [names[index] for index in subset]
        subsets[str(count)] = subset_names
        (subsets_dir / f"k{count:02d}.txt").write_text(
            "\n".join(subset_names) + "\n", encoding="utf-8"
        )
        fixed_k.append(
            {
                "encounter_count": count,
                "encounters": ";".join(subset_names),
                "search_method": search_method,
                "total_possible_combinations": combination_count,
                "evaluated_candidates": evaluated,
                "objective": state["objective"],
                "direction_union_pct": 100.0 * state["direction_union_fraction"],
                "normalized_minimum_eigenvalue": state["normalized_minimum_eigenvalue"],
                "observability_isotropy": state["observability_isotropy"],
                "information_condition_number": state["information_condition_number"],
                "minimum_center_separation_deg": state["minimum_center_separation_deg"],
                "mean_center_separation_deg": state["mean_center_separation_deg"],
                "maximum_center_separation_deg": state["maximum_center_separation_deg"],
            }
        )
        print(f"K={count}: {', '.join(subset_names)} (objective={state['objective']:.6f})")
    _write_csv(output_dir / "fixed_k_selection.csv", fixed_k)

    eval_views = _balanced_eval_frames(heldout_frames, args.max_eval_views)
    (output_dir / "evaluation_views.json").write_text(
        json.dumps(
            {
                "split": "heldout" if eval_views else "all",
                "views": eval_views,
                "note": (
                    "A fixed held-out list shared by all K runs."
                    if eval_views
                    else "No held-out frames remain because all available images are used."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "method": "geometry_only_fixed_aperture_independent_exact_k_selection",
        "selection_uses_target_structure": False,
        "selection_uses_isar_intensity": False,
        "selection_inputs": ["view_angle.npy k_r", "view_angle.npy k_d"],
        "interpretation": (
            "direction_union_pct is observation-direction coverage, not unknown object-surface "
            "coverage. Observability is derived from the range/cross-range projection information."
        ),
        "data_root": str(data_root),
        "encounter_count": len(names),
        "images_per_encounter": args.images_per_encounter,
        "available_frames_per_encounter": dict(zip(names, available_counts)),
        "selected_training_frames": selected_frames,
        "direction_cap_deg": args.direction_cap_deg,
        "sphere_samples": args.sphere_samples,
        "weights": weights,
        "max_exact_combinations": args.max_exact_combinations,
        "search_restarts": args.search_restarts,
        "search_max_passes": args.search_max_passes,
        "seed": args.seed,
        "optimized_counts": counts,
        "subsets_are_nested": False,
        "optimal_subsets_by_k": subsets,
    }
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _save_plots(output_dir, individual, fixed_k)
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
