from __future__ import annotations

import math
import hashlib
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from airspace_complexity.r03_online_complexity import (
    PilotReport,
    TerminalVolume,
    assign_airport_day_splits,
    assign_group_splits,
    build_scene_examples,
    extract_contiguous_reports,
    experiment_plan,
    empty_pair_aggregate,
    finite_or_none,
    filter_terminal_reports,
    latency_summary,
    normalize_scene_tensor,
    sparse_neighbor_indices,
    traditional_controls,
    select_pilot_sources,
)
from airspace_complexity.r03_experiment_contract import summarize_temporal_controls
from airspace_complexity.r03_runner import R03SourceRegistry, R03SourceSpec


SECOND = 1_000_000


def _reports() -> list[PilotReport]:
    reports: list[PilotReport] = []
    for second in range(0, 421, 10):
        for index, aircraft in enumerate(("A1", "A2", "A3")):
            reports.append(PilotReport(
                airport_id="KAGC",
                source_file_id="source-1",
                source_day="2026-01-01",
                aircraft_id=aircraft,
                timestamp_us=second * SECOND,
                lat_deg=40.35 + 0.01 * index + second * 0.000001,
                lon_deg=-79.93 + 0.01 * index,
                altitude_m=1_000.0 + 100.0 * index,
                speed_mps=80.0 + index,
                heading_rad=0.2 * index,
            ))
    return reports


def test_scene_uses_only_history_and_labels_use_only_future() -> None:
    examples = build_scene_examples(
        _reports(),
        history_seconds=120,
        step_seconds=10,
        horizons_seconds=(30, 120, 300),
        stale_seconds=30,
        min_aircraft=3,
        max_aircraft=100,
    )

    example = examples[0]
    assert len(example.history) == 12
    assert max(
        report.timestamp_us for step in example.history for report in step
    ) <= example.cutoff_us
    assert example.targets[120].observation_start_us > example.cutoff_us
    assert example.targets[120].observation_end_us == example.cutoff_us + 120 * SECOND


def test_terminal_filter_excludes_reports_outside_common_volume() -> None:
    volume = TerminalVolume(
        latitude_deg=40.3544376,
        longitude_deg=-79.9290467,
        elevation_m=381.5,
        radius_m=18_520.0,
        floor_offset_m=-150.0,
        ceiling_offset_m=1_524.0,
    )
    template = _reports()[0]
    reports = [
        replace(template, aircraft_id="inside", lat_deg=40.36, altitude_m=500.0),
        replace(template, aircraft_id="far", lat_deg=41.36, altitude_m=500.0),
        replace(template, aircraft_id="high", lat_deg=40.36, altitude_m=2_000.0),
        replace(template, aircraft_id="below", lat_deg=40.36, altitude_m=100.0),
    ]

    filtered, audit = filter_terminal_reports(reports, {"KAGC": volume})

    assert [report.aircraft_id for report in filtered] == ["inside"]
    assert audit["KAGC"] == {
        "input_reports": 4,
        "accepted_reports": 1,
        "outside_horizontal": 1,
        "outside_vertical": 2,
    }


def test_future_proxy_merges_asynchronous_continuous_interaction_into_one_episode() -> None:
    reports: list[PilotReport] = []
    for second in range(0, 161, 10):
        reports.extend((
            PilotReport(
                airport_id="KAGC", source_file_id="source-1",
                source_day="2026-01-01", aircraft_id="A1",
                timestamp_us=second * SECOND, lat_deg=40.35, lon_deg=-79.93,
                altitude_m=1_000.0, speed_mps=80.0, heading_rad=0.0,
            ),
            PilotReport(
                airport_id="KAGC", source_file_id="source-1",
                source_day="2026-01-01", aircraft_id="A2",
                timestamp_us=(second + 1) * SECOND, lat_deg=40.351,
                lon_deg=-79.93, altitude_m=1_050.0, speed_mps=80.0,
                heading_rad=0.0,
            ),
        ))

    example = build_scene_examples(
        reports,
        history_seconds=120,
        step_seconds=10,
        horizons_seconds=(30,),
        stale_seconds=30,
        min_aircraft=2,
        max_aircraft=100,
    )[0]

    assert example.targets[30].proximity_event_count == 1
    assert example.targets[30].min_horizontal_separation_m is not None


def _two_aircraft_proximity_reports(longitudes_by_second: dict[int, float]):
    reports = []
    for second in range(0, 151, 10):
        reports.extend((
            PilotReport(
                airport_id="KAGC", source_file_id="source-1",
                source_day="2026-01-01", aircraft_id="A1",
                timestamp_us=second * SECOND, lat_deg=40.35, lon_deg=-79.93,
                altitude_m=1_000.0, speed_mps=80.0, heading_rad=0.0,
            ),
            PilotReport(
                airport_id="KAGC", source_file_id="source-1",
                source_day="2026-01-01", aircraft_id="A2",
                timestamp_us=second * SECOND, lat_deg=40.35,
                lon_deg=longitudes_by_second.get(second, -79.929),
                altitude_m=1_050.0, speed_mps=80.0, heading_rad=0.0,
            ),
        ))
    return reports


def test_future_onset_excludes_pair_already_active_at_cutoff() -> None:
    example = build_scene_examples(
        _two_aircraft_proximity_reports({}),
        history_seconds=120, step_seconds=10, horizons_seconds=(30,),
        stale_seconds=30, min_aircraft=2, max_aircraft=100,
    )[0]

    target = example.targets[30]
    assert target.cutoff_active_pair_count == 1
    assert target.proximity_event_count == 1
    assert target.future_onset_proximity_event_count == 0


def test_future_onset_counts_clear_then_reentry_but_not_initial_persistence() -> None:
    reports = _two_aircraft_proximity_reports({
        130: -79.70,
        140: -79.929,
    })
    example = build_scene_examples(
        reports,
        history_seconds=120, step_seconds=10, horizons_seconds=(30,),
        stale_seconds=30, min_aircraft=2, max_aircraft=100,
    )[0]

    target = example.targets[30]
    assert target.cutoff_active_pair_count == 1
    assert target.proximity_event_count == 2
    assert target.future_onset_proximity_event_count == 1


def test_future_onset_counts_pair_that_first_becomes_close_after_cutoff() -> None:
    reports = _two_aircraft_proximity_reports({
        110: -79.70,
        120: -79.929,
    })
    example = build_scene_examples(
        reports,
        history_seconds=120, step_seconds=10, horizons_seconds=(30,),
        stale_seconds=30, min_aircraft=2, max_aircraft=100,
    )[0]

    target = example.targets[30]
    assert target.cutoff_active_pair_count == 0
    assert target.future_onset_proximity_event_count == 1


def test_future_onset_thresholds_are_explicit_and_configurable() -> None:
    example = build_scene_examples(
        _two_aircraft_proximity_reports({110: -79.70, 120: -79.929}),
        history_seconds=120, step_seconds=10, horizons_seconds=(30,),
        stale_seconds=30, min_aircraft=2, max_aircraft=100,
        horizontal_proximity_m=50.0, vertical_proximity_m=304.8,
    )[0]

    assert example.targets[30].future_onset_proximity_event_count == 0
    with pytest.raises(ValueError, match="proximity thresholds"):
        build_scene_examples(
            _two_aircraft_proximity_reports({}),
            history_seconds=120, step_seconds=10, horizons_seconds=(30,),
            stale_seconds=30, min_aircraft=2, max_aircraft=100,
            horizontal_proximity_m=0.0, vertical_proximity_m=304.8,
        )


def test_frozen_scene_splits_must_cover_exact_observed_airport_days() -> None:
    from scripts.run_r03_online_complexity_pilot import _validated_frozen_splits

    observed = {"KBTP\x002026-01-01", "KBTP\x002026-01-02"}
    frozen = {
        "KBTP\x002026-01-01": "train",
        "KBTP\x002026-01-02": "test",
    }

    assert _validated_frozen_splits(observed, frozen) == frozen
    with pytest.raises(ValueError, match="exactly cover"):
        _validated_frozen_splits(observed, {"KBTP\x002026-01-01": "train"})
    with pytest.raises(ValueError, match="invalid frozen split"):
        _validated_frozen_splits(
            observed,
            {"KBTP\x002026-01-01": "train", "KBTP\x002026-01-02": "holdout"},
        )


def test_scene_bytes_are_stable_under_input_permutation() -> None:
    kwargs = dict(
        history_seconds=120,
        step_seconds=10,
        horizons_seconds=(30,),
        stale_seconds=30,
        min_aircraft=3,
        max_aircraft=100,
    )
    forward = build_scene_examples(_reports(), **kwargs)
    reverse = build_scene_examples(reversed(_reports()), **kwargs)
    assert forward == reverse


def test_scene_combines_aircraft_from_multiple_sources_on_same_airport_day() -> None:
    reports = [
        replace(report, source_file_id=f"source-{report.aircraft_id}")
        for report in _reports()
    ]
    examples = build_scene_examples(
        reports,
        history_seconds=120,
        step_seconds=10,
        horizons_seconds=(30,),
        stale_seconds=30,
        min_aircraft=3,
        max_aircraft=100,
    )
    assert examples
    assert set(examples[0].source_file_ids) == {"source-A1", "source-A2", "source-A3"}


def test_stale_reports_are_not_carried_into_scene() -> None:
    reports = _reports()
    reports = [
        report for report in reports
        if not (report.aircraft_id == "A3" and 90 < report.timestamp_us / SECOND < 120)
    ]
    examples = build_scene_examples(
        reports,
        history_seconds=120,
        step_seconds=10,
        horizons_seconds=(30,),
        stale_seconds=15,
        min_aircraft=2,
        max_aircraft=100,
    )
    cutoff_110 = next(example for example in examples if example.cutoff_us == 110 * SECOND)
    assert "A3" not in {row.aircraft_id for row in cutoff_110.history[-1]}


def test_future_separation_is_missing_when_no_aircraft_pair_is_observed() -> None:
    reports = [
        report for report in _reports()
        if report.aircraft_id == "A1" or report.timestamp_us <= 110 * SECOND
    ]
    example = build_scene_examples(
        reports,
        history_seconds=120,
        step_seconds=10,
        horizons_seconds=(30,),
        stale_seconds=30,
        min_aircraft=3,
        max_aircraft=100,
    )[0]
    assert example.targets[30].min_horizontal_separation_m is None
    assert example.targets[30].min_vertical_separation_m is None


def test_group_splits_are_deterministic_and_disjoint() -> None:
    groups = [f"KAGC\x00source-{index}\x002026-01-{index:02d}" for index in range(1, 21)]
    first = assign_group_splits(groups)
    second = assign_group_splits(reversed(groups))
    assert first == second
    assert set(first) == set(groups)
    for split in ("train", "validation", "test"):
        assert any(value == split for value in first.values())


def test_airport_day_splits_reserve_validation_and_test_day_per_airport() -> None:
    groups = [
        f"{airport}\x002026-01-{day:02d}"
        for airport in ("KAGC", "KBTP")
        for day in range(1, 5)
    ]

    first = assign_airport_day_splits(groups)
    second = assign_airport_day_splits(reversed(groups))

    assert first == second
    for airport in ("KAGC", "KBTP"):
        airport_splits = {
            split for group, split in first.items()
            if group.startswith(f"{airport}\x00")
        }
        assert airport_splits == {"train", "validation", "test"}


def test_temporal_controls_capture_level_variation_and_trend() -> None:
    summary = summarize_temporal_controls(
        counts=(1.0, 2.0, 3.0),
        min_distances_m=(300.0, 200.0, 100.0),
        altitude_stds_m=(10.0, 20.0, 30.0),
        speed_stds_mps=(2.0, 4.0, 6.0),
    )

    assert summary == pytest.approx((
        2.0, (2.0 / 3.0) ** 0.5, 1.0,
        200.0, -100.0, 20.0, 4.0, 1.0,
    ))


def test_sparse_neighbors_are_bounded_and_exclude_self() -> None:
    rows = _reports()[:3]
    neighbors = sparse_neighbor_indices(rows, top_k=1)
    assert set(neighbors) == {"A1", "A2", "A3"}
    assert all(len(value) == 1 for value in neighbors.values())
    assert all(aircraft not in value for aircraft, value in neighbors.items())


def test_controls_and_latency_are_finite() -> None:
    example = build_scene_examples(
        _reports(),
        history_seconds=120,
        step_seconds=10,
        horizons_seconds=(30,),
        stale_seconds=30,
        min_aircraft=3,
        max_aircraft=100,
    )[0]
    controls = traditional_controls(example)
    assert controls["aircraft_count"] == 3.0
    assert all(math.isfinite(value) for value in controls.values())

    summary = latency_summary([1_000_000, 2_000_000, 3_000_000, 4_000_000])
    assert summary["iterations"] == 4
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(3.85)


def test_nonfinite_report_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        PilotReport(**{
            **_reports()[0].__dict__,
            "lat_deg": float("nan"),
        })


def _registry_source(
    root: Path, airport: str, source_index: int, *, drop_terminal: bool = False,
    duplicate_first: bool = False, late_dense: bool = False,
) -> R03SourceSpec:
    relative = f"raw/{source_index}.csv"
    path = root / airport.lower() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ID,Time,Date,Altitude,Speed,Heading,Lat,Lon,Age,Range,Bearing,Tail"
    ]
    final_second = 7200 if late_dense else 420
    for second in range(0, final_second + 1, 10):
        hour, remainder = divmod(second, 3600)
        minute, second_value = divmod(remainder, 60)
        aircraft_values = (
            ("A1", "A2", "A3") if not late_dense or second >= 3600 else ("A1",)
        )
        for aircraft_index, aircraft in enumerate(aircraft_values):
            lines.append(
                f"{aircraft},{hour:02d}:{minute:02d}:{second_value:02d},2026-01-01,"
                f"{3000 + 100 * aircraft_index},100,90,"
                f"{40.35 + 0.01 * aircraft_index},"
                f"{-79.93 + 0.01 * aircraft_index},1,1,90,N{aircraft_index}"
            )
        if second == 0 and duplicate_first:
            lines.append(
                "A1,00:00:00,2026-01-01,9999,100,90,40.35,-79.93,2,1,90,N0"
            )
    payload = ("\r\n".join(lines) + "\r\n").encode("utf-8")
    if drop_terminal:
        payload += b'BROKEN,"unterminated'
    path.write_bytes(payload)
    return R03SourceSpec(
        airport_id=airport,
        source_file_id=f"{airport}-source-{source_index}",
        relative_path=relative,
        r01_disposition=(
            "drop_terminal_record_in_R02" if drop_terminal else "read_extracted"
        ),
        source_sequence=source_index,
        extracted_sha256=hashlib.sha256(payload).hexdigest(),
        archive_relative_path=None,
        archive_sha256=None,
        archive_member=None,
        member_crc32=None,
        member_size_bytes=None,
        authenticated_row_count=len(lines) - 1,
    )


def test_extractor_reads_only_selected_sources_and_preserves_continuity(
    tmp_path: Path,
) -> None:
    specs = tuple(
        _registry_source(tmp_path, airport, source_index)
        for airport in ("KAGC", "KBTP")
        for source_index in (1, 2)
    )
    registry = R03SourceRegistry(
        schema_version="r03_source_registry_v1",
        registry_id="1" * 64,
        sources=specs,
        raw_manifest_sha256="2" * 64,
        archive_comparison_sha256="3" * 64,
        r02_run_manifest_sha256="4" * 64,
        r02_decision_sha256="5" * 64,
        partition_manifest_sha256_by_airport=(("KAGC", "6" * 64), ("KBTP", "7" * 64)),
        source_sequence_manifest_sha256="8" * 64,
    )

    result = extract_contiguous_reports(
        registry,
        {"KAGC": tmp_path / "kagc", "KBTP": tmp_path / "kbtp"},
        Path(__file__).resolve().parents[1],
        sources_per_airport=1,
        max_hours_per_source=1,
    )

    assert set(result.opened_source_ids) == set(result.manifest["selected_source_ids"])
    assert len(result.opened_source_ids) == 2
    by_source: dict[str, list[PilotReport]] = {}
    for report in result.reports:
        by_source.setdefault(report.source_file_id, []).append(report)
    assert all(
        all(right.timestamp_us >= left.timestamp_us for left, right in zip(rows, rows[1:]))
        for rows in by_source.values()
    )
    assert {report.airport_id for report in result.reports} == {"KAGC", "KBTP"}


def test_source_selection_prefers_larger_authenticated_contiguous_sources(
    tmp_path: Path,
) -> None:
    small = _registry_source(tmp_path, "KAGC", 1)
    large = R03SourceSpec(**{
        **_registry_source(tmp_path, "KAGC", 2).__dict__,
        "authenticated_row_count": small.authenticated_row_count + 10_000,
    })
    registry = R03SourceRegistry(
        schema_version="r03_source_registry_v1",
        registry_id="1" * 64,
        sources=(small, large),
        raw_manifest_sha256="2" * 64,
        archive_comparison_sha256="3" * 64,
        r02_run_manifest_sha256="4" * 64,
        r02_decision_sha256="5" * 64,
        partition_manifest_sha256_by_airport=(("KAGC", "6" * 64),),
        source_sequence_manifest_sha256="8" * 64,
    )

    selected = select_pilot_sources(registry, sources_per_airport=1)

    assert selected == (large,)


def test_source_selection_honors_exact_frozen_source_ids(tmp_path: Path) -> None:
    first = _registry_source(tmp_path, "KAGC", 1)
    second = _registry_source(tmp_path, "KAGC", 2)
    registry = R03SourceRegistry(
        schema_version="r03_source_registry_v1",
        registry_id="1" * 64,
        sources=(first, second),
        raw_manifest_sha256="2" * 64,
        archive_comparison_sha256="3" * 64,
        r02_run_manifest_sha256="4" * 64,
        r02_decision_sha256="5" * 64,
        partition_manifest_sha256_by_airport=(("KAGC", "6" * 64),),
        source_sequence_manifest_sha256="8" * 64,
    )

    selected = select_pilot_sources(
        registry,
        sources_per_airport=1,
        exact_source_ids=(first.source_file_id,),
    )

    assert selected == (first,)


def test_scene_normalization_accepts_frozen_training_statistics() -> None:
    np = pytest.importorskip("numpy")
    values = np.zeros((2, 1, 1, 7), dtype=np.float32)
    values[:, :, :, 6] = 1.0
    values[0, 0, 0, :6] = np.arange(6, dtype=np.float32)
    values[1, 0, 0, :6] = np.arange(6, dtype=np.float32) + 2.0
    mean = np.ones(6, dtype=np.float32)
    scale = np.full(6, 2.0, dtype=np.float32)

    normalized, actual_mean, actual_scale = normalize_scene_tensor(
        values,
        np.array(["external_test", "external_test"]),
        frozen_normalization=(mean, scale),
    )

    assert np.array_equal(actual_mean, mean)
    assert np.array_equal(actual_scale, scale)
    assert normalized[0, 0, 0, 0] == pytest.approx(-0.5)
    assert normalized[1, 0, 0, 5] == pytest.approx(3.0)


def test_extractor_honors_authenticated_drop_terminal_semantics(
    tmp_path: Path,
) -> None:
    spec = _registry_source(tmp_path, "KAGC", 1, drop_terminal=True)
    registry = R03SourceRegistry(
        schema_version="r03_source_registry_v1",
        registry_id="1" * 64,
        sources=(spec,),
        raw_manifest_sha256="2" * 64,
        archive_comparison_sha256="3" * 64,
        r02_run_manifest_sha256="4" * 64,
        r02_decision_sha256="5" * 64,
        partition_manifest_sha256_by_airport=(("KAGC", "6" * 64),),
        source_sequence_manifest_sha256="8" * 64,
    )

    result = extract_contiguous_reports(
        registry,
        {"KAGC": tmp_path / "kagc"},
        Path(__file__).resolve().parents[1],
        sources_per_airport=1,
        max_hours_per_source=1,
    )

    assert len(result.reports) == spec.authenticated_row_count


def test_extractor_resolves_duplicate_natural_keys_before_scene_build(
    tmp_path: Path,
) -> None:
    spec = _registry_source(tmp_path, "KAGC", 1, duplicate_first=True)
    registry = R03SourceRegistry(
        schema_version="r03_source_registry_v1",
        registry_id="1" * 64,
        sources=(spec,),
        raw_manifest_sha256="2" * 64,
        archive_comparison_sha256="3" * 64,
        r02_run_manifest_sha256="4" * 64,
        r02_decision_sha256="5" * 64,
        partition_manifest_sha256_by_airport=(("KAGC", "6" * 64),),
        source_sequence_manifest_sha256="8" * 64,
    )

    result = extract_contiguous_reports(
        registry,
        {"KAGC": tmp_path / "kagc"},
        Path(__file__).resolve().parents[1],
        sources_per_airport=1,
        max_hours_per_source=1,
    )

    keys = [(row.aircraft_id, row.timestamp_us) for row in result.reports]
    assert len(keys) == len(set(keys))
    first_timestamp = min(row.timestamp_us for row in result.reports)
    selected = next(
        row for row in result.reports
        if row.aircraft_id == "A1" and row.timestamp_us == first_timestamp
    )
    assert selected.altitude_m != pytest.approx(9999 * 0.3048)


def test_extractor_selects_densest_contiguous_interval_without_future_labels(
    tmp_path: Path,
) -> None:
    spec = _registry_source(tmp_path, "KAGC", 1, late_dense=True)
    registry = R03SourceRegistry(
        schema_version="r03_source_registry_v1",
        registry_id="1" * 64,
        sources=(spec,),
        raw_manifest_sha256="2" * 64,
        archive_comparison_sha256="3" * 64,
        r02_run_manifest_sha256="4" * 64,
        r02_decision_sha256="5" * 64,
        partition_manifest_sha256_by_airport=(("KAGC", "6" * 64),),
        source_sequence_manifest_sha256="8" * 64,
    )

    result = extract_contiguous_reports(
        registry,
        {"KAGC": tmp_path / "kagc"},
        Path(__file__).resolve().parents[1],
        sources_per_airport=1,
        max_hours_per_source=1,
    )

    assert {row.aircraft_id for row in result.reports} == {"A1", "A2", "A3"}
    assert (
        min(row.timestamp_us for row in result.reports)
        % (24 * 3600 * SECOND)
    ) >= 3600 * SECOND


def test_online_pilot_cli_exposes_only_bounded_research_controls() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_r03_online_complexity_pilot.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--sources-per-airport" in completed.stdout
    assert "--max-hours-per-source" in completed.stdout
    assert "--prepare-only" in completed.stdout
    assert "--run-experiment" in completed.stdout
    assert "--models" in completed.stdout
    assert "--result-name" in completed.stdout
    assert "--raw-root" not in completed.stdout
    assert "--publish" not in completed.stdout


def test_methods_share_split_features_targets_and_latency_protocol() -> None:
    plan = experiment_plan(
        split_sha256="a" * 64,
        target_sha256="b" * 64,
        seeds=(1701, 1702, 1703),
    )
    assert {run.split_sha256 for run in plan} == {"a" * 64}
    assert {run.target_sha256 for run in plan} == {"b" * 64}
    assert all(
        run.warmup_iterations == 10 and run.timed_iterations == 100
        for run in plan
    )
    assert {run.seed for run in plan} == {1701, 1702, 1703}
    assert {run.model for run in plan} == {
        "deepsets", "masked_temporal", "invariant_relation",
        "calibrated_invariant_relation", "calibrated_relation_no_adversary",
        "calibrated_invariant_no_relation",
        "multitask_invariant_relation",
        "airport_conditioned_multitask",
        "masked_relation_ssl",
    }
    relation_runs = [run for run in plan if run.model in {
        "invariant_relation", "calibrated_invariant_relation",
        "calibrated_invariant_no_relation",
    }]
    assert all(run.airport_adversarial_weight == pytest.approx(0.1) for run in relation_runs)
    assert all(run.airport_adversarial_weight == 0.0 for run in plan if run not in relation_runs)
    calibrated = [run for run in plan if run.model.startswith("calibrated_") or run.model == "airport_conditioned_multitask"]
    assert all(run.uses_site_calibration for run in calibrated)
    assert all(not run.uses_site_calibration for run in plan if run not in calibrated)
    assert all(
        run.uses_pair_interactions
        for run in plan if "relation" in run.model and "no_relation" not in run.model
    )
    multitask = [run for run in plan if run.model in {
        "multitask_invariant_relation", "airport_conditioned_multitask"
    }]
    assert {run.target_horizons_seconds for run in multitask} == {(30, 120, 300)}
    assert all(
        run.target_horizons_seconds == (120,)
        for run in plan if run.model not in {
            "multitask_invariant_relation", "airport_conditioned_multitask",
            "masked_relation_ssl"
        }
    )
    ssl_runs = [run for run in plan if run.model == "masked_relation_ssl"]
    assert {run.training_signal for run in ssl_runs} == {"masked_self_supervision"}
    assert {run.target_horizons_seconds for run in ssl_runs} == {()}
    assert {run.probe_horizons_seconds for run in ssl_runs} == {(30, 120, 300)}


def test_empty_temporal_pair_set_has_neutral_aggregate() -> None:
    assert empty_pair_aggregate(0) == 0.0
    assert empty_pair_aggregate(1) == 0.0
    assert empty_pair_aggregate(2) is None


def test_nonfinite_research_metric_is_explicitly_missing() -> None:
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(1.25) == pytest.approx(1.25)
