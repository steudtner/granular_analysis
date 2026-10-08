import importlib
import os

import pytest


@pytest.mark.parametrize(
    "options, expected",
    [
        (["--counter", "1682"], "1682"),
        (["--counter", "0"], "0"),
        (["--histogram-timesteps", "0,100,-1"], "0,100,-1"),
        (["--counter", "1682", "--histogram-timesteps", "10,20"], "10,20"),
        ([], None),
    ],
)
def test_energy_dispatches_histogram_selection(monkeypatch, tmp_path, options, expected):
    import granular_analysis.scripts.run_all as run_all
    calls = {}
    monkeypatch.setattr(run_all, "run_energy", lambda **kwargs: calls.update(kwargs))
    result = run_all.main([
        "--only", "energy", "--path", str(tmp_path),
        "--out", str(tmp_path / "out"), *options,
    ])
    assert result == 0
    assert calls["histogram_timesteps"] == expected


def test_run_all_prefers_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRANULAR_DATA_FOLDER", raising=False)
    monkeypatch.delenv("GRANULAR_RESULTS_DIR", raising=False)
    import granular_analysis.config as config
    import granular_analysis.scripts.run_all as run_all
    importlib.reload(config)
    importlib.reload(run_all)
    assert run_all._resolve_data_folder() == str(
        tmp_path / "results" / "ellipsoid" / "ellipsoid_trn"
    )
    assert config.RESULTS_DIR == str(tmp_path / "results_granular_analysis")


def test_dry_run_does_not_create_output(tmp_path):
    from granular_analysis.scripts.run_all import main
    output = tmp_path / "not-created"
    assert main(["--only", "2d", "--dry", "--path", str(tmp_path), "--out", str(output)]) == 0
    assert not output.exists()


def test_slices_dispatch_uses_supported_arguments(monkeypatch, tmp_path):
    import granular_analysis.scripts.run_all as run_all
    calls = {}
    monkeypatch.setattr(run_all, "latest_trn_counter", lambda case: 12)
    monkeypatch.setattr(run_all, "run_slices", lambda **kwargs: calls.update(kwargs))
    monkeypatch.setattr(run_all, "_run_exact_slice_plot", lambda *args: calls.update(exact=True))
    result = run_all.main([
        "--only", "slices", "--path", str(tmp_path), "--out", str(tmp_path / "out"),
        "--z", "2.0", "--alpha", "0.7", "--skip-points", "5",
    ])
    assert result == 0
    assert calls["counter"] == 12
    assert calls["z_value"] == 2.0
    assert calls["alpha"] == 0.7
    assert calls["skip_first"] == 5
    assert calls["exact"] is True


def test_pipeline_failure_returns_nonzero(monkeypatch, tmp_path):
    import granular_analysis.scripts.run_all as run_all
    monkeypatch.setattr(
        run_all,
        "run_2d",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = run_all.main([
        "--only", "2d", "--path", str(tmp_path), "--out", str(tmp_path / "out")
    ])
    assert result == 1


def test_alpha_sensitivity_dispatches_sweep_arguments(monkeypatch, tmp_path):
    import granular_analysis.scripts.run_all as run_all
    calls = {}
    monkeypatch.setattr(run_all, "run_sensitivity", lambda *args, **kwargs: calls.update(args=args, **kwargs))
    result = run_all.main([
        "--only", "alpha_sensitivity", "--path", str(tmp_path),
        "--out", str(tmp_path / "out"), "--z", "1.25", "--lz", "2.5",
        "--alpha-start", "0.1", "--alpha-stop", "2.0", "--alpha-step", "0.1",
    ])
    assert result == 0
    assert calls["z_value"] == 1.25
    assert calls["lz"] == 2.5
    assert calls["alpha_stop"] == 2.0
    assert calls["args"][1] == tmp_path / "out" / "alpha_sensitivity"


def test_slope_alpha_sensitivity_dispatches_sweep_arguments(monkeypatch, tmp_path):
    import granular_analysis.scripts.run_all as run_all
    calls = {}
    monkeypatch.setattr(
        run_all,
        "run_slope_alpha_sensitivity",
        lambda *args, **kwargs: calls.update(args=args, **kwargs),
    )
    result = run_all.main([
        "--only", "slope_alpha_sensitivity", "--path", str(tmp_path),
        "--out", str(tmp_path / "out"), "--z", "1.25", "--lz", "2.5",
        "--alpha-start", "0.1", "--alpha-stop", "2.0", "--alpha-step", "0.1",
    ])
    assert result == 0
    assert calls["z_value"] == 1.25
    assert calls["lz"] == 2.5
    assert calls["alpha_start"] == 0.1
    assert calls["args"][1] == tmp_path / "out" / "slope_alpha_sensitivity"


@pytest.mark.parametrize("counter", [0, 10])
def test_collision_dispatch_single_stage(monkeypatch, tmp_path, counter):
    import granular_analysis.scripts.run_all as run_all
    from granular_analysis.paths import CasePaths
    case = CasePaths.from_input(tmp_path)
    case.trn.mkdir(parents=True)
    case.coll.mkdir(parents=True)
    for value in (0, 10):
        (case.trn / f"ell_trn_{value:06d}.bin").touch()
        (case.coll / f"ell_coll_{value:06d}_1.bin").touch()
    calls = {}
    monkeypatch.setattr(run_all, "run_collision_distances", lambda **kw: calls.update(kw))
    assert run_all.main([
        "--only", "collision_distances", "--path", str(tmp_path),
        "--out", str(tmp_path / "out"), "--counter", str(counter),
    ]) == 0
    assert calls["counter"] == counter


def test_collision_analysis_detects_single_stage(monkeypatch, tmp_path):
    from granular_analysis.scripts import run_collision_distances as script
    from granular_analysis.paths import CasePaths
    case = CasePaths.from_input(tmp_path)
    case.coll.mkdir(parents=True)
    (case.coll / "ell_coll_000002_1.bin").touch()
    calls = []
    def load(**kwargs):
        calls.append(kwargs)
        return []
    monkeypatch.setattr(script, "load_data", load)
    script.main(counter=2, path=str(tmp_path), out=str(tmp_path / "out"))
    assert [call["data"] for call in calls] == ["trn", "coll"]
    assert all(call["dry"] is True for call in calls)


def test_renderview_dispatches_axis_limits(monkeypatch, tmp_path):
    from granular_analysis.scripts import run_all
    calls = {}
    monkeypatch.setattr(run_all, 'run_renderview', lambda **kwargs: calls.update(kwargs))
    assert run_all.main([
        '--only', 'renderview', '--path', str(tmp_path), '--counter', '12',
        '--out', str(tmp_path / 'out'), '--xlim', '5', '25', '--ylim', '2', '10',
    ]) == 0
    assert calls['xlim'] == (5, 25)
    assert calls['ylim'] == (2, 10)


def test_renderview_bounds_and_cache(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from granular_analysis.scripts import run_all
    from granular_analysis.paths import CasePaths

    calls = []
    def render(particles, **kwargs):
        calls.append(kwargs['domain_bounds'])
        kwargs['output_dir'].mkdir(parents=True, exist_ok=True)
        for view in kwargs['views']:
            for ext in ("png", "pdf", "svg"):
                (kwargs['output_dir'] / f"{kwargs['filename_prefix']}_{view}.{ext}").touch()

    monkeypatch.setitem(sys.modules, 'granular_analysis.plotting.particle_3d_plotter',
                        SimpleNamespace(render_particle_views=render))
    monkeypatch.setattr(run_all, '_load_particles', lambda *args: [])
    options = dict(case=CasePaths.from_input(tmp_path), counter=12, out_dir=tmp_path / 'out')
    run_all.run_renderview(**options, xlim=(5, 25), ylim=(2, 10))
    assert calls == [(5, 25, 2, 10, *run_all.RENDER_BOUNDS[4:])]
    run_all.run_renderview(**options, xlim=(5, 25), ylim=(2, 10))
    assert len(calls) == 1
    run_all.run_renderview(**options, xlim=(5, 30), ylim=(2, 10))
    assert len(calls) == 2
    run_all.run_renderview(**options, ylim=(2, 10))
    assert calls[-1][:4] == (*run_all.RENDER_BOUNDS[:2], 2, 10)
