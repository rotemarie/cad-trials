from cad_trials.common.execute import execute_program, valid_geometry

GOOD = "import cadquery as cq\nr = cq.Workplane('XY').box(2, 2, 2)"
NO_R = "import cadquery as cq\nx = cq.Workplane('XY').box(1, 1, 1)"
BROKEN = "import cadquery as cq\nr = cq.Workplane('XY').box(2, 2)"   # missing arg
HANG = "import cadquery as cq\nwhile True:\n    pass\nr = None"


def test_valid_program_executes(tmp_path):
    res = execute_program(GOOD, tmp_path / "a.stl")
    assert res.ok
    assert (tmp_path / "a.stl").exists()
    assert res.volume ==  __import__("pytest").approx(8.0, rel=0.05)
    assert valid_geometry(res)
    # process=True + merge_vertices(): a closed box must come back watertight, so
    # the recorded `watertight` flag (and the volume behind it) is meaningful.
    assert res.watertight is True
    assert res.n_vertices == 8


def test_result_variable_fallback(tmp_path):
    res = execute_program(NO_R, tmp_path / "b.stl")
    assert res.ok and res.volume == __import__("pytest").approx(1.0, rel=0.05)


def test_broken_program_reports_error(tmp_path):
    res = execute_program(BROKEN, tmp_path / "c.stl")
    assert not res.ok and res.error
    assert not (tmp_path / "c.stl").exists()
    assert not valid_geometry(res)


def test_timeout(tmp_path):
    res = execute_program(HANG, tmp_path / "d.stl", timeout=3)
    assert not res.ok and "timeout" in res.error.lower()
