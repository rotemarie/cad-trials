import trimesh
import pytest


@pytest.fixture
def unit_cube_mesh():
    return trimesh.creation.box((1, 1, 1))


@pytest.fixture
def shifted_cube_mesh():
    m = trimesh.creation.box((1, 1, 1))
    m.apply_translation((0.5, 0.0, 0.0))
    return m
