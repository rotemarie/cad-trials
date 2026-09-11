import numpy as np
import trimesh
import pytest
from PIL import Image
from cad_trials.common.render import (
    render_style, four_diagonal_tile, ortho_three_view, STYLES)


@pytest.fixture
def part():
    m = trimesh.creation.box((4, 2, 1))
    m.apply_translation((0, 0, 0.5))
    return m


@pytest.mark.parametrize("style", STYLES)
def test_render_style_returns_image(part, style):
    im = render_style(part, style, size=128)
    assert isinstance(im, Image.Image)
    assert im.size == (128, 128)
    # not blank: some non-background pixels
    arr = np.asarray(im.convert("L"))
    assert arr.std() > 3


def test_four_diagonal_tile_shape(part):
    im = four_diagonal_tile(part, size=64)
    # 2x2 of (64 + 6 border) -> 140
    assert im.size == (140, 140)


def test_ortho_three_view(part):
    im = ortho_three_view(part, size=100)
    assert im.size[0] == 300 and im.size[1] == 100
    assert np.asarray(im.convert("L")).min() < 128   # has dark line pixels


def test_render_style_rejects_unknown(part):
    with pytest.raises(ValueError):
        render_style(part, "nope")
