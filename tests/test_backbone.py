"""RGB-D tensor-layout dispatch.

The dataset emits channels-first while the deployment path feeds channels-last.
An earlier revision permuted unconditionally, which silently transposed
channels-first inputs into garbage rather than raising -- RGB planes with means
0/1/2 came back as ~1/1/1, and depth [B,1,H,W] became [B,H,1,W]. These tests
pin the dispatch that replaced it.
"""

import pytest
import torch

from navdp_safety.models.backbone import NavDP_RGBD_Backbone

to_cf = NavDP_RGBD_Backbone._to_channels_first


def _planes_chw():
    return torch.stack([torch.full((224, 224), float(i)) for i in range(3)]).unsqueeze(0)


def _planes_hwc():
    return torch.stack([torch.full((224, 224), float(i)) for i in range(3)], dim=-1).unsqueeze(0)


def test_channels_first_rgb_passes_through_unchanged():
    x = _planes_chw()
    out = to_cf(x, 3)
    assert out.shape == (1, 3, 224, 224)
    assert [out[0, c].mean().item() for c in range(3)] == [0.0, 1.0, 2.0]
    assert out.data_ptr() == x.data_ptr()          # no copy


def test_unconditional_permute_would_have_destroyed_the_channels():
    """Pins the original defect so it cannot silently return."""
    x = _planes_chw()
    broken = x.permute(0, 3, 1, 2)
    assert [round(broken[0, c].mean().item(), 3) for c in range(3)] == [1.0, 1.0, 1.0]


def test_channels_last_rgb_is_converted():
    out = to_cf(_planes_hwc(), 3)
    assert out.shape == (1, 3, 224, 224)
    assert [out[0, c].mean().item() for c in range(3)] == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("shape,expected", [
    ((2, 1, 224, 224), (2, 1, 224, 224)),          # already channels-first
    ((2, 224, 224, 1), (2, 1, 224, 224)),          # channels-last
])
def test_depth_layouts(shape, expected):
    assert to_cf(torch.rand(*shape), 1).shape == expected


@pytest.mark.parametrize("shape,expected", [
    ((2, 8, 3, 224, 224), (2, 8, 3, 224, 224)),
    ((2, 8, 224, 224, 3), (2, 8, 3, 224, 224)),
])
def test_temporal_layouts(shape, expected):
    assert to_cf(torch.rand(*shape), 3).shape == expected


def test_unlocatable_channel_axis_raises():
    with pytest.raises(ValueError, match="channel axis"):
        to_cf(torch.rand(2, 5, 224, 224), 3)
