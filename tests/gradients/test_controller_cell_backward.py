import importlib.util

import pytest
import torch

from LNN_controller.cfc_controller import CfCControllerBlock

requires_mamba_ssm = pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None, reason="mamba-ssm is unavailable"
)


def test_cfc_cell_backward_is_finite():
    block = CfCControllerBlock(8, units=8, backbone_units=16)
    state = block.init_state(2)
    loss = torch.zeros(())
    for _ in range(5):
        out, state = block.step(torch.randn(2, 8), state)
        loss = loss + out.square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())


@requires_mamba_ssm
@pytest.mark.gpu
def test_mamba_cell_backward_is_finite():
    from mamba_controller.mamba_controller import MambaControllerCell

    cell = MambaControllerCell(8, d_state=4, d_conv=2, expand=1)
    conv_state, ssm_state = cell.init_state(2)
    loss = torch.zeros(())
    for _ in range(5):
        out, conv_state, ssm_state = cell.step(torch.randn(2, 8), conv_state, ssm_state)
        loss = loss + out.square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in cell.parameters())


@requires_mamba_ssm
@pytest.mark.gpu
def test_mamba2_cell_backward_is_finite():
    from mamba_controller.mamba2_controller import Mamba2ControllerCell

    cell = Mamba2ControllerCell(64, d_state=4, d_conv=2, expand=1, headdim=8)
    conv_state, ssm_state = cell.init_state(1)
    loss = torch.zeros(())
    for _ in range(5):
        out, conv_state, ssm_state = cell.step(torch.randn(1, 64), conv_state, ssm_state)
        loss = loss + out.square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in cell.parameters())


@requires_mamba_ssm
@pytest.mark.gpu
def test_mamba3_cell_backward_is_finite():
    from mamba_controller.mamba3_controller import Mamba3ControllerCell

    cell = Mamba3ControllerCell(64, d_state=4, expand=1, headdim=8)
    state = cell.init_state(1)
    loss = torch.zeros(())
    for _ in range(5):
        out, state = cell.step(torch.randn(1, 64), state)
        loss = loss + out.square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in cell.parameters())
