import pytest
import torch
from inference.capabilities import (
    IncompatibleModelError,
    detect_capabilities,
    require_dims_match,
    require_type_supported,
)


def _checkpoint(config):
    return {
        "model_config": config,
        "output_proj_state_dict": {"weight": torch.zeros(4, 8)},
    }


def test_recorded_and_legacy_capabilities():
    recorded = detect_capabilities(_checkpoint({"input_size": 8, "supported_dataset_types": ["*"]}))
    assert recorded.source == "recorded"
    assert recorded.supports("graph")
    assert require_type_supported(recorded, "graph", "x.pt") == "graph"
    legacy = detect_capabilities(_checkpoint({"input_size": 8}))
    assert legacy.source in {"inferred-legacy", "unknown"}
    with pytest.raises(IncompatibleModelError):
        require_type_supported(legacy, "not-a-task", "x.pt")


def test_dimension_mismatch_is_rejected():
    caps = detect_capabilities(_checkpoint({"input_size": 8, "supported_dataset_types": ["*"]}))

    class Task:
        input_dim = 9
        output_dim = 4
        name = "tiny"

    class MatchingTask:
        input_dim = 8
        output_dim = 4
        name = "tiny"

    assert require_dims_match(caps, MatchingTask(), "x.pt") is None
    with pytest.raises(IncompatibleModelError, match="expects"):
        require_dims_match(caps, Task(), "x.pt")
