import random

import torch

from initium.inference.cache.cached_engine import CachedInferenceEngine
from initium.inference.cache.prefix_cache import PrefixStateCache
from initium.inference.cache.result_cache import ResultCache
from initium.inference.cache.stores import MemoryStore, TieredStore
from initium.inference.capabilities import detect_capabilities
from initium.inference.checkpoint_io import load_checkpoint
from initium.inference.engine import InferenceEngine
from initium.inference.inference_config import WILDCARD_TYPE
from initium.inference.metrics import aggregate
from initium.inference.model_loader import load_model
from initium.inference.tasks.graph_traversal_task import GraphTraversalTask
from initium.mamba_controller.mamba_controller import MambaDNC
from initium.memory_manipulation.stochastic_write_head_v2 import install_stochastic_write_heads


def test_inference_and_result_prefix_cache_smoke(tmp_path, synthetic_graph_edges):
    model = MambaDNC(
        input_size=92,
        hidden_size=16,
        rnn_type="lstm",
        num_layers=1,
        num_hidden_layers=1,
        nr_cells=16,
        cell_size=8,
        read_heads=2,
        independent_linears=True,
        share_memory_between_layers=True,
    )
    install_stochastic_write_heads(model, sample=False)
    output_proj = torch.nn.Linear(92, 90)
    path = tmp_path / "inference.pt"
    torch.save(
        {
            "rnn_state_dict": model.state_dict(),
            "output_proj_state_dict": output_proj.state_dict(),
            "model_config": {
                "input_size": 92,
                "input_dim": 92,
                "hidden_size": 16,
                "nr_cells": 16,
                "cell_size": 8,
                "read_heads": 2,
                "num_hidden_layers": 1,
                "controller_type": "lstm",
                "supported_dataset_types": [WILDCARD_TYPE],
            },
            "step": 2,
            "beta_target": 0.0,
        },
        path,
    )
    ckpt = load_checkpoint(str(path))
    caps = detect_capabilities(ckpt)
    assert caps.source == "recorded"
    loaded = load_model(ckpt, torch.device("cpu"), deterministic_write=True)

    task = GraphTraversalTask(synthetic_graph_edges, path_length_range=(1, 2))
    episodes = task.build_episodes(2, random.Random(17))
    engine = InferenceEngine(loaded.rnn, loaded.output_proj, task, torch.device("cpu"))
    results = engine.run(episodes, reset_experience=True)
    assert aggregate([result.score for result in results])["n_episodes"] == 2

    result_cache = ResultCache(TieredStore(MemoryStore(8_000_000)), "fp", torch.device("cpu"))
    result_engine = CachedInferenceEngine(
        loaded.rnn, loaded.output_proj, task, torch.device("cpu"), [result_cache]
    )
    assert len(result_engine.run(episodes, reset_experience=True)) == 2

    shared_task = GraphTraversalTask(
        synthetic_graph_edges, path_length_range=(1, 2), shared_context=True
    )
    shared_episodes = shared_task.build_episodes(2, random.Random(19))
    prefix_cache = PrefixStateCache(TieredStore(MemoryStore(8_000_000)), "fp", torch.device("cpu"))
    prefix_engine = CachedInferenceEngine(
        loaded.rnn, loaded.output_proj, shared_task, torch.device("cpu"), [prefix_cache]
    )
    assert len(prefix_engine.run(shared_episodes, reset_experience=True)) == 2
