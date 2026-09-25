import random

import torch
from initium.memory_manipulation.dynamic_memory_resize import resize_memory


def test_resize_transplants_weights_head_and_optimizer_state(build_tiny_rnn):
    model, _, heads, optimizer = build_tiny_rnn("lstm")
    memory = model.memories[0]
    head = memory.write_vector_transform
    stable_layer = memory.read_keys_transform
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    old_parameter = stable_layer.weight
    assert old_parameter in optimizer.state
    resized = resize_memory(model, 20, device="cpu", optimizer=optimizer)
    assert model.memories[0].nr_cells == 20
    assert resized.write_vector_transform is head is heads[0]
    assert model._modules["rnn_layer_memory_shared"] is resized
    torch.testing.assert_close(resized.read_keys_transform.weight, old_parameter)
    new_parameter = resized.read_keys_transform.weight
    assert new_parameter in optimizer.state
    assert any(new_parameter is p for group in optimizer.param_groups for p in group["params"])


def test_resized_cell_count_is_checkpointed_and_reloaded(
    build_tiny_rnn, tmp_checkpoint_dir, monkeypatch
):
    import initium.core_training as core_training
    from initium.inference.checkpoint_io import load_checkpoint
    from initium.inference.model_loader import load_model

    model, _, heads, optimizer = build_tiny_rnn("lstm")
    resize_memory(model, 20, device="cpu", optimizer=optimizer)
    monkeypatch.setattr(core_training, "MODEL_HIDDEN_SIZE", 16)
    monkeypatch.setattr(core_training, "MODEL_CELL_SIZE", 8)
    monkeypatch.setattr(core_training, "MODEL_READ_HEADS", 2)
    projection = torch.nn.Linear(8, 8)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    class Curriculum:
        lesson = 0

    path = tmp_checkpoint_dir / "resized.pt"
    core_training.save_checkpoint(
        str(path),
        model,
        projection,
        heads,
        optimizer,
        Curriculum(),
        step=3,
        beta_target=0.0,
        run_id="resize-checkpoint",
        scaler=scaler,
        ood_rng=random.Random(3),
        controller_type="lstm",
    )
    loaded = load_model(load_checkpoint(str(path)), torch.device("cpu"), deterministic_write=True)
    assert loaded.rnn.memories[0].nr_cells == 20
