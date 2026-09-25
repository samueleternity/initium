from memory_manipulation.dynamic_memory_resize import resize_memory
from memory_manipulation.link_matrix_ablation import AblatableSparseLinkMemory, patch_link_matrix


def test_link_mode_survives_resize_and_rebinds_module(build_tiny_rnn):
    model, _, _, optimizer = build_tiny_rnn("lstm")
    patch_link_matrix(model, "ablated")
    resized = resize_memory(model, 20, device="cpu", optimizer=optimizer)
    assert isinstance(resized, AblatableSparseLinkMemory)
    assert resized.link_matrix_mode == "ablated"
    assert model.memories[0] is resized
    assert model._modules["rnn_layer_memory_shared"] is resized
