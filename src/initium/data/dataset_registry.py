"""
file: data/dataset_registry.py

Single entry point used by core_training.py:
    dataset = get_dataset(dataset_type, dataset_link, test_dataset_link=None)

v2: added text / audio / video / multimodal, each a self-contained package
under data/<name>/ mirroring data/graph_traversal/'s structure. "multimodal"
fuses two or more of {text, audio, video} -- which ones is given via
dataset_link as a '+'-joined list, e.g. --dataset-type multimodal
--dataset-link text+audio (dataset_link is repurposed this one way for
multimodal only; every other type still treats it as a data-file path).

v3: dataset_link/test_dataset_link now optionally point at REAL data (a
text file, an audio file, a video file, a graph edge file -- see each
data/<name>/<name>_dataset.py and data/common/real_data.py /
data/common/graph_io.py for the loading/tokenization pipeline). Omitting
both keeps every type's original synthetic-data behavior unchanged -- this
is the explicit default the project requires. multimodal does not yet
support per-sub-modality real-data links (dataset_link is still only the
'+'-joined modality-name list there); it stays synthetic-only for now.
"""

GRAPH_ALIASES = ("graph", "graph-traversal", "graph_traversal")
TEXT_ALIASES = ("text",)
AUDIO_ALIASES = ("audio",)
VIDEO_ALIASES = ("video",)
MULTIMODAL_ALIASES = ("multimodal", "multi-modal", "multi_modal")


def get_dataset(
    dataset_type: str = "graph",
    dataset_link: str | None = None,
    test_dataset_link: str | None = None,
    prepared_dataset=None,
    **kwargs,
):
    if prepared_dataset is not None:
        return prepared_dataset
    t = (dataset_type or "graph").lower()
    if t in {"text-classic", "audio-classic", "video-classic", "multimodal-classic"}:
        from initium.config.classic_config import (
            CLASSIC_OOD_EVAL_EPISODES,
            CLASSIC_PROBE_DISTANCES,
            CLASSIC_PROBE_GAMMA,
            CLASSIC_WINDOW_SIZE,
        )
        from initium.data.common.classic_task import ClassicDataset

        modalities = {
            "text-classic": ["text"],
            "audio-classic": ["audio"],
            "video-classic": ["video"],
            "multimodal-classic": kwargs.pop("modalities", ["text", "audio"]),
        }[t]
        if t == "multimodal-classic" and len(modalities) < 2:
            raise ValueError("multimodal-classic requires at least two modalities")
        return ClassicDataset(
            modalities,
            dataset_link,
            test_dataset_link=test_dataset_link,
            window_size=kwargs.pop("window_size", CLASSIC_WINDOW_SIZE),
            probe_distances=tuple(kwargs.pop("probe_distances", CLASSIC_PROBE_DISTANCES)),
            probe_gamma=kwargs.pop("probe_gamma", CLASSIC_PROBE_GAMMA),
            eval_episodes=kwargs.pop("eval_episodes", 32),
            ood_eval_episodes=kwargs.pop("ood_eval_episodes", CLASSIC_OOD_EVAL_EPISODES),
        )
    if t in GRAPH_ALIASES:
        from initium.data.graph_traversal.graph_traversal import GraphTraversalDataset

        link = None if dataset_link in (None, "graph-traversal") else dataset_link
        return GraphTraversalDataset(dataset_link=link, test_dataset_link=test_dataset_link)
    if t in TEXT_ALIASES:
        from initium.data.text.text_dataset import TextChainDataset

        return TextChainDataset(dataset_link=dataset_link, test_dataset_link=test_dataset_link)
    if t in AUDIO_ALIASES:
        from initium.data.audio.audio_dataset import AudioChainDataset

        return AudioChainDataset(dataset_link=dataset_link, test_dataset_link=test_dataset_link)
    if t in VIDEO_ALIASES:
        from initium.data.video.video_dataset import VideoChainDataset

        return VideoChainDataset(dataset_link=dataset_link, test_dataset_link=test_dataset_link)
    if t in MULTIMODAL_ALIASES:
        from initium.data.multimodal.multimodal_dataset import MultimodalDataset

        return MultimodalDataset((dataset_link or "text+audio").split("+"))
    raise ValueError(f"unknown dataset type {dataset_type!r}")
