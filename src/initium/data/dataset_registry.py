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
SURPRISE_RECALL_ALIASES = ("surprise-recall", "surprise_recall", "sgr")
WORKSPACE_CAPABILITY_ALIASES = ("workspace-capability", "workspace_capability")


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
    if t in SURPRISE_RECALL_ALIASES:
        from initium.data.surprise_recall.surprise_recall import SurpriseRecallDataset

        return SurpriseRecallDataset()
    if t in WORKSPACE_CAPABILITY_ALIASES:
        from initium.data.workspace_capability.dataset import WorkspaceCapabilityDataset

        return WorkspaceCapabilityDataset(
            task_track=kwargs.get("workspace_task_track", "all"),
            batch_size_hint=kwargs.get("batch_size_hint", 16),
        )
    raise ValueError(f"unknown dataset type {dataset_type!r}")
