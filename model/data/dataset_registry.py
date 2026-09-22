"""
file: data/dataset_registry.py

Single entry point used by core_training.py:
    dataset = get_dataset(dataset_type, dataset_link)

v2: added text / audio / video / multimodal, each a self-contained package
under data/<name>/ mirroring data/graph_traversal/'s structure. "multimodal"
fuses two or more of {text, audio, video} -- which ones is given via
dataset_link as a '+'-joined list, e.g. --dataset-type multimodal
--dataset-link text+audio (dataset_link is repurposed this one way for
multimodal only; every other type still treats it as a data-file path).
"""

GRAPH_ALIASES = ("graph", "graph-traversal", "graph_traversal")
TEXT_ALIASES = ("text",)
AUDIO_ALIASES = ("audio",)
VIDEO_ALIASES = ("video",)
MULTIMODAL_ALIASES = ("multimodal", "multi-modal", "multi_modal")


def get_dataset(dataset_type: str = "graph", dataset_link: str = None, **kwargs):
    t = (dataset_type or "graph").lower()
    if t in GRAPH_ALIASES:
        if dataset_link not in (None, "graph-traversal"):
            raise NotImplementedError(
                f"graph dataset: custom --dataset-link {dataset_link!r} not supported yet; "
                "omit it (or pass 'graph-traversal') for the built-in synthetic "
                "curriculum + London Underground OOD eval."
            )
        from data.graph_traversal.graph_traversal import GraphTraversalDataset
        return GraphTraversalDataset()
    if t in TEXT_ALIASES:
        from data.text.text_dataset import TextChainDataset
        return TextChainDataset()
    if t in AUDIO_ALIASES:
        from data.audio.audio_dataset import AudioChainDataset
        return AudioChainDataset()
    if t in VIDEO_ALIASES:
        from data.video.video_dataset import VideoChainDataset
        return VideoChainDataset()
    if t in MULTIMODAL_ALIASES:
        from data.multimodal.multimodal_dataset import MultimodalDataset
        return MultimodalDataset((dataset_link or "text+audio").split("+"))
    raise ValueError(f"unknown dataset type {dataset_type!r}")