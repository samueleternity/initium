"""
file: inference/tasks/task_registry.py

Single entry point used by run_inference.py (counterpart of
data/dataset_registry.py):
    task = get_task(dataset_type, dataset_link)

v2: text/audio/video/multimodal, mirroring data/dataset_registry.py's v2.
multimodal is excluded from IMPLEMENTED_TYPES (legacy dim-inference list in
capabilities.py) since its input/output dims depend on WHICH modalities are
combined, not a fixed per-type signature -- it's only usable as an explicit
--dataset-type, never inferred from a legacy checkpoint's tensor shapes.
"""

GRAPH_ALIASES = ("graph", "graph-traversal", "graph_traversal")
TEXT_ALIASES = ("text",)
AUDIO_ALIASES = ("audio",)
VIDEO_ALIASES = ("video",)
MULTIMODAL_ALIASES = ("multimodal", "multi-modal", "multi_modal")
IMPLEMENTED_TYPES = ("graph", "text", "audio", "video", "text-classic", "audio-classic", "video-classic")


def canonical_type(dataset_type: str) -> str:
    t = (dataset_type or "graph").lower()
    if t in GRAPH_ALIASES:
        return "graph"
    if t in TEXT_ALIASES:
        return "text"
    if t in AUDIO_ALIASES:
        return "audio"
    if t in VIDEO_ALIASES:
        return "video"
    if t in MULTIMODAL_ALIASES:
        return "multimodal"
    if t in ("text-classic", "audio-classic", "video-classic", "multimodal-classic"):
        return t
    return t


def get_task_class(dataset_type: str):
    t = canonical_type(dataset_type)
    if t == "graph":
        from initium.inference.tasks.graph_traversal_task import GraphTraversalTask

        return GraphTraversalTask
    if t == "text":
        from initium.inference.tasks.text_task import TextChainTask

        return TextChainTask
    if t == "audio":
        from initium.inference.tasks.audio_task import AudioChainTask

        return AudioChainTask
    if t == "video":
        from initium.inference.tasks.video_task import VideoChainTask

        return VideoChainTask
    if t == "multimodal":
        from initium.inference.tasks.multimodal_task import MultimodalTask

        return MultimodalTask
    if t in ("text-classic", "audio-classic", "video-classic", "multimodal-classic"):
        from initium.inference.tasks.classic_task import ClassicInferenceTask

        return ClassicInferenceTask
    raise ValueError(f"unknown dataset type {dataset_type!r}")


def get_task(dataset_type: str = "graph", dataset_link: str | None = None, **kwargs):
    t = canonical_type(dataset_type)
    cls = get_task_class(t)
    if t not in ("text-classic", "audio-classic", "video-classic", "multimodal-classic"):
        for key in ("test_dataset_link", "classic_modalities", "classic_window", "probe_distances", "probe_gamma"):
            kwargs.pop(key, None)
    if t == "multimodal":
        kwargs.pop("prepared_dataset", None)
        return cls((dataset_link or "text+audio").split("+"), **kwargs)
    if t in ("text-classic", "audio-classic", "video-classic", "multimodal-classic"):
        return cls(t, dataset_link, **kwargs)
    kwargs.pop("test_dataset_link", None)
    return cls(dataset_link, **kwargs)
