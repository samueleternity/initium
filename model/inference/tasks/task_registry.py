"""
file: inference/tasks/task_registry.py

Single entry point used by run_inference.py (counterpart of
data/dataset_registry.py):
    task = get_task(dataset_type, dataset_link)
"""

GRAPH_ALIASES = ("graph", "graph-traversal", "graph_traversal")
IMPLEMENTED_TYPES = ("graph",)


def canonical_type(dataset_type: str) -> str:
    t = (dataset_type or "graph").lower()
    return "graph" if t in GRAPH_ALIASES else t


def get_task_class(dataset_type: str):
    t = canonical_type(dataset_type)
    if t == "graph":
        from inference.tasks.graph_traversal_task import GraphTraversalTask
        return GraphTraversalTask
    if t in ("text", "audio", "video"):
        raise NotImplementedError(
            f"inference task type {t!r}: not implemented yet. Implement "
            "inference.tasks.base_task.BaseInferenceTask and register it here."
        )
    raise ValueError(f"unknown dataset type {dataset_type!r}")


def get_task(dataset_type: str = "graph", dataset_link: str = None, **kwargs):
    return get_task_class(dataset_type)(dataset_link, **kwargs)