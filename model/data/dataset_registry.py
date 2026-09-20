"""
file: data/dataset_registry.py

Single entry point used by core_training.py:
    dataset = get_dataset(dataset_type, dataset_link)
"""

GRAPH_ALIASES = ("graph", "graph-traversal", "graph_traversal")


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
    if t in ("text", "audio", "video"):
        raise NotImplementedError(
            f"dataset type {t!r}: not implemented yet. Implement "
            "data.base_dataset.BaseDataset and register it here."
        )
    raise ValueError(f"unknown dataset type {dataset_type!r}")