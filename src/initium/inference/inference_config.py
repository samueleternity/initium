"""
file: inference/inference_config.py

Inference-side constants. Pure constants, imported explicitly by the other
inference modules (same convention as config/train_config.py).
"""

INFERENCE_LOG_DIR = "./inference_logs"

# Default task: graph traversal on the London Underground graph.
DEFAULT_DATASET_TYPE = "graph"
DEFAULT_DATASET_LINK = None          # None -> built-in London Underground graph

DEFAULT_NUM_EPISODES = 200           # fresh mode (and persistent mode when --loop is omitted)
DEFAULT_WINDOW = 10                  # episodes per window in the persistent-mode adaptation table
DEFAULT_SEED = 0
DEFAULT_VERBOSE_N = 3                # print predictions of the first N episodes
PROGRESS_EVERY = 25                  # running-accuracy console line every N episodes (0 = off)

# A checkpoint whose supported_dataset_types contains this accepts ANY task
# whose input/output dims match its weights.
WILDCARD_TYPE = "*"