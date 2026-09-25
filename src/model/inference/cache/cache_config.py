"""
file: inference/cache/cache_config.py

Cache-side constants. Pure constants, imported explicitly (same convention as
inference_config.py).
"""

# "off" | "prefix" | "result" | "prefix,result" | "all". Default off: the default episodes
# reshuffle the edge list every time (nothing to reuse), and persistent-mode experiments
# should not change behaviour silently. Set to "all" to make caching on-by-default.
DEFAULT_CACHE = "off"

DEFAULT_CACHE_RAM_MB = 512
DEFAULT_CACHE_DISK_MB = 2048

# Number of cache hits that are re-computed WITHOUT the cache and compared. On a
# mismatch the cache is disabled for the rest of the run (correctness over speed).
DEFAULT_VERIFY_HITS = 2
DEFAULT_VERIFY_TOL = 1e-3  # max abs diff on scored outputs and on carried state

ROOT_CHAIN = "root"  # history-chain seed (persistent-mode result keys)
