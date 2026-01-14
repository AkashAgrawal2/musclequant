from __future__ import annotations

# Global defaults / tunables for MuscleQuant.
DEFAULT_MODEL = "cyto2"        # or path to your fine-tuned model
DEFAULT_DIAMETER = 70          # px (None = auto; slower)
FLOW_THRESH = 0.45
CELLPROB_THRESH = 0.0
DEFAULT_PX_UM = 0.108
SYNAPSE_RING_PX = 3            # px thickness for membrane band
SYNAPSE_PERCENTILE = 95.0      # percentile of membrane intensities to call synaptic
CHUNK_COUNT_DEFAULT = 2        # default number of chunks (per dimension) when chunking first
TILE_SIZE_PX = 2048            # initial tile size for OOM fallback on GPU
MIN_TILE_PX = 512              # don't go below this when shrinking tiles
TILE_OVERLAP_PX = 64           # overlap to reduce cell splits across tiles
