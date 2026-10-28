"""DroneGen server: text -> image -> 3D mesh for drone show figures.

One FastAPI process with a single job queue and a single worker thread, so that
GPU models never run side by side. Backends are pluggable (see ``backends``).
"""

__version__ = "0.1.0"
