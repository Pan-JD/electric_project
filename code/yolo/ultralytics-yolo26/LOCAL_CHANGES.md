# Project integration

This source tree is bundled with electronic_project. Original license and attribution are retained.

Local change: `ultralytics/data/base.py` resolves image-list entries beginning with `./` by joining the list directory with `x[2:]`. It replaces the original global string replacement, which corrupted portable paths such as `./../filter_v1/...`.

The project wrapper supplies local model weights, fonts and runtime settings. It disables automatic dependency installation; users manage their conda environment.
