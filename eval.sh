#!/bin/bash

set -x

# TODO
python run_eval.py --base_architecture dinov2_vits_exp --num_prototypes 1000 --resume saved_models/CUB/dinov2_vits_exp-1000-aug/15nopush0.8554.pth
# TODO
python run_eval.py --base_architecture dinov2_vits_exp --num_prototypes 2000 --resume saved_models/CUB/dinov2_vits_exp-2000-aug/21nopush0.8602.pth

# TODO
python run_eval.py --base_architecture dinov2_vitb_exp --num_prototypes 1000 --resume saved_models/CUB/dinov2_vitb_exp-1000-aug/27nopush0.8814.pth
# TODO
python run_eval.py --base_architecture dinov2_vitb_exp --num_prototypes 2000 --resume saved_models/CUB/dinov2_vitb_exp-2000aug/0nopush0.8830.pth
