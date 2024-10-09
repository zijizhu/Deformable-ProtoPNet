#!/bin/bash

set -x

# TODO
python run_eval.py --base_architecture dinov2_vits_exp --num_prototypes 1000 --resume saved_models/CUB/dinov2_vits_exp_1000/
# TODO
python run_eval.py --base_architecture dinov2_vits_exp --num_prototypes 2000 --resume saved_models/CUB/dinov2_vits_exp_2000/

# TODO
python run_eval.py --base_architecture dinov2_vitb_exp --num_prototypes 1000 --resume saved_models/CUB/dinov2_vitb_exp_1000/
# TODO
python run_eval.py --base_architecture dinov2_vitb_exp --num_prototypes 2000 --resume saved_models/CUB/dinov2_vitb_exp_2000/
