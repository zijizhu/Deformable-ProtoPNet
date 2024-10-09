python3 main.py -gpuid='0' \
                    -m=0.1 \
                    -last_layer_fixed=True \
                    -subtractive_margin=True \
                    -using_deform=True \
                    -topk_k=1 \
                    -num_prototypes=1000 \
                    -incorrect_class_connection=-0.5 \
                    -deformable_conv_hidden_channels=128 \
                    -rand_seed=1 \
                    -base_architecture="dinov2_vitb_exp" \
                    -offline_augmentation

python3 main.py -gpuid='0' \
                    -m=0.1 \
                    -last_layer_fixed=True \
                    -subtractive_margin=True \
                    -using_deform=True \
                    -topk_k=1 \
                    -num_prototypes=2000 \
                    -incorrect_class_connection=-0.5 \
                    -deformable_conv_hidden_channels=128 \
                    -rand_seed=1 \
                    -base_architecture="dinov2_vitb_exp" \
                    -offline_augmentation

python3 main.py -gpuid='0' \
                    -m=0.1 \
                    -last_layer_fixed=True \
                    -subtractive_margin=True \
                    -using_deform=True \
                    -topk_k=1 \
                    -num_prototypes=1000 \
                    -incorrect_class_connection=-0.5 \
                    -deformable_conv_hidden_channels=128 \
                    -rand_seed=1 \
                    -base_architecture="dinov2_vits_exp" \
                    -offline_augmentation

python3 main.py -gpuid='0' \
                    -m=0.1 \
                    -last_layer_fixed=True \
                    -subtractive_margin=True \
                    -using_deform=True \
                    -topk_k=1 \
                    -num_prototypes=2000 \
                    -incorrect_class_connection=-0.5 \
                    -deformable_conv_hidden_channels=128 \
                    -rand_seed=1 \
                    -base_architecture="dinov2_vits_exp" \
                    -offline_augmentation
