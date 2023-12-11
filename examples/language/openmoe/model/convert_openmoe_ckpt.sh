ROOT="/data/personal/nus-hx/t5_checkpoints/t5_1_1_small"

python convert_openmoe_ckpt.py \
--t5x_checkpoint_path ${ROOT}/checkpoint_1000000 \
--config_file ${ROOT}/config.json \
--pytorch_dump_path ${ROOT}/
