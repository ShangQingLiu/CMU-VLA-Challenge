#!/bin/bash


CHUNKS=1
# MODEL_PATH="model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split" 
MODEL_PATH="/dataSSD/chen/navid_ws/NaVid-VLN-CE/model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split"


#R2R
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
CONFIG_PATH="/dataSSD/chen/navid_ws/NaVid-VLN-CE/VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
# SAVE_PATH="tmp/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split_on_r2r" 
SAVE_PATH="/homeL/chen/Videos/"

#RxR
#CONFIG_PATH="VLN_CE/vlnce_baselines/config/rxr_baselines/navid_rxr.yaml"
#SAVE_PATH="tmp/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split_on_rxr" 


for IDX in $(seq 0 $((CHUNKS-1))); do
    echo $(( IDX % 8 ))
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python /dataSSD/chen/CMU-VLA-Challenge/ai_module/src/dummy_vlm/src/navid_ws/NaVid-VLN-CE/run.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --model-path $MODEL_PATH \
    --result-path $SAVE_PATH &
    
done

wait

