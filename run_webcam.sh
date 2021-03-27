#!/usr/bin/env bash

# Inference on KITTI 2015 test set for submission
# CUDA_VISIBLE_DEVICES=0 python webcam_inference.py \
# --mode test \
# --pretrained_aanet pretrained/aanet+_kitti15-2075aea1.pth \
# --batch_size 1 \
# --img_height 384 \
# --img_width 1248 \
# --feature_type ganet \
# --feature_pyramid \
# --refinement_type hourglass \
# --no_intermediate_supervision \
# --output_dir output/kitti15_test

# StereoNet-AA
# CUDA_VISIBLE_DEVICES=0 python webcam_inference.py \
# --mode test \
# --pretrained_aanet pretrained/stereonet-aa_sceneflow-1fbe2dea.pth \
# --batch_size 1 \
# --img_height 576 \
# --img_width 960 \
# --feature_type stereonet \
# --num_scales 1 \
# --num_fusions 4 \
# --num_deform_blocks 4 \
# --refinement_type stereonet

# aanet+, sceneflow 
CUDA_VISIBLE_DEVICES=0 python webcam_inference.py \
--mode test \
--pretrained_aanet pretrained/aanet+_sceneflow-d3e13ef0.pth \
--batch_size 1 \
--img_height 576 \
--img_width 960 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--no_intermediate_supervision