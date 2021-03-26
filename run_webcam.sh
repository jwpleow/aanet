#!/usr/bin/env bash

# Inference on KITTI 2015 test set for submission
CUDA_VISIBLE_DEVICES=0 python webcam_inference.py \
--mode test \
--pretrained_aanet pretrained/aanet+_kitti15-2075aea1.pth \
--batch_size 1 \
--img_height 384 \
--img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--no_intermediate_supervision \
--output_dir output/kitti15_test

