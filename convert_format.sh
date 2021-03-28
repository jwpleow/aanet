CUDA_VISIBLE_DEVICES=0 python convert_format.py \
--pretrained_aanet pretrained/aanet+_sceneflow-d3e13ef0.pth \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--no_intermediate_supervision