"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys

import torch.onnx
import argparse
import os

import nets
from utils import utils

parser = argparse.ArgumentParser()
# AANet
parser.add_argument('--feature_type', default='aanet', type=str, help='Type of feature extractor')
parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction')
parser.add_argument('--feature_pyramid', action='store_true', help='Use pyramid feature')
parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN')
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')
parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')
parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')
parser.add_argument('--pretrained_aanet', default=None, type=str, help='Pretrained network')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

aanet = nets.AANet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_type=args.feature_type,
                       no_feature_mdconv=args.no_feature_mdconv,
                       feature_pyramid=args.feature_pyramid,
                       feature_pyramid_network=args.feature_pyramid_network,
                       feature_similarity=args.feature_similarity,
                       aggregation_type=args.aggregation_type,
                       num_scales=args.num_scales,
                       num_fusions=args.num_fusions,
                       num_stage_blocks=args.num_stage_blocks,
                       num_deform_blocks=args.num_deform_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type,
                       mdconv_dilation=args.mdconv_dilation,
                       deformable_groups=args.deformable_groups).to(device)
    
if os.path.exists(args.pretrained_aanet):
    print('=> Loading pretrained AANet:', args.pretrained_aanet)
    utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
else:
    raise Exception(f'Model not found! {args.pretrained_aanet}')

# net.to("cuda")


left = torch.randn(1, 3, 384, 672).cuda()
right = torch.randn(1, 3, 384, 672).cuda()
inputs = (left, right)


traced_gpu = torch.jit.trace(aanet, inputs, strict=False)
torch.jit.save(traced_gpu, "gpu_model.pt")
# torch.onnx.export(aanet, (left, right), "model.onnx", verbose=False, input_names=['left', 'right'], output_names=['disparity_pyramid'], opset_version=11)