import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import skimage.io
import argparse
import numpy as np
import time
import os
import cv2
import math

# from memory_profiler import profile
import nets
import dataloader
from dataloader import transforms
from utils import utils
from utils.file_io import write_pfm

import webcamgrabber
import visualise

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str,
                    help='Validation mode on small subset or test mode on full test data')

parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=576, type=int, help='Image height for inference')
parser.add_argument('--img_width', default=960, type=int, help='Image width for inference')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--output_dir', default='output', type=str,
                    help='Directory to save inference results')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')

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

parser.add_argument('--save_type', default='png', choices=['pfm', 'png', 'npy'], help='Save file type')
parser.add_argument('--visualize', action='store_true', help='Visualize disparity map')

# Log

args = parser.parse_args()

model_name = os.path.basename(args.pretrained_aanet)[:-4]
model_dir = os.path.basename(os.path.dirname(args.pretrained_aanet))
args.output_dir = os.path.join(args.output_dir, model_dir + '-' + model_name)

utils.check_path(args.output_dir)
utils.save_command(args.output_dir)

# @profile
def main():
    # cam = webcamgrabber.Arducam("rtsp://192.168.1.70:8554/test")
    # cam = webcamgrabber.Arducam("parallel_dining.mp4")
    cam = webcamgrabber.Arducam("udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=JPEG, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink")
    left, right = cam.read()
    img_height, img_width= left.shape[:2]

    vis = visualise.Visualiser(cam.Q_)
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test loader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    
    print(f"Creating AANet...")
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

    # print(aanet)

    if os.path.exists(args.pretrained_aanet):
        print('=> Loading pretrained AANet:', args.pretrained_aanet)
        utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
    else:
        raise Exception(f'Model not found! {args.pretrained_aanet}')

    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)

    # Inference
    aanet.eval()
    inference_time = 0
    framecount = 0
    print(f"Finished warmup, starting inference...")
    while True:
        print(f"Frame {framecount}")
        left_img, right_img = cam.read()
        cv2.imshow("left", left_img)
        cv2.imshow("right", right_img)
        img = {'left': left_img, 'right': right_img}
        img = test_transform(img)

        left = img['left'].unsqueeze(0).to(device)  # [B, 3, H, W]
        right = img['right'].unsqueeze(0).to(device)
        
        # Pad
        ori_height, ori_width = left.size()[2:]

        factor = 48 if args.refinement_type != 'hourglass' else 96
        img_height = math.ceil(ori_height / factor) * factor
        img_width = math.ceil(ori_width / factor) * factor

        if ori_height < img_height or ori_width < img_width:
            top_pad = img_height - ori_height
            right_pad = img_width - ori_width

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))

        framecount += left.size(0)

        # print("Performing inference...")
        with torch.no_grad():
            time_start = time.perf_counter()
            pred_disp = aanet(left, right)  # [B, C, H, W]
            inference_time += time.perf_counter() - time_start

        if pred_disp.size(-1) < left.size(-1):
            print("Interpolating disparity...")
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
                                      mode='bilinear', align_corners=True, recompute_scale_factor=True) * (left.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.cpu().squeeze(1)  # [B, H, W]

        # Crop
        if ori_height < img_height or ori_width < img_width:
            if right_pad != 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
            else:
                pred_disp = pred_disp[:, top_pad:]

        disp = pred_disp[0].detach().cpu().numpy()

        vis.update(disp, left_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            vis.release()
            break

    # save image
    disp = 255 * disp
    img = disp.astype(np.uint8)
    cv2.imwrite("disparity.png", img)
    cv2.imwrite("left.png", left_img)
    cv2.imwrite("right.png", right_img)
    print('=> Mean inference time for %d images: %.3fs' % (framecount, inference_time / framecount))


if __name__ == '__main__':
    main()
