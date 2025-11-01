import os

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from rgbddepth.dpt import RGBDDepth

# Automatically select the best available device for inference
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Model configurations for different Vision Transformer (ViT) encoder sizes
# Each config specifies the encoder type, feature dimensions, and output channels
model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },  # Small
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },  # Base
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },  # Large
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },  # Giant
}


def load_images(rgb_path, depth_path, depth_scale, max_depth):
    """Load and preprocess RGB and depth images.

    Args:
        rgb_path: Path to RGB image file
        depth_path: Path to depth image file
        depth_scale: Scale factor to convert depth values to meters
        max_depth: Maximum valid depth value (values above this are set to 0)

    Returns:
        tuple: (rgb_image, depth_low_res, similarity_depth_low_res)
            - rgb_image: RGB image in numpy array format (BGR -> RGB)
            - depth_low_res: Depth values in meters
            - similarity_depth_low_res: Inverse depth values (1/depth) for model input
    """
    # Load RGB image and convert from BGR to RGB
    rgb_src = np.asarray(cv2.imread(rgb_path)[:, :, ::-1])
    if rgb_src is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")

    # Load depth image (usually 16-bit)
    depth_low_res = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_low_res is None:
        raise ValueError(f"Could not load depth image from {depth_path}")

    # Convert depth to meters and clamp invalid values
    depth_low_res = np.asarray(depth_low_res).astype(np.float32) / depth_scale
    depth_low_res[depth_low_res > max_depth] = 0.0  # Remove values beyond max range

    # Create similarity depth (inverse depth) for model input
    # Only compute inverse for valid depth values
    simi_depth_low_res = np.zeros_like(depth_low_res)
    simi_depth_low_res[depth_low_res > 0] = 1 / depth_low_res[depth_low_res > 0]

    # print(f"Images loaded: RGB {rgb_src.shape}, Depth {depth_low_res.shape}")
    return rgb_src, depth_low_res, simi_depth_low_res


def load_model(encoder, model_path):
    """Load and initialize the RGBD depth estimation model.

    Args:
        encoder: Model encoder type ('vits', 'vitb', 'vitl', 'vitg')
        model_path: Path to the model checkpoint file

    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    # Initialize model with configuration for specified encoder
    model = RGBDDepth(**model_configs[encoder])

    # Load checkpoint and extract state dict
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model" in checkpoint:
        # Handle checkpoints that wrap state dict in 'model' key
        # Remove 'module.' prefix if present (from DataParallel training)
        states = {k[7:]: v for k, v in checkpoint["model"].items()}
    elif "state_dict" in checkpoint:
        states = checkpoint["state_dict"]
        states = {k[9:]: v for k, v in states.items()}
    else:
        # Direct state dict checkpoint
        states = checkpoint

    # Load weights and move to device
    model.load_state_dict(states, strict=False)
    model = model.to(DEVICE).eval()

    print(f"Model loaded: {encoder} from {model_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--dataset', type=str, default='PhoCAL', choices=['HAMMER', 'HouseCat6D', 'PhoCAL', 'TransCG', 'XYZ-IBD', 'YCB-V', 'T-LESS', 'GN-Trans'])
    parser.add_argument('--dataset_root', type=str, default='/data/robotarm/dataset')
    parser.add_argument('--split', type=str, default='/home/robotarm/object_depth_percetion/dataset/splits/PhoCAL_test.txt', help='Path to split file listing RGB images')
    parser.add_argument('--output_root', type=str, default='/data/robotarm/result/depth/mixed')
    parser.add_argument('--method', type=str, default='cdm_zs')
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="Model encoder type",
    )
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--min-depth', type=float, default=0.001)
    parser.add_argument('--max-depth', type=float, default=5)
    parser.add_argument('--camera', type=str, default='d435', choices=['l515', 'd435', 'tof'])
    args = parser.parse_args()

    args.pred_only = True
    args.grayscale = True
    depth_factor = 1000.0

    # Load the trained model
    model = load_model(args.encoder, args.model_path)

    # ===== 读取 split 文件中的 rgb 路径 =====
    with open(args.split, 'r') as f:
        rgb_lines = [line.strip().split()[0] for line in f if line.strip()]

    for rgb_rel_path in tqdm(rgb_lines):
        rgb_path = os.path.join(args.dataset_root, args.dataset, rgb_rel_path)
        depth_scale = 1.0

        # 推导出 raw depth 路径
        if args.dataset == 'HAMMER':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'polarization', f'depth_{args.camera}', f'{frame_id:06d}.png')
        elif args.dataset == 'HouseCat6D':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'PhoCAL':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'TransCG':
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(rgb_rel_path.split('/')[-2])
            if args.camera == 'd435':
                depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id}', 'depth1.png')
            elif args.camera == 'l515':
                depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id}', 'depth2.png')                
        elif args.dataset == 'XYZ-IBD':
            depth_scale = 0.09999999747378752
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'val', scene_name, 'depth_xyz', f'{frame_id:06d}.png')
        elif args.dataset == 'GN-Trans':
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(rgb_rel_path.split('/')[-1].split('_')[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id:04d}_depth_sim.png')
        elif args.dataset == 'YCB-V':
            depth_scale = 0.1
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'test', scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'T-LESS':
            depth_scale = 0.1
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'test_primesense', scene_name, 'depth', f'{frame_id:06d}.png')
            
        if not os.path.exists(depth_path):
            print(f'[Warning] Raw depth not found: {depth_path}, skipping')
            continue

        # Load and preprocess input images
        rgb_src, depth_low_res, simi_depth_low_res = load_images(
            rgb_path, depth_path, depth_factor/depth_scale, args.max_depth
        )

        # Run model inference
        pred_depth = model.infer_image(rgb_src, simi_depth_low_res, input_size=args.input_size)
        # print(
        #     f"Prediction info: shape={pred_depth.shape}, min={pred_depth.min():.4f}, max={pred_depth.max():.4f}"
        # )

        # Convert from inverse depth back to regular depth
        pred_depth = 1 / pred_depth

        save_dir = os.path.join(args.output_root, args.dataset, args.method, args.encoder, scene_name)
        os.makedirs(save_dir, exist_ok=True)

        metric_depth = (pred_depth * depth_factor).astype(np.uint16)
        cv2.imwrite(os.path.join(save_dir, f'{frame_id:06d}_depth.png'), metric_depth)
