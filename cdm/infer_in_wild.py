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

import time
def benchmark_inference(model,
                        example_inputs,
                        device='cuda',
                        n_warmup=10,
                        n_iters=50):
    """
    精确测 PyTorch 模型单次 forward 时间（不含数据加载等开销）

    model: 已构建好的 nn.Module
    example_inputs: 和真实推理时 shape 一致的输入 tensor（或 tuple/list of tensors）
    device: 'cuda' or 'cpu'
    n_warmup: 预热次数（不计时）
    n_iters: 正式计时的迭代次数
    """

    # 关闭梯度
    torch.set_grad_enabled(False)

    # -------- warm-up，不计时 --------
    for _ in range(n_warmup):
        _ = model(example_inputs)
    if device == 'cuda':
        torch.cuda.synchronize()

    times_ms = []

    # -------- 正式计时 --------
    if device == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)

        for _ in range(n_iters):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            torch.cuda.synchronize()              # 等 GPU 完成
            times_ms.append(starter.elapsed_time(ender))  # 单位: ms
    else:
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(example_inputs)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)   # s -> ms

    times = torch.tensor(times_ms)
    mean_ms = times.mean().item()
    std_ms  = times.std(unbiased=False).item()

    # batch 维度（用于算 per-image 时间 / FPS）
    if torch.is_tensor(example_inputs):
        batch_size = example_inputs.size(0)
    elif isinstance(example_inputs, (list, tuple)) and torch.is_tensor(example_inputs[0]):
        batch_size = example_inputs[0].size(0)
    else:
        batch_size = 1

    print(f"[{device}] {n_iters} runs, batch_size={batch_size}")
    print(f"  mean  : {mean_ms:.3f} ms / batch")
    print(f"  std   : {std_ms:.3f} ms")
    print(f"  per-img: {mean_ms / batch_size:.3f} ms")
    print(f"  FPS   : {1000.0 / (mean_ms / batch_size):.2f}")

    return mean_ms, std_ms, times_ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--input-root',type=str, default='/home/robotarm/object_depth_percetion/data/in-wild/12.3.2025')
    parser.add_argument('--output-root', type=str, default='/home/robotarm/object_depth_percetion/vis_in_wild')
    parser.add_argument('--method', type=str, default='cdm_zs')
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="Model encoder type",
    )
    parser.add_argument('--model-path', type=str, default='cdm_d435.ckpt')
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
    color_paths = sorted(
        glob.glob(os.path.join(args.input_root, "color_*.png"))
    )
    print(f"Found {len(color_paths)} color images in {args.input_root}")
    
    save_root = os.path.join(args.output_root, args.method, args.encoder)
    os.makedirs(save_root, exist_ok=True)

    for rgb_path in tqdm(color_paths):
        name = os.path.basename(rgb_path)  # color_xxx.png
        scene_name = rgb_path.split("/")[-2]
        idx = name.replace("color_", "").replace(".png", "")
        depth_path = os.path.join(args.input_root, f"depth_{idx}.png")
        
        if not os.path.isfile(depth_path):
            print(f"[WARN] depth file not found for {name}, skip.")
            continue

        # Load and preprocess input images
        rgb_src, depth_low_res, simi_depth_low_res = load_images(
            rgb_path, depth_path, depth_factor, args.max_depth
        )

        # Run model inference
        pred_depth = model.infer_image(rgb_src, simi_depth_low_res, input_size=args.input_size)
        # inputs, (h, w) = model.image2tensor(rgb_src, simi_depth_low_res, args.input_size)
        # benchmark_inference(model, inputs, 'cuda', n_warmup=10, n_iters=50)
        # print(
        #     f"Prediction info: shape={pred_depth.shape}, min={pred_depth.min():.4f}, max={pred_depth.max():.4f}"
        # )

        # Convert from inverse depth back to regular depth
        pred_depth = 1 / pred_depth

        out_path = os.path.join(save_root, f"{scene_name}_{idx}_depth.png")
        
        metric_depth = (pred_depth * depth_factor).astype(np.uint16)
        cv2.imwrite(out_path, metric_depth)
