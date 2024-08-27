# All Rights Reserved
# 
# Copyright © 2024 Patrick Devaney
# 
# THE CONTENTS OF THIS PROJECT ARE PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED COPYING, TRANSFERRING, OR REPRODUCTION OF THE CONTENTS OF THIS PROJECT, VIA ANY MEDIUM, IS STRICTLY PROHIBITED.
# 
# The receipt or possession of the source code and/or any parts thereof does not convey or imply any right to use them for any purpose other than the purpose for which they were provided to you.
# 
# The software is provided “AS IS”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the software.

import argparse
import os

import rembg
import torch
from PIL import Image
from tqdm import tqdm

from sf3d.system import SF3D
from sf3d.utils import remove_background, resize_foreground

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", type=str, help="Path to input folder containing .png images."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to use. If no CUDA-compatible device is found, the baking will fail. Default: 'cuda:0'",
    )
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/stable-fast-3d",
        type=str,
        help="Path to the pretrained model. Could be either a huggingface model id or a local path. Default: 'stabilityai/stable-fast-3d'",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=0.85,
        type=float,
        help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="Output directory to save the results. Default: 'output/'",
    )
    parser.add_argument(
        "--texture-resolution",
        default=1024,
        type=int,
        help="Texture atlas resolution. Default: 1024",
    )
    parser.add_argument(
        "--remesh_option",
        choices=["none", "triangle", "quad"],
        default="none",
        help="Remeshing option",
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for inference"
    )
    args = parser.parse_args()

    # Ensure args.device contains cuda
    if "cuda" not in args.device:
        raise ValueError(
            "CUDA device is required for baking and hence running the method."
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    model = SF3D.from_pretrained(
        args.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()

    rembg_session = rembg.new_session()
    images = []
    image_names = []
    for image_path in os.listdir(args.image):
        if image_path.endswith(".png"):
            # Handle each .png image
            full_image_path = os.path.join(args.image, image_path)
            image = remove_background(
                Image.open(full_image_path).convert("RGBA"), rembg_session
            )
            image = resize_foreground(image, args.foreground_ratio)
            image_name = os.path.splitext(image_path)[0]
            image_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_output_dir, exist_ok=True)
            image.save(os.path.join(image_output_dir, "input.png"))
            images.append(image)
            image_names.append(image_name)

    for i in tqdm(range(0, len(images), args.batch_size)):
        batch_images = images[i : i + args.batch_size]
        batch_names = image_names[i : i + args.batch_size]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mesh, glob_dict = model.run_image(
                    batch_images,
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                )
        print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

        if len(batch_images) == 1:
            out_mesh_path = os.path.join(output_dir, batch_names[0], f"{batch_names[0]}.glb")
            mesh.export(out_mesh_path, include_normals=True)
        else:
            for j in range(len(mesh)):
                out_mesh_path = os.path.join(output_dir, batch_names[j], f"{batch_names[j]}.glb")
                mesh[j].export(out_mesh_path, include_normals=True)
