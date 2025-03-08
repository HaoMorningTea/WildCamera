import os
import json
import torch
from PIL import Image
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

def run_inference(annotation_file, images_base_path, output_file):
    model = NEWCRFIF(version='large07', pretrained=None)
    model.eval()
    model.cuda()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_path = os.path.join(os.path.dirname(script_dir), 'model_zoo/Release', 'wild_camera_all.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    with open(annotation_file, "r") as f:
        data = json.load(f)
    
    images_data = data["images"]

    for idx, img_info in enumerate(images_data):
        rel_path = img_info["file_path"]
        images_path = os.path.join(images_base_path, rel_path)

        K = img_info["K"]
        fx = K[0][0]
        fy = K[1][1]
        focal_gt = (fx + fy) / 2.0

        img_pil = Image.open(images_path).convert("RGB")
        intrinsic_est, _ = model.inference(img_pil, wtassumption=False)
        focal_est = intrinsic_est[0, 0].item()

        print(
            f"Image {idx}: {rel_path}\n"
            f"  GT focal   = {focal_gt:.2f}\n"
            f"  Est focal  = {focal_est:.2f}\n"
            f"  Percent error = {(focal_est - focal_gt) / focal_gt:.2%}\n"
        )
        
        with open(output_file, "a") as f:
            f.write(
                f"Image {idx}: {rel_path}\n"
                f"  GT focal   = {focal_gt:.2f}\n"
                f"  Est focal  = {focal_est:.2f}\n"
                f"  Percent error = {(focal_est - focal_gt) / focal_gt:.2%}\n"
            )

if __name__ == '__main__':
    datasets = {
        "SUNRGBD": {
            "annotation_file": "/p/weakocc/datasets/omni3d/Omni3D/SUNRGBD_test.json",
            "images_base_path": "/p/weakocc/datasets/omni3d",
            "output_file": "sunrgbd_focal_length_errors.txt"
        },
        "ARKitScenes": {
            "annotation_file": "/p/weakocc/datasets/omni3d/Omni3D/ARKitScenes_test.json",
            "images_base_path": "/p/weakocc/datasets/omni3d",
            "output_file": "arkitscenes_focal_length_errors.txt"
        },
        "Hypersim": {
            "annotation_file": "/p/weakocc/datasets/omni3d/Omni3D/Hypersim_test.json",
            "images_base_path": "/p/weakocc/datasets/omni3d",
            "output_file": "hypersim_focal_length_errors.txt"
        },
        "Objectron": {
            "annotation_file": "/p/weakocc/datasets/omni3d/Omni3D/Objectron_test.json",
            "images_base_path": "/p/weakocc/datasets/omni3d",
            "output_file": "objectron_focal_length_errors.txt"
        },
        "KITTI": {
            "annotation_file": "/p/weakocc/datasets/omni3d/Omni3D/KITTI_test.json",
            "images_base_path": "/p/weakocc/datasets/omni3d",
            "output_file": "kitti_focal_length_errors.txt"
        },
        "nuScenes": {
            "annotation_file": "/p/weakocc/datasets/omni3d/Omni3D/nuScenes_test.json",
            "images_base_path": "/p/weakocc/datasets/omni3d",
            "output_file": "nuscenes_focal_length_errors.txt"
        }
    }

    for dataset_name, dataset_info in datasets.items():
        print(f"Running inference on {dataset_name} dataset...")
        run_inference(
            dataset_info["annotation_file"],
            dataset_info["images_base_path"],
            os.path.join(os.path.dirname(__file__), dataset_info["output_file"])
        )