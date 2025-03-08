import os
import json
import torch
from PIL import Image
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

if __name__ == '__main__':
    model = NEWCRFIF(version='large07', pretrained=None)
    model.eval()
    model.cuda()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_path = os.path.join(os.path.dirname(script_dir), 'model_zoo/Release', 'wild_camera_all.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    # Path to SUNRGBD_test.json and read data
    annotation_file = "/p/weakocc/datasets/omni3d/Omni3D/SUNRGBD_test.json"
    with open(annotation_file, "r") as f:
        data = json.load(f)
    
    images_data = data["images"]

    sunrgbd_pics_path = "/p/weakocc/datasets/omni3d"
    output_file = os.path.join(os.path.dirname(__file__), "sunrgbd_focal_length_errors.txt")
    
    # Loop over each image entry
    for idx, img_info in enumerate(images_data):
        # "file_path" might be something like:
        # "SUNRGBD/kv2/kinect2data/.../image/0000103.jpg"
        rel_path = img_info["file_path"]

        images_path = os.path.join(sunrgbd_pics_path, rel_path)

        # Extract the intrinsics matrix K:
        # e.g., [[529.5, 0.0, 365.0],
        #        [0.0, 529.5, 265.0],
        #        [0.0,   0.0,   1.0]]
        K = img_info["K"]
        fx = K[0][0]  # typically focal length in pixels
        fy = K[1][1]
        
        focal_gt = (fx + fy) / 2.0

        # Run inference
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
        
        