from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
from omegaconf import OmegaConf
import cv2
from PIL import Image
import imageio

import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from models.hybrik.HRNetSMPLXCamKidReg import HRNetSMPLXCamKidReg
from models.openpose import OpenposeDetector, draw_poses
from utils.transforms import test_transform
from utils.vis import get_one_box
from utils.render_pytorch3d import get_mesh, calc_focal, render_mesh, get_all_smpl_imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default="checkpoints/")
    parser.add_argument("--config_file", type=Path, default="configs/hybrik_main_config.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_vid", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)

    det_transform = T.Compose([T.ToTensor()])
    det_model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(args.device)

    hybrik_model = HRNetSMPLXCamKidReg(**cfg).eval().to(args.device)
    save_dict = torch.load(cfg.PRETRAINED, map_location='cpu')
    hybrik_model.load_state_dict(save_dict)

    smplx_faces = torch.from_numpy(hybrik_model.smplx_layer.faces.astype(np.int32))

    model_openpose = OpenposeDetector(args.device)
    model_openpose.load_dw_model(args.checkpoint_dir / "openpose")

    if args.output_path is None:
        output_path = Path("outputs") / args.input_path.stem
    else:
        output_path = args.output_path

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "frames").mkdir(parents=True, exist_ok=True)
    (output_path / "depth").mkdir(parents=True, exist_ok=True)
    (output_path / "dwpose").mkdir(parents=True, exist_ok=True)
    (output_path / "mask").mkdir(parents=True, exist_ok=True)
    (output_path / "normal").mkdir(parents=True, exist_ok=True)
    (output_path / "semantic_map").mkdir(parents=True, exist_ok=True)

    if args.input_path.is_dir(): # Input : Directory of Images
        input_imgs = sorted(list(args.input_path.glob("*.png")) + list(args.input_path.glob("*.jpg")))
    elif args.input_path.suffix in [".png", ".jpg"]: # Input : Single Image
        input_imgs = [args.input_path]
    else: # Input : Video
        print("Reading Video...")
        reader = imageio.get_reader(args.input_path)
        md = reader.get_meta_data()
        length = int(md["fps"] * md["duration"])

        index = 0
        for frame in tqdm(reader, total=length):
            Image.fromarray(frame).save(output_path / f"frames/{index:04d}.png")
            index += 1
        input_imgs = sorted(list(output_path.glob("frames/*.png")))

    with torch.no_grad():
        sample_imgs = []
        for i, img_path in tqdm(enumerate(input_imgs), total=len(input_imgs)):
            result_dict = {}
            input_image = Image.open(img_path)
            W, H = input_image.size
            result_dict["frames"] = input_image

            poses = model_openpose.detect_poses_dw(np.asarray(input_image)[:,:,::-1])
            if len(poses) > 0:
                openpose_result = draw_poses(poses[0:1], H, W, draw_body=True, draw_hand=True, draw_face=True)
                openpose_image = Image.fromarray(openpose_result)
            else:
                print("DWPose Fail, creating empty image...")
                openpose_image = Image.new("RGB", (W, H), (0,0,0))
            result_dict["dwpose"] = openpose_image

            det_input = det_transform(np.asarray(input_image)).to(args.device)
            det_result = det_model([det_input])
            if len(det_result) > 0:
                tight_bbox = get_one_box(det_result[0])  # xyxy

                pose_input, bbox, img_center = test_transform(np.asarray(input_image), tight_bbox)
                pose_input = pose_input.unsqueeze(0).to(args.device)
                bbox = torch.from_numpy(np.array(bbox)).float().unsqueeze(0).to(args.device)
                img_center = torch.from_numpy(img_center).float().unsqueeze(0).to(args.device)
                pose_result = hybrik_model(pose_input, flip_test=True, bboxes=bbox, img_center=img_center)

                vertices = pose_result.pred_vertices.detach()
                faces = smplx_faces.expand(1, *smplx_faces.shape).to(args.device)
                translation = pose_result.transl.detach()
                mesh = get_mesh(vertices, faces, translation, device=args.device)

                focal = calc_focal(bbox[0])
                fragments, result = render_mesh(mesh, focal, H, W, device=args.device)
                smpl_dict = get_all_smpl_imgs(fragments, result, mesh)
            else:
                print("HybrIK Fail, creating empty image...")
                smpl_dict = {
                    "depth" : Image.new("RGB", (W, H), (0,0,0)),
                    "mask" : Image.new("RGB", (W, H), (0,0,0)),
                    "normal" : Image.new("RGB", (W, H), (0,0,0)),
                    "semantic_map" : Image.new("RGB", (W, H), (68,1,84))
                }
            result_dict.update(smpl_dict)

            for k, v in result_dict.items():
                v.save(output_path / f"{k}/{i:04d}.png")

            if args.sample_vid:
                sum_image = Image.new("RGB", (W * 6, H))
                for ind, (k, v) in enumerate(result_dict.items()):
                    sum_image.paste(v, (W*ind, 0))
                sample_imgs.append(sum_image)
    
    if args.sample_vid:
        sample_vidpath = output_path / f"{args.input_path.stem}.mp4"

        writer = imageio.get_writer(sample_vidpath, fps=24)
        for frame in sample_imgs:
            writer.append_data(np.asarray(frame))
        writer.close()