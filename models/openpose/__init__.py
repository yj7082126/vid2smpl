# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from . import util
from .body import Body, BodyResult, Keypoint
from .hand import Hand
from .face import Face
from .types import HandResult, FaceResult, HumanPoseResult, AnimalPoseResult

from typing import Tuple, List, Callable, Union, Optional
from urllib.parse import urlparse

body_model_path = (
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
)
hand_model_path = (
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
)
face_model_path = (
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"
)

remote_onnx_det = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
remote_onnx_pose = (
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
)



def draw_poses(
    poses: List[HumanPoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True
):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[HumanPoseResult]): A list of HumanPoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas

def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file

class OpenposeDetector:
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """

    def __init__(self, device):
        self.device = device
        self.body_estimation = None
        self.hand_estimation = None
        self.face_estimation = None

        self.dw_pose_estimation = None
        self.animal_pose_estimation = None

    def load_model(self, model_dir):
        """
        Load the Openpose body, hand, and face models.
        """
        body_modelpath = os.path.join(model_dir, "body_pose_model.pth")
        hand_modelpath = os.path.join(model_dir, "hand_pose_model.pth")
        face_modelpath = os.path.join(model_dir, "facenet.pth")

        if not os.path.exists(body_modelpath):
            load_file_from_url(body_model_path, model_dir=model_dir)

        if not os.path.exists(hand_modelpath):
            load_file_from_url(hand_model_path, model_dir=model_dir)

        if not os.path.exists(face_modelpath):
            load_file_from_url(face_model_path, model_dir=model_dir)

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)
        self.face_estimation = Face(face_modelpath)

    def load_dw_model(self, model_dir):
        from .wholebody import Wholebody  # DW Pose

        def load_model(filename: str, remote_url: str):
            local_path = os.path.join(model_dir, filename)
            if not os.path.exists(local_path):
                load_file_from_url(remote_url, model_dir=model_dir)
            return local_path

        onnx_det = load_model("yolox_l.onnx", remote_onnx_det)
        onnx_pose = load_model("dw-ll_ucoco_384.onnx", remote_onnx_pose)
        self.dw_pose_estimation = Wholebody(onnx_det, onnx_pose)

    def unload_model(self):
        """
        Unload the Openpose models by moving them to the CPU.
        Note: DW Pose models always run on CPU, so no need to `unload` them.
        """
        if self.body_estimation is not None:
            self.body_estimation.model.to("cpu")
            self.hand_estimation.model.to("cpu")
            self.face_estimation.model.to("cpu")

    def detect_hands(
        self, body: BodyResult, oriImg
    ) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in util.handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y : y + w, x : x + w, :]).astype(
                np.float32
            )
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(
                    W
                )
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(
                    H
                )

                hand_result = [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = util.faceDetect(body, oriImg)
        if face is None:
            return None

        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y : y + w, x : x + w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(
            np.float32
        )
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            return [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]

        return None

    def detect_poses(
        self, oriImg, include_hand=False, include_face=False
    ) -> List[HumanPoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[HumanPoseResult]: A list of HumanPoseResult objects containing the detected poses.
        """
        self.body_estimation.model.to(self.device)
        self.hand_estimation.model.to(self.device)
        self.face_estimation.model.to(self.device)

        self.body_estimation.cn_device = self.device
        self.hand_estimation.cn_device = self.device
        self.face_estimation.cn_device = self.device

        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = (None,) * 3
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)

                results.append(
                    HumanPoseResult(
                        BodyResult(
                            keypoints=[
                                Keypoint(
                                    x=keypoint.x / float(W), y=keypoint.y / float(H)
                                )
                                if keypoint is not None
                                else None
                                for keypoint in body.keypoints
                            ],
                            total_score=body.total_score,
                            total_parts=body.total_parts,
                        ),
                        left_hand,
                        right_hand,
                        face,
                    )
                )

            return results

    def detect_poses_dw(self, oriImg) -> List[HumanPoseResult]:
        """
        Detect poses in the given image using DW Pose:
        https://github.com/IDEA-Research/DWPose

        Args:
            oriImg (numpy.ndarray): The input image for pose detection.

        Returns:
            List[HumanPoseResult]: A list of HumanPoseResult objects containing the detected poses.
        """
        from .wholebody import Wholebody  # DW Pose

        with torch.no_grad():
            keypoints_info = self.dw_pose_estimation(oriImg.copy())
            return Wholebody.format_result(keypoints_info)

    def __call__(
        self,
        oriImg,
        include_body=True,
        include_hand=False,
        include_face=False,
        use_dw_pose=True,
    ):
        """
        Detect and draw poses in the given image.

        Args:
            oriImg (numpy.ndarray): The input image for pose detection and drawing.
            include_body (bool, optional): Whether to include body keypoints. Defaults to True.
            include_hand (bool, optional): Whether to include hand keypoints. Defaults to False.
            include_face (bool, optional): Whether to include face keypoints. Defaults to False.
            use_dw_pose (bool, optional): Whether to use DW pose detection algorithm. Defaults to False.

        Returns:
            numpy.ndarray: The image with detected and drawn poses.
        """
        H, W, _ = oriImg.shape
        animals = []
        poses = []
        if use_dw_pose:
            poses = self.detect_poses_dw(oriImg)
        else:
            poses = self.detect_poses(oriImg, include_hand, include_face)

        return draw_poses(
            poses,
            H,
            W,
            draw_body=include_body,
            draw_hand=include_hand,
            draw_face=include_face,
        )
