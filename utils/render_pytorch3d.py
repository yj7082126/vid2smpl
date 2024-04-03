from pathlib import Path
import json
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
from PIL import Image
import torch

from pytorch3d import renderer, structures
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes

SMPLX_SEM = "checkpoints/smplx/smplx_vert_segmentation.json"

def get_semantic_labels(part_segm_path, n_vertices, device="cuda:0"):
    part_segm = json.loads(Path(part_segm_path).read_bytes())
    segm_keys = [
        'hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg', 
        'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase', 
        'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm', 
        'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1'
    ]

    vertex_labels = np.zeros((n_vertices, 3))
    for part_idx, segm_key in enumerate(segm_keys):
        if segm_key == "head":
            v = part_segm["head"] + part_segm["eyeballs"]
        else:
            v = part_segm[segm_key]
        faces_segm = ((np.array([part_idx]) / 27) * 255.).clip(0, 255).astype(np.uint8)
        faces_segm = cv2.applyColorMap(faces_segm, cv2.COLORMAP_VIRIDIS)[:,:,::-1]
        vertex_labels[v] = faces_segm[0] / 255.
    faces_segm = ((np.array([12]) / 27) * 255.).clip(0, 255).astype(np.uint8)
    faces_segm = cv2.applyColorMap(faces_segm, cv2.COLORMAP_VIRIDIS)[:,:,::-1]
    vertex_labels[np.unique((vertex_labels == [0, 0, 0]).nonzero()[0])] = faces_segm[0] / 255.

    verts_rgb = torch.from_numpy(vertex_labels).unsqueeze(0).float().to(device)
    return verts_rgb

def get_mesh(vertices, faces, translation, use_semantic=True, color=None, z_rotate=True, device="cuda:0"):
    if use_semantic:
        verts_rgb = get_semantic_labels(SMPLX_SEM, vertices.shape[1], device=device)
    elif color is not None:
        verts_rgb = torch.ones_like(vertices) * color
    else:
        verts_rgb = torch.ones_like(vertices)
    texture_map = renderer.TexturesVertex(verts_rgb)

    vertices = vertices + translation[:, None, :]
    if z_rotate:
        rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
        rot = torch.from_numpy(rot).to(device).expand(1, 3, 3)
        vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)

    mesh = structures.Meshes(verts=vertices, faces=faces, textures=texture_map)
    return mesh

def calc_focal(bbox, init_focal=1000.0, init_size=256):
    focal = init_focal
    bbox_xywh = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]]
    focal = focal / init_size * bbox_xywh[2]
    return focal

def render_mesh(mesh, focal, H, W, backg=np.array([68/255., 1/255., 84/255.]), device="cuda:0"):
    cameras = renderer.PerspectiveCameras(
        focal_length=((
            2 * focal / min(H, W), 
            2 * focal / min(H, W)
        ),),
        device=device,
    )
    raster_settings = renderer.RasterizationSettings(
        image_size=(H, W),   # (H, W)
        blur_radius=np.log(1.0 / 1e-4) * 1e-7,
        faces_per_pixel=30,
        bin_size=-1
    )
    rasterizer = renderer.MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    lights = renderer.AmbientLights(device=device)
    blend_params = renderer.blending.BlendParams(1e-4, 1e-8, (backg))
    shader = renderer.SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params
    )

    fragments = rasterizer(mesh)
    result = shader(fragments, mesh)
    return fragments, result

def get_pil_img(array):
    array = array.detach().cpu().numpy()
    result = Image.fromarray((array * 255).astype(np.uint8).clip(0, 255))
    return result

def get_depth_mask(fragments):
    depth_image = fragments.zbuf
    depth_image = (depth_image[0, ..., 0]+1) 
    mask = depth_image > 0

    min_depth = depth_image[mask].min()
    max_depth = depth_image[mask].max()
    depth_image[mask] = (depth_image[mask] - min_depth) / (max_depth - min_depth)
    depth_image[mask] = (1.0 - depth_image[mask] * 0.5)
    depth_image = get_pil_img(depth_image)
    return mask, depth_image

def get_normal_img(fragments, mesh, mask):
    faces = mesh.faces_packed()  # (F, 3)
    vertex_normals = mesh.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]

    normal_image = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals #  torch.ones_like()
    )[0,:,:,0,:]

    min_normal = normal_image[mask].min()
    max_normal = normal_image[mask].max()
    normal_image[mask] = (normal_image[mask] - min_normal) / (max_normal - min_normal)
    normal_image[mask] = (1.0 - normal_image[mask])
    normal_image = get_pil_img(normal_image)
    return normal_image

def get_all_smpl_imgs(fragments, result, mesh):
    mask, depth_image = get_depth_mask(fragments)
    mask_image = get_pil_img(mask)
    normal_image = get_normal_img(fragments, mesh, mask)
    semantic_image = get_pil_img(result[0, :, :, :3])

    return {
        "depth" : depth_image, 
        "mask" : mask_image, 
        "normal" : normal_image, 
        "semantic_map" : semantic_image
    }