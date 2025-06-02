import torch
import numpy as np
import open3d as o3d
def get_points_and_colors(depths, rgbs, fx, fy, cx, cy):
    """
    Back-project a batch of depth and RGB images to 3D point clouds.
    
    Args:
        depths: Tensor of shape (B, H, W) representing depth in meters.
        rgbs: Tensor of shape (B, H, W, 3) representing RGB colors, range [0, 1] or [0, 255].
        fx, fy, cx, cy: camera intrinsics.
    
    Returns:
        points: Tensor of shape (B, H*W, 3) representing 3D points.
        colors: Tensor of shape (B, H*W, 3) representing RGB colors for each point.
    """
    B, H, W = depths.shape
    device = depths.device
    
    # Create meshgrid of pixel coordinates
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # shape (H, W)
    
    # Flatten pixel coordinates
    grid_u_flat = grid_u.reshape(-1)  # (H*W,)
    grid_v_flat = grid_v.reshape(-1)  # (H*W,)
    
    # Flatten depth and color
    z = depths.reshape(B, -1)  # (B, H*W)
    colors = rgbs.reshape(B, -1, 3)  # (B, H*W, 3)
    
    # Back-project to camera coordinates
    x = (grid_u_flat[None, :] - cx) * z / fx  # (B, H*W)
    y = (grid_v_flat[None, :] - cy) * z / fy  # (B, H*W)
    
    # Stack into point sets
    points = torch.stack((x, y, z), dim=-1)  # (B, H*W, 3)
    
    return points, colors


def iou_2d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute the 2D Intersection over Union (IoU) of two axis-aligned boxes.

    Args:
        box1: array_like of shape (4,), [xmin, ymin, xmax, ymax]
        box2: array_like of shape (4,), [xmin, ymin, xmax, ymax]

    Returns:
        IoU value (float) in [0.0, 1.0].
    """
    # Ensure inputs are numpy arrays
    b1 = np.array(box1, dtype=np.float64)
    b2 = np.array(box2, dtype=np.float64)

    # Intersection rectangle
    inter_xmin = max(b1[0], b2[0])
    inter_ymin = max(b1[1], b2[1])
    inter_xmax = min(b1[2], b2[2])
    inter_ymax = min(b1[3], b2[3])

    # Compute intersection width and height (clamp to zero if no overlap)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h  # area of overlap :contentReference[oaicite:0]{index=0}

    # Areas of the input boxes
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])

    # Union area
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0  # avoid division by zero :contentReference[oaicite:1]{index=1}

    # IoU is overlap divided by union :contentReference[oaicite:2]{index=2}
    return inter_area / union_area

def iou_3d(bbox1: o3d.geometry.AxisAlignedBoundingBox, bbox2: o3d.geometry.AxisAlignedBoundingBox) -> float:
    """
    Compute the 3D Intersection over Union (IoU) of two Open3D axis-aligned bounding boxes.
    
    Args:
        bbox1: An open3d.geometry.AxisAlignedBoundingBox instance.
        bbox2: An open3d.geometry.AxisAlignedBoundingBox instance.
    
    Returns:
        IoU value as a float in [0.0, 1.0].
    """
    # Get the min and max corner coordinates of each box
    min1 = np.array(bbox1.get_min_bound(), dtype=np.float64)
    max1 = np.array(bbox1.get_max_bound(), dtype=np.float64)
    min2 = np.array(bbox2.get_min_bound(), dtype=np.float64)
    max2 = np.array(bbox2.get_max_bound(), dtype=np.float64)

    # Compute the intersection box bounds
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    # Compute intersection dimensions (clamp to zero if no overlap)
    inter_dims = np.clip(inter_max - inter_min, a_min=0.0, a_max=None)
    inter_vol = np.prod(inter_dims)

    # Compute volumes of each box
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    # Compute union volume
    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0

    return float(inter_vol / union_vol)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert a 6-vector [x, y, z, roll, pitch, yaw] (radians) 
    into a 4×4 homogeneous transform.
    """
    pose = [float(p) for p in pose]  # ensure float type
    x, y, z, roll, pitch, yaw = pose

    # Rotation about X axis (roll)
    Rx = np.array([
        [1,            0,             0],
        [0,  np.cos(roll), -np.sin(roll)],
        [0,  np.sin(roll),  np.cos(roll)],
    ])

    # Rotation about Y axis (pitch)
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])

    # Rotation about Z axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1],
    ])

    # Combined rotation: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert a 4×4 homogeneous matrix back into a 6-vector
    [x, y, z, roll, pitch, yaw] with angles in radians.
    Assumes T[:3,:3] = Rz @ Ry @ Rx.
    """
    # translation
    x, y, z = T[:3, 3]

    # rotation matrix
    R = T[:3, :3]

    # recover pitch = asin(–R[2,0])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))

    # to avoid gimbal‐lock edge cases you could test cos(pitch)≈0
    # but for most cases:
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw  = np.arctan2(R[1, 0], R[0, 0])

    return np.array([x, y, z, roll, pitch, yaw])

def in_image(point: np.ndarray,
             obs_pose: np.ndarray,
             I: dict) -> bool:
    """
    Check if a 3D point in world coordinates projects into the image frame.

    Args:
        point: (3,) array giving the 3D point in world coordinates.
        obs_pose: (6,) array or list [x, y, z, roll, pitch, yaw] for camera pose in world.
        I: dict with camera intrinsics:
            - 'fx', 'fy': focal lengths
            - 'cx', 'cy': principal point
            - 'width', 'height': image size
            - 'model', 'coeffs': distortion model & parameters (ignored here if coeffs==0)

    Returns:
        True if the point projects within [0,width)×[0,height) and z_cam>0, else False.
    """
    # 1) build the camera-to-world transform, then invert to get world→camera
    T_cam2world = pose_to_matrix(obs_pose)
    T_world2cam = np.linalg.inv(T_cam2world)

    # 2) homogeneous world point → camera frame
    p_w = np.ones(4)
    p_w[:3] = point
    p_cam = T_world2cam @ p_w
    x_cam, y_cam, z_cam = p_cam[:3]

    # 3) must be in front of camera
    if z_cam <= 0:
        return False

    # 4) pinhole projection (no distortion since coeffs are zero)
    u = I['fx'] * (x_cam / z_cam) + I['cx']
    v = I['fy'] * (y_cam / z_cam) + I['cy']

    # 5) check image bounds
    in_x = (0.0 <= u) and (u < I['width'])
    in_y = (0.0 <= v) and (v < I['height'])
    return in_x and in_y


def nms(boxes, scores, iou_threshold, extra_data_lists=None, three_d=False):
    if extra_data_lists is None:
        extra_data_lists = []
    iou_func = iou_3d if three_d else iou_2d

    # sanity-check extra lists
    n_extra = len(extra_data_lists)
    for i, ed in enumerate(extra_data_lists):
        if len(ed) != len(boxes):
            raise ValueError(f"extra_data_lists[{i}] has length {len(ed)}, "
                             f"but you have {len(boxes)} boxes")

    # pack everything together: (score, box, [extra1, extra2, ...])
    items = []
    for idx, (b, s) in enumerate(zip(boxes, scores)):
        extras = [extra_data_lists[j][idx] for j in range(n_extra)]
        items.append((s, b, extras))

    # sort by score desc
    items.sort(key=lambda x: x[0], reverse=True)

    kept = []
    while items:
        # pop highest‐score
        curr_score, curr_box, curr_extras = items.pop(0)

        if not items:
            # no more to compare → keep as is
            kept.append((curr_score, curr_box, curr_extras))
            break

        # compute IoU vs. all remaining
        ious = np.array([iou_func(curr_box, b2) for (_, b2, _) in items])
        # find which ones exceed threshold
        discard_idx = np.where(ious >= iou_threshold)[0]

        # sum up the *scores* of those to-be-discarded
        discarded_scores = sum(items[i][0] for i in discard_idx)
        curr_score += discarded_scores

        # rebuild items, skipping the discarded indices
        items = [item for i, item in enumerate(items) if i not in discard_idx]

        kept.append((curr_score, curr_box, curr_extras))

    # unzip results
    kept_scores = [s for s, _, _ in kept]
    kept_boxes  = [b for _, b, _ in kept]
    kept_extras = [
        [ext_list[j] for _, _, ext_list in kept]
        for j in range(n_extra)
    ]

    return kept_boxes, kept_scores, kept_extras
