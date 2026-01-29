import numpy as np

def project_mask_to_3d(object_mask: np.ndarray, depth_image: np.ndarray, camera_intrinsics: dict):
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    if object_mask.dtype == np.uint8:
        object_mask = object_mask.astype(bool)

    v_coords, u_coords = np.where(object_mask)
    if len(v_coords) == 0:
        return None

    depths = depth_image[v_coords, u_coords].astype(np.float32)
    valid = (depths > 0.1) & (depths < 10.0)
    if not np.any(valid):
        return None

    u = u_coords[valid].astype(np.float32)
    v = v_coords[valid].astype(np.float32)
    Z = depths[valid]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points_3d = np.stack([X, Y, Z], axis=1)

    centroid = np.mean(points_3d, axis=0)
    min_bounds = np.min(points_3d, axis=0)
    max_bounds = np.max(points_3d, axis=0)
    dimensions = max_bounds - min_bounds

    return {
        "points_3d": points_3d,
        "centroid": centroid,
        "min_bounds": min_bounds,
        "max_bounds": max_bounds,
        "dimensions": dimensions,
        "num_points": int(points_3d.shape[0]),
    }