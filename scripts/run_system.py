def load_data(data_path: str, max_frames: int | None = None):
    """
    Load RGB-D dataset from disk.

    Supported formats:
    - RealSense capture (meta.json + rgb/*.png + depth/*.png)
    - Legacy format (intrinsics.json + rgb/*.png + depth/*.png or depth/*.npy)

    Returns:
        rgb_images: list[np.ndarray]  (H,W,3) RGB
        depth_images: list[np.ndarray] (H,W) depth in METERS
        poses: None
        intrinsics: dict {fx, fy, cx, cy}
    """
    if data_path is None:
        raise ValueError("--data is required unless --dry-run is set")

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    rgb_dir = data_path / "rgb"
    depth_dir = data_path / "depth"

    if not rgb_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError("Expected subfolders: rgb/ and depth/")

    # ---------------------------
    # Load intrinsics + depth scale
    # ---------------------------
    intrinsics: Dict[str, float] | None = None

    # Default depth scale:
    # - RealSense PNG depth is typically uint16 in millimeters -> 0.001 m/unit
    # - If meta.json provides depth_scale, we trust it.
    depth_scale = 0.001

    meta_path = data_path / "meta.json"
    intr_path = data_path / "intrinsics.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        intr = meta.get("intrinsics", {})
        intrinsics = {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        }

        # meta depth_scale is expected to be "meters per unit"
        if "depth_scale" in meta and meta["depth_scale"] is not None:
            depth_scale = float(meta["depth_scale"])

    elif intr_path.exists():
        with open(intr_path, "r", encoding="utf-8") as f:
            intr = json.load(f)

        # Be tolerant: cast everything to float
        intrinsics = {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        }

        # If intrinsics.json exists but no meta.json, we still assume RealSense-style depth PNGs
        # (uint16 in mm) unless depth is provided as .npy already in meters.
        depth_scale = float(intr.get("depth_scale", depth_scale))

    else:
        raise FileNotFoundError(
            f"Missing intrinsics. Expected one of:\n"
            f"  - {meta_path}\n"
            f"  - {intr_path}"
        )

    # ---------------------------
    # Load images
    # ---------------------------
    rgb_files = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.jpeg")))
    depth_files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.npy")))

    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in: {rgb_dir}")
    if not depth_files:
        raise FileNotFoundError(f"No depth files found in: {depth_dir}")

    if max_frames:
        rgb_files = rgb_files[:max_frames]
        depth_files = depth_files[:max_frames]

    # Pair conservatively in case counts differ
    n = min(len(rgb_files), len(depth_files))
    if n == 0:
        raise FileNotFoundError("No paired RGB/Depth frames found.")
    if len(rgb_files) != len(depth_files):
        print(f"[WARN] rgb frames={len(rgb_files)} depth frames={len(depth_files)}; using first {n} pairs")

    rgb_images: List[np.ndarray] = []
    depth_images: List[np.ndarray] = []

    for rgb_f, depth_f in zip(rgb_files[:n], depth_files[:n]):
        rgb = cv2.imread(str(rgb_f), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_f}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_images.append(rgb)

        if depth_f.suffix == ".png":
            depth_raw = cv2.imread(str(depth_f), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise RuntimeError(f"Failed to read depth image: {depth_f}")

            # Convert to meters. If uint16 -> multiply by depth_scale.
            # If already float (rare), keep as-is.
            if depth_raw.dtype == np.uint16 or depth_raw.dtype == np.int16:
                depth_m = depth_raw.astype(np.float32) * depth_scale
            else:
                depth_m = depth_raw.astype(np.float32)

        else:
            # .npy expected to be already in meters
            depth_m = np.load(depth_f).astype(np.float32)

        # Basic sanitization
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_images.append(depth_m)

    print(f"[OK] Loaded {len(rgb_images)} RGB-D frames from {data_path}")
    print(f"[OK] Intrinsics: {intrinsics}")
    print(f"[OK] Depth scale (m/unit): {depth_scale}")

    poses = None
    return rgb_images, depth_images, poses, intrinsics