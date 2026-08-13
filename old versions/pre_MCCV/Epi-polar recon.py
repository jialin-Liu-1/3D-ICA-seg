

import os
import re
import glob
import numpy as np
import astra
import traceback


# ============================================================
# Main folders
# ============================================================

base_folder = r"Y:\Projects\MC-DSA\1E-08"
save_path =r'Z:\Users\Jialin'
save_folder = os.path.join(save_path, "Epipolar_recons")
os.makedirs(save_folder, exist_ok=True)


# ============================================================
# Parameters
# ============================================================

NX = 512
NZ = 512
image_index = 1
eps = 1e-8

epi_projection_indices = [15, 60]

sod_mm = 350.0
oid_mm = 350.0
detector_size_mm = 280.0
angle_step_deg = 1.944

recon_size = 256
recon_voxel_size_mm = 0.5

short_scan = True
angle_sign = 1
flip_u = False
flip_v = False

skip_existing = False   # change to True if you do not want to overwrite existing recon files


# ============================================================
# Helper functions
# ============================================================

def get_projection_number(path):
    """
    Handles both:
        something.dat_0049.raw  -> projection 49
        something.dat.raw       -> projection 0
    """
    name = os.path.basename(path)

    match = re.search(r"dat_(\d+)\.raw$", name)
    if match is not None:
        return int(match.group(1))

    if name.endswith(".dat.raw"):
        return 0

    raise ValueError(f"Could not extract projection number from: {name}")


def find_projection_files(folder):
    """
    Finds MC-GPU projection raw files.

    Supports:
        mask_water_image.dat.raw
        contrast_vessel_image.dat_0000.raw
        contrast_vessel_image.dat_0001.raw
    """
    all_raw = glob.glob(os.path.join(folder, "*.raw"))

    projection_files = []

    for f in all_raw:
        name = os.path.basename(f)

        if re.search(r"dat_(\d+)\.raw$", name):
            projection_files.append(f)

        elif name.endswith(".dat.raw"):
            projection_files.append(f)

    projection_files = sorted(projection_files, key=get_projection_number)

    if len(projection_files) == 0:
        raise FileNotFoundError(
            f"No MC-GPU projection raw files found in:\n{folder}\n"
            "Expected files like something.dat.raw or something.dat_0000.raw"
        )

    return projection_files


def load_mcgpu_projection(path, nx=512, nz=512, image_index=1):
    data = np.fromfile(path, dtype=np.float32)

    expected_one = nx * nz
    expected_five = 5 * nx * nz

    if data.size == expected_five:
        img5 = data.reshape(5, nz, nx)
        return img5[image_index].astype(np.float32)

    elif data.size == expected_one:
        return data.reshape(nz, nx).astype(np.float32)

    else:
        raise ValueError(
            f"Unexpected file size for {path}\n"
            f"Got {data.size} values.\n"
            f"Expected {expected_one} or {expected_five}."
        )


def load_projection_stack(folder, nx=512, nz=512, image_index=1):
    files = find_projection_files(folder)
    numbers = [get_projection_number(f) for f in files]

    stack = []

    for f in files:
        img = load_mcgpu_projection(
            f,
            nx=nx,
            nz=nz,
            image_index=image_index
        )
        stack.append(img)

    stack = np.stack(stack, axis=0).astype(np.float32)

    return stack, numbers, files


def find_folder_case_insensitive(parent_folder, target_name):
    """
    Finds folder named target_name inside parent_folder.
    First checks direct children, then searches recursively.

    Example:
        target_name = "Mask"
        target_name = "contrast"
    """

    target_lower = target_name.lower()

    # First: direct child search
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path) and item.lower() == target_lower:
            return item_path

    # Second: recursive search
    matches = []

    for root, dirs, files in os.walk(parent_folder):
        for d in dirs:
            if d.lower() == target_lower:
                matches.append(os.path.join(root, d))

    if len(matches) == 0:
        return None

    # Choose shortest path first, usually the correct one
    matches = sorted(matches, key=lambda x: len(x))
    return matches[0]


def make_dsa_stack(mask_folder, contrast_folder):
    """
    Loads mask and contrast projections, matches them, and creates DSA stack.
    """

    mask_stack, mask_numbers, mask_files = load_projection_stack(
        mask_folder,
        nx=NX,
        nz=NZ,
        image_index=image_index
    )

    contrast_stack, contrast_numbers, contrast_files = load_projection_stack(
        contrast_folder,
        nx=NX,
        nz=NZ,
        image_index=image_index
    )

    print("Mask stack shape:", mask_stack.shape)
    print("Contrast stack shape:", contrast_stack.shape)

    if mask_stack.shape[0] == 1:
        print("Only one mask projection found. Reusing it for all contrast projections.")

        mask_for_dsa = np.repeat(mask_stack, contrast_stack.shape[0], axis=0)
        dsa_numbers = contrast_numbers

    elif mask_stack.shape[0] == contrast_stack.shape[0]:
        print("Same number of mask and contrast projections found. Matching by order.")

        mask_for_dsa = mask_stack
        dsa_numbers = contrast_numbers

    else:
        print("Different number of mask and contrast projections. Matching by projection number.")

        mask_map = {n: i for i, n in enumerate(mask_numbers)}
        contrast_map = {n: i for i, n in enumerate(contrast_numbers)}

        common_numbers = sorted(set(mask_numbers).intersection(set(contrast_numbers)))

        if len(common_numbers) == 0:
            raise ValueError("No matching projection numbers between mask and contrast.")

        mask_for_dsa = np.stack(
            [mask_stack[mask_map[n]] for n in common_numbers],
            axis=0
        ).astype(np.float32)

        contrast_stack = np.stack(
            [contrast_stack[contrast_map[n]] for n in common_numbers],
            axis=0
        ).astype(np.float32)

        dsa_numbers = common_numbers

    dsa_stack = np.log((mask_for_dsa + eps) / (contrast_stack + eps))
    dsa_stack = dsa_stack.astype(np.float32)
    dsa_stack[dsa_stack < 0] = 0

    return dsa_stack, dsa_numbers


def astra_cone_fdk_from_mcgpu(
    projections,
    sod_mm=350.0,
    oid_mm=350.0,
    detector_size_mm=280.0,
    angle_step_deg=1.944,
    recon_size=256,
    recon_voxel_size_mm=0.5,
    short_scan=True,
    angle_sign=1,
    flip_u=False,
    flip_v=False,
):
    """
    projections shape: [num_angles, det_rows, det_cols]
    Example: [108, 512, 512]
    """

    projections = projections.astype(np.float32)

    if flip_u:
        projections = projections[:, :, ::-1]

    if flip_v:
        projections = projections[:, ::-1, :]

    n_angles, det_rows, det_cols = projections.shape

    detector_pixel_size_mm = detector_size_mm / det_cols

    print("Projection shape:", projections.shape)
    print("Detector pixel size [mm]:", detector_pixel_size_mm)
    print("SOD [mm]:", sod_mm)
    print("OID [mm]:", oid_mm)
    print("SID [mm]:", sod_mm + oid_mm)

    angles = angle_sign * np.arange(n_angles, dtype=np.float32) * np.deg2rad(angle_step_deg)

    # ASTRA expects projection data as [det_rows, angles, det_cols]
    sino = np.transpose(projections, (1, 0, 2)).copy()

    proj_geom = astra.create_proj_geom(
        "cone",
        detector_pixel_size_mm,
        detector_pixel_size_mm,
        det_rows,
        det_cols,
        angles,
        sod_mm,
        oid_mm,
    )

    fov_mm = recon_size * recon_voxel_size_mm
    half_fov = fov_mm / 2.0

    vol_geom = astra.create_vol_geom(
        recon_size,
        recon_size,
        recon_size,
        -half_fov,
        half_fov,
        -half_fov,
        half_fov,
        -half_fov,
        half_fov,
    )

    proj_id = astra.data3d.create("-proj3d", proj_geom, sino)
    recon_id = astra.data3d.create("-vol", vol_geom)

    alg_id = None

    try:
        cfg = astra.astra_dict("FDK_CUDA")
        cfg["ProjectionDataId"] = proj_id
        cfg["ReconstructionDataId"] = recon_id
        cfg["option"] = {"ShortScan": short_scan}

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        recon = astra.data3d.get(recon_id).astype(np.float32)
        recon[recon < 0] = 0

    finally:
        if alg_id is not None:
            astra.algorithm.delete(alg_id)

        astra.data3d.delete(proj_id)
        astra.data3d.delete(recon_id)

    return recon


# ============================================================
# Main batch loop
# ============================================================

case_folders = [
    os.path.join(base_folder, d)
    for d in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, d))
]

case_folders = sorted(case_folders)

print("Number of folders found:", len(case_folders))
print("Base folder:", base_folder)
print("Save folder:", save_folder)


for case_folder in case_folders:

    case_name = os.path.basename(case_folder)

    # Skip output folder itself
    if case_name.lower() in ["epipolar_recons", "recons", "recon"]:
        continue

    print("\n" + "=" * 80)
    print("Processing case:", case_name)
    print("Case folder:", case_folder)

    output_path = os.path.join(save_folder, f"recon_{case_name}.raw")

    if skip_existing and os.path.exists(output_path):
        print("Output already exists. Skipping:")
        print(output_path)
        continue

    try:
        mask_folder = find_folder_case_insensitive(case_folder, "Mask")
        contrast_folder = find_folder_case_insensitive(case_folder, "contrast")

        if mask_folder is None:
            print("No Mask folder found. Skipping this case.")
            continue

        if contrast_folder is None:
            print("No contrast folder found. Skipping this case.")
            continue

        print("Mask folder:", mask_folder)
        print("Contrast folder:", contrast_folder)

        # ----------------------------------------------------
        # Load projections and create DSA
        # ----------------------------------------------------

        dsa_stack, dsa_numbers = make_dsa_stack(mask_folder, contrast_folder)

        print("DSA stack shape:", dsa_stack.shape)
        print("First DSA projection numbers:", dsa_numbers[:10])
        print("Last DSA projection numbers:", dsa_numbers[-10:])

        # ----------------------------------------------------
        # Create epipolar DSA stack
        # Only keep selected views
        # ----------------------------------------------------

        max_index = max(epi_projection_indices)

        if dsa_stack.shape[0] <= max_index:
            raise ValueError(
                f"This case has only {dsa_stack.shape[0]} projections, "
                f"but requested projection index {max_index}."
            )

        epi_dsa = np.zeros_like(dsa_stack, dtype=np.float32)

        for idx in epi_projection_indices:
            epi_dsa[idx, :, :] = dsa_stack[idx, :, :]

        # ----------------------------------------------------
        # Reconstruct
        # ----------------------------------------------------

        recon_c = astra_cone_fdk_from_mcgpu(
            epi_dsa,
            sod_mm=sod_mm,
            oid_mm=oid_mm,
            detector_size_mm=detector_size_mm,
            angle_step_deg=angle_step_deg,
            recon_size=recon_size,
            recon_voxel_size_mm=recon_voxel_size_mm,
            short_scan=short_scan,
            angle_sign=angle_sign,
            flip_u=flip_u,
            flip_v=flip_v,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        recon_c.tofile(output_path)

        print("Saved reconstruction:")
        print(output_path)
        print("Recon shape:", recon_c.shape)
        print("Recon min/max:", np.min(recon_c), np.max(recon_c))

    except Exception as e:
        print("FAILED case:", case_name)
        print("Error:", str(e))
        traceback.print_exc()
        continue

print("\nDone.")