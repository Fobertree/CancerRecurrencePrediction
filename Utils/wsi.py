import numpy as np
import os
import tifffile as tiff
import itertools
import pandas as pd
import torch
from torch_geometric.data import Data
import pandas as pd

import openslide
import cv2
from PIL import Image
import logging
from tqdm import tqdm

import glob

# logger
logger = logging.Logger("wsi")
file_handler = logging.FileHandler("Logs/wsi.log", mode='a')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

'''
Preprocessing:
- Load .tiff image (not of standard size)
- Find way to standardize image into tensor: TODO @aliu
'''
def load_images(
    directory_path: str,
    metadata_path: str,
    output_dir: str = "dinov2_patches",
    patch_size: int = 256,
    num_patches: int = 100,
    tissue_threshold: float = 0.8,
    sampling: str = "random",
    threshold: int | None = None,
    return_images: bool = False,
):
    """
    Load preprocessed JPEG images (downsampled WSI) and extract patches.

    Args:
        directory_path: folder containing .jpeg images
        metadata_path: CSV file containing 'svs_name' and 'label'
        output_dir: folder to save generated patches
        patch_size: patch size in pixels
        num_patches: max number of patches per image
        tissue_threshold: fraction of pixels considered tissue
        sampling: "random" or "grid"
        threshold: max number of images to process
        return_images: if True, also return numpy arrays of patches

    Returns:
        list of tuples: [(patch_files, patch_centers, metadata_dict), ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    res = []

    # Load metadata
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_path}")

    if 'svs_name' not in df.columns:
        raise ValueError("Metadata CSV must contain 'svs_name' column")
    df['svs_stem'] = (
        df['svs_name'].astype(str)
        .str.lower().str.strip()
        .str.replace('.svs','').str.replace('.tif','')
    )

    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".jpeg")]
    if threshold is not None:
        image_files = image_files[:threshold]

    for img_file in tqdm(image_files, desc="Loading images"):
        img_path = os.path.join(directory_path, img_file)
        img_stem = os.path.splitext(img_file)[0].lower()

        # Match metadata
        meta_row = df[df['svs_stem'] == img_stem]
        if meta_row.empty:
            print(f"Warning: no metadata found for {img_file}")
            metadata = {}
        else:
            metadata = meta_row.iloc[0].to_dict()

        # Open image
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)
            height, width, _ = img_np.shape
        except Exception as e:
            print(f"Error opening {img_file}: {e}")
            continue

        # Generate thumbnail-based tissue mask for patch selection
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        _, s_chan, _ = cv2.split(img_hsv)
        try:
            _, tissue_mask = cv2.threshold(s_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            tissue_mask = (s_chan > 10).astype(np.uint8) * 255
        tissue_coords = np.argwhere(tissue_mask > 0)
        if tissue_coords.size == 0:
            print(f"No tissue found in {img_file}. Skipping.")
            continue

        # Sampling patches
        saved_files = []
        centers = []
        images_out = []

        count = 0
        max_tries = num_patches * 10
        tries = 0
        while count < num_patches and tries < max_tries:
            tries += 1
            idx = np.random.randint(len(tissue_coords))
            y, x = tissue_coords[idx]

            # make sure patch is fully inside image
            if x + patch_size > width or y + patch_size > height:
                continue

            patch = img_np[y:y+patch_size, x:x+patch_size, :]
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            s_patch = patch_hsv[:, :, 1]
            tissue_percentage = np.sum(s_patch > 0) / s_patch.size
            if tissue_percentage < tissue_threshold:
                continue

            patch_fname = f"{img_stem}_patch_{count:04d}_{x}_{y}.png"
            patch_path = os.path.join(output_dir, patch_fname)
            Image.fromarray(patch).save(patch_path)
            saved_files.append(patch_path)
            centers.append((x + patch_size // 2, y + patch_size // 2))
            if return_images:
                images_out.append(patch)
            count += 1

        if count == 0:
            print(f"No tissue-rich patches extracted from {img_file}.")
            continue

        if return_images:
            res.append((saved_files, centers, images_out, metadata))
        else:
            res.append((saved_files, centers, metadata))

    return res

# def load_images(directory_path : str, metadata_path: str, threshold : int | None = None):
#     '''
#     Load image instead of slide
#     '''
#     res = []

#     metadata_df = pd.read_csv(metadata_path)
#     # Don't think we need concurrency here
#     for row in tqdm(metadata_df):
#         fname = row['svs']
#         try:
#             img = Image.open(os.path.join(directory_path, f"{fname}.jpeg"))
#         except FileNotFoundError:
#             print(f"File not found: {os.path.join(directory_path, f"{fname}.jpeg")}")
#             continue
#         img_array = np.asarray(img)
#         # slide = openslide.OpenSlide(os.path.join(directory_path, f"{svs_name}.tiff"))
#         metadata = row.drop(['svs', 'label']).values()
#         label = row['label']

#         res.append((img_array, metadata, label))
        

#         if len(res) >= threshold:
#             break

#     # (slide, metadata feature-vec, label)
#     return res

def load_wsi(
    directory_path: str,
    metadata_path: str,
    output_dir: str = "dinov2_patches",
    threshold: int | None = None,
    patch_size: int = 256,
    num_patches: int = 100,
    magnification: int = 20,
    tissue_threshold: float = 0.8,
    sampling: str = "random",
):
    """
    Load WSI files, extract patches, and return them along with coordinates and metadata.

    Returns:
        list of tuples: [(patch_files, patch_centers, metadata_dict), ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    res = []

    # Load metadata CSV
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_path}")

    # Standardize CSV column for matching
    if 'svs_name' not in df.columns:
        raise ValueError("Metadata CSV must contain 'svs_name' column")
    df['svs_stem'] = (
        df['svs_name'].astype(str)
        .str.lower().str.strip()
        .str.replace('.svs','').str.replace('.tif','')
    )

    # Collect slide files
    wsi_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(directory_path)
        for f in files
        if f.lower().endswith((".tif", ".svs"))
    ]

    if threshold is not None:
        wsi_files = wsi_files[:threshold]

    for slide_path in tqdm(wsi_files, desc="Loading WSIs"):
        slide_name = os.path.basename(slide_path).lower()
        slide_stem = os.path.splitext(slide_name)[0].strip()

        # Match metadata
        meta_row = df[df['svs_stem'] == slide_stem]
        if meta_row.empty:
            print(f"Warning: no metadata found for {slide_name}")
            metadata = {}
            continue
        else:
            metadata = meta_row.iloc[0].to_dict()

        try:
            # Extract patches from slide
            patch_files, patch_centers = generate_wsi_patches(
                slide_path=slide_path,
                output_dir=os.path.join(output_dir, slide_stem),
                patch_size=patch_size,
                num_patches=num_patches,
                magnification=magnification,
                tissue_threshold=tissue_threshold,
                sampling=sampling,
            )

            # Add slide properties to metadata
            slide_obj = openslide.OpenSlide(slide_path)
            metadata.update({
                "slide_dimensions": slide_obj.dimensions,
                "level_count": slide_obj.level_count,
                "openslide_properties": dict(slide_obj.properties)
            })
            slide_obj.close()

            res.append((patch_files, patch_centers, metadata))

        except Exception as e:
            print(f"Error processing slide {slide_name}: {e}")

    return res

# def load_wsi(directory_path: str, metadata_path: str, threshold: int | None = None):
#     '''
#     Loads (up to threshold) wsi from path and samples random patches
    
#     We call this in pre-processing
#     - Take the output of this function and generate a graph from it to pass into GNN blocks
#     '''
#     os.makedirs("dinov2_patches", exist_ok=True)
#     res = []
#     # TODO @thomas: match the logic of this to the dataset structure
    
#     # process at most threshold images
#     # load metadata
#     try:
#         df = pd.read_csv(metadata_path)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Metadata CSV not found at {metadata_path}")

#     # collect image files
#     wsi_files = []
#     for root, _, files in os.walk(directory_path):
#         for f in files:
#             if f.lower().endswith((".tif", ".svs")):
#                 wsi_files.append(os.path.join(root, f))

#     if threshold is not None:
#         wsi_files = wsi_files[:threshold]

#     # process each WSI
#     for slide_path in tqdm(wsi_files, desc="Loading WSIs"):
#         slide_name = os.path.basename(slide_path).lower()

#         # Match metadata row for this slide (case-insensitive)
#         # Standardize slide filename (no extension, lowercase, stripped)
#         slide_stem = os.path.splitext(os.path.basename(slide_path))[0].lower().strip()

#         # Standardize CSV column
#         df['svs_stem'] = df['svs_name'].astype(str).str.lower().str.strip().str.replace('.svs','').str.replace('.tif','')

#         # Match
#         meta_row = df[df['svs_stem'] == slide_stem]
#         if meta_row.empty:
#             logger.error(f"No metadata found for slide {slide_name}")
#             print(f"Warning: no metadata found for {slide_name}")
#             metadata = {}
#         else:
#             metadata = meta_row.iloc[0].to_dict()

#         try:
#             slide = openslide.OpenSlide(slide_path)
#             patches, coords = sample_random_wsi_patches(slide)

#             # Add internal OpenSlide info too
#             metadata.update({
#                 "slide_dimensions": slide.dimensions,
#                 "level_count": slide.level_count,
#                 "openslide_properties": dict(slide.properties)
#             })

#             res.append((patches, coords, metadata))

#         except openslide.OpenSlideError:
#             print(f"Error opening slide {slide_path}")
#             logger.error(f"OpenSlideError::load_wsi: slide path: {slide_path}")
#         finally:
#             if 'slide' in locals():
#                 slide.close()
    
#     return res

# def full_patch_wsi(slide : openslide.OpenSlide,
#                    magnification=20,
#                    tissue_threshold=0.8,
#                    patch_size=4):
#     '''
#     Full patching above threshold

#     Also builds and returns the graph
#     '''

#     try:
#         # Read the full-resolution slide
#         full_res = slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB")

#         # Downsample manually to target magnification
#         target_scale = 1 / magnification
#         new_size = (int(full_res.width * target_scale), int(full_res.height * target_scale))
#         downsamp_img = full_res.resize(new_size, Image.Resampling.LANCZOS)
#         downsamp_img = np.array(downsamp_img)

#         print(f"Full slide size: {full_res.size}, Downsampled size: {downsamp_img.shape[:2]}")

#     except KeyError:
#         print("Warning: Could not determine magnification. Using slide level 1.")
#         downsamp_img = np.array(slide.read_region((0, 0), 1, slide.level_dimensions[1]).convert("RGB"))
    
#     height,width,_c = downsamp_img.shape
#     patches, coords = [], []

#     # --- Sequentially extract patches ---
#     for y in range(0, height, patch_size):
#         for x in range(0, width, patch_size):
#             patch = downsamp_img[y:y+patch_size, x:x+patch_size, :]
#             if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
#                 continue  # skip incomplete border patches

#             gray = np.mean(patch, axis=2) # mean rgb channel values
#             tissue_ratio = np.mean(gray < 220)

#             #TODO: filter out overly green/blue/coffee-brown patches

#             if tissue_ratio >= tissue_threshold:
#                 feature = np.mean(patch.reshape(-1, 3), axis=0)  # simple color feature
#                 patches.append(feature)
#                 coords.append((x, y))

#     if not patches:
#         print("No tissue-rich patches found!")
#         return None

#     patches = np.array(patches, dtype=np.float32)
#     coords = np.array(coords, dtype=np.int32)

#     # --- Build adjacency (8-neighbor) ---
#     coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
#     edges = []

#     offsets = [
#         (-patch_size, -patch_size), (-patch_size, 0), (-patch_size, patch_size),
#         (0, -patch_size),                     (0, patch_size),
#         (patch_size, -patch_size),  (patch_size, 0),  (patch_size, patch_size)
#     ]

#     for i, (x, y) in enumerate(coords):
#         for dx, dy in offsets:
#             neighbor = (x + dx, y + dy)
#             if neighbor in coord_to_idx:
#                 j = coord_to_idx[neighbor]
#                 edges.append((i, j))

#     # Convert to tensor
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     x = torch.tensor(patches, dtype=torch.float)
#     pos = torch.tensor(coords, dtype=torch.float)

#     data = Data(x=x, edge_index=edge_index, pos=pos)

#     # Normalize positions
#     data.pos = (data.pos - data.pos.mean(dim=0)) / data.pos.std(dim=0)    

#     print(f"Graph built: {data.num_nodes} nodes, {data.num_edges} edges")

#     return data

def sample_random_wsi_patches(
        slide : openslide.OpenSlide,
        output_dir="dinov2_patches",
        patch_size=256,
        num_patches=100,
        magnification=20,
        tissue_threshold=0.8
    ):
    res = []
    patch_centers = []
    centers = []
    # Find the best level for the target magnification
    try:
        # MPP: microns per pixel
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))  # default 0.25 µm/px if missing
        downsample = mpp_x * (magnification / 20)
        level = slide.get_best_level_for_downsample(downsample)

    except KeyError:
        print("Warning: Could not determine best level from MPP. Using slide level 1.")
        level = 1

    downsample = int(slide.level_downsamples[level])
    level_dims = slide.level_dimensions[level]
    downsamples_img_pil = slide.read_region((0,0), 4., level_dims)

    # Use a low-resolution thumbnail to create a tissue mask
    thumbnail = slide.get_thumbnail((level_dims[0], level_dims[1]))
    thumbnail_np = np.array(thumbnail.convert("RGB"))
    thumbnail_hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)

    # Threshold the thumbnail image to find tissue regions
    h, s, v = cv2.split(thumbnail_hsv)
    _, tissue_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the coordinates of valid (non-background) tissue regions
    tissue_coords = np.argwhere(tissue_mask > 0)
    if not tissue_coords.any():
        print("No tissue found in the slide. Aborting.")
        return

    # Sample random patch locations within the tissue region
    patch_count = 0
    while patch_count < num_patches:
        # Pick a random point from the tissue coordinates
        random_index = np.random.randint(len(tissue_coords))
        thumb_y, thumb_x = tissue_coords[random_index]

        # Scale (upper-left) coordinates to the full resolution
        x_origin = int(thumb_x * downsample)
        y_origin = int(thumb_y * downsample)

        # Read the patch from the WSI
        patch_pil = slide.read_region(
            (x_origin, y_origin), level, (patch_size, patch_size)
        )
        patch_np = np.array(patch_pil.convert("RGB"))

        # Optional: Further filter patches to ensure sufficient tissue content
        patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
        s_patch = cv2.split(patch_hsv)[1]
        
        # Calculate tissue percentage
        tissue_pixels = np.sum(s_patch > 0)
        total_pixels = s_patch.size
        tissue_percentage = tissue_pixels / total_pixels
        
        # Save the patch if it meets the tissue threshold
        if tissue_percentage > tissue_threshold:
            img_numpy = np.array(patch_pil)
            H, W = img_numpy.shape
            res.append(img_numpy)
            # up_left_corners.append((x_origin, y_origin))
            patch_centers.append((x_origin + H // 2, y_origin + W // 2))
            patch_filename = f"patch_{patch_count:04d}_{x_origin}_{y_origin}.png"
            patch_pil.save(os.path.join(output_dir, patch_filename))
            patch_count += 1
            if patch_count % 10 == 0:
                # could replace this with a tqdm
                print(f"Extracted {patch_count} patches...")
    
    return res, patch_centers


def generate_wsi_patches(
    slide_path: str,
    output_dir: str = "dinov2_patches",
    patch_size: int = 256,
    num_patches: int = 100,
    magnification: int = 20,
    tissue_threshold: float = 0.8,
    sampling: str = "random",   # "random" or "grid"
    grid_stride: int | None = None, # if grid sampling, stride (default = patch_size)
    seed: int | None = None,
    return_images: bool = False, # if True return list of numpy arrays and centers
    thumbnail_max_dim: int | None = None, # size for slide.get_thumbnail (if None uses level dims)
):
    """
    Generate and save patches from a single WSI file using OpenSlide.
    - slide_path: path to .svs/.tif WSI
    - output_dir: where patch pngs will be written (created if missing)
    - patch_size: requested patch size in pixels at the chosen level
    - num_patches: number of patches to extract (for "grid" this is treated as max)
    - magnification: target magnification (20 => no downsampling relative to level0)
    NOTE: mapping magnification -> level is approximate unless MPP/metadata is available.
    - tissue_threshold: fraction of pixels (by saturation) considered tissue (0..1)
    - sampling: "random" or "grid"
    - grid_stride: spacing for grid; default = patch_size (no overlap)
    - seed: reproducible random seed
    - return_images: returns (filenames, centers, optionally images)
    - thumbnail_max_dim: when building thumbnail, you can control size; if None uses level dims
    """
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # --- Skip if already patched ---
    existing_patches = glob.glob(os.path.join(output_dir, "patch_*.png"))
    coords_path = os.path.join(output_dir, "coords.npy")

    if len(existing_patches) > 0:
        print(f"Skipping {os.path.basename(slide_path)} — {len(existing_patches)} patches already exist.")
        if os.path.exists(coords_path):
            try:
                centers = np.load(coords_path, allow_pickle=True)
            except Exception:
                centers = []
        else:
            centers = []
        if return_images:
            return existing_patches, centers, []
        else:
            return existing_patches, centers


    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open slide {slide_path}: {e}")

    # --- Choose an integer level that best matches the requested magnification ---
    # Heuristic: assume level 0 is ~20x. Choose level whose downsample is closest to (magnification / 20).
    # (If slides contain better metadata (MPP or AppMag) you could compute a more accurate mapping.)
    try:
        level_downsamples = slide.level_downsamples  # list of floats
        target_downsample = magnification / 20.0
        # find level minimizing |level_downsample - target_downsample|
        level = int(np.argmin([abs(float(d) - target_downsample) for d in level_downsamples]))
    except Exception:
        level = 1  # fallback
    level = max(0, min(level, slide.level_count - 1))

    # level dimensions
    level_w, level_h = slide.level_dimensions[level]

    # Build thumbnail to detect tissue (we will threshold saturation channel)
    if thumbnail_max_dim is None:
        thumb_w, thumb_h = level_w, level_h
    else:
        thumb_w = min(level_w, thumbnail_max_dim)
        thumb_h = min(level_h, thumbnail_max_dim)

    thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb_np = np.array(thumbnail)

    # convert to HSV and threshold by saturation (S) channel using Otsu
    thumb_hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    _, s_chan, _ = cv2.split(thumb_hsv)
    # if thumbnail is tiny or constant, Otsu might fail; handle that
    try:
        _, tissue_mask = cv2.threshold(s_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        tissue_mask = (s_chan > 10).astype(np.uint8) * 255

    # coordinates of tissue pixels in thumbnail coordinate space
    tissue_coords = np.argwhere(tissue_mask > 0)
    if tissue_coords.size == 0:
        slide.close()
        print(f"No tissue detected in thumbnail for {slide_path}. Aborting patch extraction.")
        return ([], [])

    # helper to map a thumbnail coordinate back to level0 coordinates:
    # scale factors between thumbnail and chosen level
    thumb_h, thumb_w = thumb_np.shape[:2]
    # mapping from thumbnail->level coordinates (float multipliers)
    scale_x = level_w / thumb_w
    scale_y = level_h / thumb_h

    saved_files = []
    centers = []
    images_out = []  # optional

    def read_patch_at_level(x_level, y_level, lvl, size):
        """Read a patch at requested level returning RGB numpy array and actual top-left coords"""
        # read_region takes location in level 0 reference frame; we need to convert
        # x_level, y_level are coordinates at 'level' resolution. Convert to level0 coordinates:
        scale_to_level0 = int(slide.level_downsamples[level])  # integer approximation
        # safer: compute factor between level and level0:
        factor_level_to_level0 = slide.level_downsamples[level]  # float
        # compute level0 coordinates
        x0 = int(round(x_level * factor_level_to_level0))
        y0 = int(round(y_level * factor_level_to_level0))
        # read from level 0 at (x0,y0) and then downsample to requested size if needed by PIL
        # but OpenSlide supports reading directly at multiple levels: pass lvl as level index
        patch = slide.read_region((x0, y0), lvl, (size, size)).convert("RGB")
        return patch, (x0, y0)

    # Sampling loop
    count = 0
    if sampling == "random":
        # sample uniformly among tissue pixels from the thumbnail
        max_tries = num_patches * 10
        tries = 0
        while count < num_patches and tries < max_tries:
            tries += 1
            idx = np.random.randint(len(tissue_coords))
            thumb_y, thumb_x = tissue_coords[idx]  # note: argwhere returns (row, col)
            # map to level coords
            x_level = int(round(thumb_x * scale_x))
            y_level = int(round(thumb_y * scale_y))

            # make sure patch is fully inside the level dims
            if x_level + patch_size > level_w or y_level + patch_size > level_h:
                continue

            # read patch at the chosen 'level' (integer)
            try:
                patch_pil, (x0, y0) = read_patch_at_level(x_level, y_level, level, patch_size)
            except Exception:
                # sometimes read_region fails for border coords; skip
                continue

            patch_np = np.array(patch_pil)
            # compute tissue ratio by saturation channel
            patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
            s_patch = patch_hsv[:, :, 1]
            tissue_pixels = np.sum(s_patch > 0)
            total_pixels = s_patch.size
            tissue_percentage = tissue_pixels / total_pixels

            if tissue_percentage >= tissue_threshold:
                fname = f"patch_{count:04d}_{x0}_{y0}.png"
                patch_pil.save(os.path.join(output_dir, fname))
                saved_files.append(os.path.join(output_dir, fname))
                # center in level0 coords
                center_x0 = x0 + patch_size // 2
                center_y0 = y0 + patch_size // 2
                centers.append((center_x0, center_y0))
                if return_images:
                    images_out.append(patch_np)
                count += 1
                if count % 10 == 0:
                    print(f"Saved {count} patches from {os.path.basename(slide_path)}...")
    else:
        # grid sampling
        stride = grid_stride if grid_stride is not None else patch_size
        # iterate over level coords
        for y in range(0, level_h - patch_size + 1, stride):
            for x in range(0, level_w - patch_size + 1, stride):
                if count >= num_patches:
                    break
                # Optionally skip grid cells with no tissue in the thumbnail mask:
                # map this level (x,y) back to thumbnail coords and quickly check tissue_mask
                thumb_x = int(round(x / scale_x))
                thumb_y = int(round(y / scale_y))
                # keep inside thumbnail
                thumb_x = min(max(0, thumb_x), thumb_w - 1)
                thumb_y = min(max(0, thumb_y), thumb_h - 1)
                if tissue_mask[thumb_y, thumb_x] == 0:
                    continue

                try:
                    patch_pil, (x0, y0) = read_patch_at_level(x, y, level, patch_size)
                except Exception:
                    continue
                patch_np = np.array(patch_pil)
                patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
                s_patch = patch_hsv[:, :, 1]
                tissue_pixels = np.sum(s_patch > 0)
                total_pixels = s_patch.size
                tissue_percentage = tissue_pixels / total_pixels
                if tissue_percentage >= tissue_threshold:
                    fname = f"patch_{count:04d}_{x0}_{y0}.png"
                    patch_pil.save(os.path.join(output_dir, fname))
                    saved_files.append(os.path.join(output_dir, fname))
                    centers.append((x0 + patch_size // 2, y0 + patch_size // 2))
                    if return_images:
                        images_out.append(patch_np)
                    count += 1
            if count >= num_patches:
                break

    slide.close()
    if return_images:
        return saved_files, centers, images_out
    return saved_files, centers

if __name__ == "__main__":
    logger.info("hello")