import numpy as np
import os
import tifffile as tiff
import itertools
import pandas as pd

import openslide
import cv2
from PIL import Image
import logging
from tqdm import tqdm

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

def load_images(directory_path : str, threshold : int | None = None) -> list[np.array]:
    cnt = 0
    res = []
    # TODO @thomas: load labels in this as well
    for dirname, _, filenames in os.walk(directory_path):
        for filename in filenames:
            f_path = os.path.join(dirname, filename)
            img = tiff.imread(f_path)
            res.append(img)
            cnt+=1
            if cnt >= threshold:
                return res
    
    return res

def load_wsi(directory_path: str, metadata_path: str, threshold: int | None = None):
    '''
    Loads (up to threshold) wsi from path and samples random patches
    
    We call this in pre-processing
    - Take the output of this function and generate a graph from it to pass into GNN blocks
    '''
    os.makedirs("dinov2_patches", exist_ok=True)
    res = []
    # TODO @thomas: match the logic of this to the dataset structure
    
    # process at most threshold images
    # load metadata
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_path}")

    # collect image files
    wsi_files = []
    for root, _, files in os.walk(directory_path):
        for f in files:
            if f.lower().endswith((".tif", ".svs")):
                wsi_files.append(os.path.join(root, f))

    if threshold is not None:
        wsi_files = wsi_files[:threshold]

    # process each WSI
    for slide_path in tqdm(wsi_files, desc="Loading WSIs"):
        slide_name = os.path.basename(slide_path).lower()

        # Match metadata row for this slide (case-insensitive)
        # Standardize slide filename (no extension, lowercase, stripped)
        slide_stem = os.path.splitext(os.path.basename(slide_path))[0].lower().strip()

        # Standardize CSV column
        df['svs_stem'] = df['svs_name'].astype(str).str.lower().str.strip().str.replace('.svs','').str.replace('.tif','')

        # Match
        meta_row = df[df['svs_stem'] == slide_stem]
        meta_row['Oncotype DX Breast Recurrence Score'] = "1" if meta_row['Oncotype DX Breast Recurrence Score'] > 25 else "0"

        if meta_row['ER'] == 1 and meta_row['HER'] == 0:
            print("Slide does not match criteria. Skipping...")

        if meta_row.empty:
            logger.error(f"No metadata found for slide {slide_name}")
            print(f"Warning: no metadata found for {slide_name}")
            metadata = {}
        else:
            metadata = meta_row.iloc[0].to_dict()

        try:
            slide = openslide.OpenSlide(slide_path)
            patches, coords = sample_random_wsi_patches(slide)

            # Add internal OpenSlide info too
            metadata.update({
                "slide_dimensions": slide.dimensions,
                "level_count": slide.level_count,
                "openslide_properties": dict(slide.properties)
            })

            res.append((patches, coords, metadata))

        except openslide.OpenSlideError:
            print(f"Error opening slide {slide_path}")
            logger.error(f"OpenSlideError::load_wsi: slide path: {slide_path}")
        finally:
            if 'slide' in locals():
                slide.close()
    
    return res

'''
I got the below code from Gemini, I will keep this for now until we can build something better

I think the WSI preprocessing is the biggest point of improvement for us but also arguably the most complex
- I think it's best to build out the rest of the model architecture then improve this WSI to graph feature generation as much as possible
'''

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


if __name__ == "__main__":
    logger.info("hello")