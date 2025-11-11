'''
Full rewrite data pipeline
'''
import openslide
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import logging
import math
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logger = logging.Logger("ppl2", level=logging.INFO)
log_file = 'Logs/ppl2.log'
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s -%(levelname)s :: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



def _process_slide(slide_path, metadata_row, scale_factor=20):
    """Worker for processing a single WSI slide."""
    try:
        slide_name = os.path.basename(slide_path).lower()
        slide_stem = os.path.splitext(slide_name)[0].strip()

        slide = openslide.OpenSlide(slide_path)
        level = slide.get_best_level_for_downsample(1)

        wsi = slide.read_region((0, 0), level, slide.level_dimensions[level])
        img_rgb = wsi.convert('RGB')

        # Downsample
        large_w, large_h = slide.dimensions
        new_w = math.floor(large_w / scale_factor)
        new_h = math.floor(large_h / scale_factor)
        img_downsampled = img_rgb.resize((new_w, new_h), Image.LANCZOS)

        out_name = f"{slide_stem}.png"
        out_path = os.path.join("Image", out_name)
        # save image
        img_downsampled.save(out_path)

        logger.info(f"[{slide_stem}] original: {slide.dimensions}, downsampled: {img_downsampled.size}")

        slide.close()

        # Return useful metadata for downstream patch extraction
        return {
            "slide_name": slide_stem,
            "path": out_path,
            "width": new_w,
            "height": new_h,
            "metadata": metadata_row.to_dict()
        }

    except Exception as e:
        logger.error(f"Error processing {slide_path}: {e}")
        return None


def load_wsi2fast(
    directory_path: str,
    metadata_path: str,
    output_dir: str = "dinov2_patches",
    threshold: int | None = None,
    replace: bool = False
):
    """
    Parallelized WSI preprocessing:
    - Reads slides (.tif/.svs)
    - Downsamples them
    - Saves thumbnails to ./Image/
    - Returns metadata for each processed WSI

    @replace - if True, replace image if detected at path

    NOTE: It seems that sometimes this function leaks 1 semaphore
    - My hypothesis: This is a multiprocessing + tqdm bug. Don't think it's critical
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("Image", exist_ok=True)

    # Load metadata
    df = pd.read_csv(metadata_path)
    if "svs_name" not in df.columns:
        raise ValueError("Metadata CSV must contain 'svs_name' column")

    df["svs_stem"] = (
        df["svs_name"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(".svs", "", regex=False)
        .str.replace(".tif", "", regex=False)
    )

    # Gather slide files
    wsi_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(directory_path)
        for f in files
        if f.lower().endswith((".tif", ".svs"))
    ]

    if threshold is not None:
        wsi_files = wsi_files[:threshold]

    logger.info(f"Found {len(wsi_files)} WSI files to process.")

    results = []
    # max_workers = max(1, multiprocessing.cpu_count() - 3)
    max_workers = 6
    print(max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_slide = {}

        for slide_path in wsi_files:
            slide_name = os.path.basename(slide_path).lower()
            slide_stem = os.path.splitext(slide_name)[0].strip()
            meta_row = df[df["svs_stem"] == slide_stem]

            if not replace and os.path.exists(os.path.join("Image", f"{slide_stem}.png")):
                continue

            if meta_row.empty:
                logger.warning(f"No metadata found for {slide_name}, skipping.")
                continue

            future = executor.submit(
                _process_slide,
                slide_path,
                meta_row.iloc[0],
                20
            )
            future_to_slide[future] = slide_name

        for future in tqdm(as_completed(future_to_slide), total=len(future_to_slide), desc="Processing WSIs"):
            result = future.result()
            if result:
                results.append(result)

    logger.info(f"Completed processing {len(results)} slides.")
    return results

def patch_img(image_directory="Image", 
            output_dir: str = "dinov2_patches_seq",
            threshold: int | None = None,
            patch_size: int = 28,
            tissue_threshold: float = 0.8):

    os.makedirs(output_dir, exist_ok=True)

    # Collect slide files
    img_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(image_directory)
        for f in files
        if f.lower().endswith((".png"))
    ]
    
    for img_path in tqdm(img_files, desc="Loading images"):
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)
            height, width, _ = img_np.shape
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
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
            print(f"No tissue found in {img_path}. Skipping.")
            continue
        
        # PATCH from img_pil or img_np then save as png format 
        slide_name = os.path.splitext(os.path.basename(img_path))[0]
        slide_dir = os.path.join(output_dir, slide_name)
        os.makedirs(slide_dir, exist_ok=True)

        # patch_{num}_{x}_{y}.png
        # sequential grid-based patching with coordinates
        patch_count = 0
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                patch_np = img_np[y:y + patch_size, x:x + patch_size, :]
                mask_patch = tissue_mask[y:y + patch_size, x:x + patch_size]

                # Compute tissue coverage ratio
                tissue_ratio = np.mean(mask_patch > 0)

                # Skip patches with too little tissue
                if tissue_ratio < tissue_threshold:
                    continue

                # Save patch
                patch_pil = Image.fromarray(patch_np)
                patch_filename = f"patch_{patch_count}_{x}_{y}.png"
                patch_pil.save(os.path.join(slide_dir, patch_filename))
                patch_count += 1
        
        if patch_count == 0:
            print(f"No valid patches for {slide_name}")
        else:
            print(f"Saved {patch_count} patches for {slide_name}")



if __name__ == "__main__":
    # download wsis
    load_wsi2fast("Data", "new_metadata.csv")

    # patch images
    patch_img()

    # graph construction
    from graphbuildertransform import build_graphs_seq

    build_graphs_seq(replace=False)

    pass