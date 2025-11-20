"""
Split image preprocessing and patching (simple: read level-0, downsample, save JPEG)
"""

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import openslide
import os
from pathlib import Path
from PIL import Image
import numpy as np

VALID_EXTS = (".tiff", ".tif", ".svs")

def _process_one(src_path: Path, dst_dir: Path, magnification: float) -> int:
    """
    Top-level worker so it can be pickled on Windows.
    Returns 0 on success, 1 on fallback/handled warning, 2 on error.
    """
    try:
        slide = openslide.OpenSlide(str(src_path))
        # Read the full-resolution slide
        full_res = slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB")

        # Downsample manually to target magnification
        target_scale = 1.0 / float(magnification)
        new_size = (max(1, int(full_res.width * target_scale)),
                    max(1, int(full_res.height * target_scale)))
        downsamp_img = full_res.resize(new_size, Image.Resampling.LANCZOS)

        # Save JPEG
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_path = dst_dir / (src_path.stem + ".jpeg")
        downsamp_img.save(out_path, format="JPEG", quality=90, optimize=True)

        # Optional: print sizes
        # print(f"{src_path.name}: {full_res.size} -> {downsamp_img.size}")

        return 0

    except KeyError:
        # Fallback to level 1 if level 0 props missing
        try:
            downsamp_img = np.array(slide.read_region((0, 0), 1, slide.level_dimensions[1]).convert("RGB"))
            out_path = dst_dir / (src_path.stem + ".jpeg")
            Image.fromarray(downsamp_img).save(out_path, format="JPEG", quality=90, optimize=True)
            return 1
        except Exception as e:
            print(f"ERROR (fallback) {src_path.name}: {e}")
            return 2
    except Exception as e:
        print(f"ERROR {src_path.name}: {e}")
        return 2


def downsample_slides_to_img(directory_path: str = "Data",
                             destination_path: str = "PreprocessedData",
                             magnification: float = 20.0,
                             max_workers: int = 4) -> None:
    """
    Downsamples and saves images as preprocessing.
    Note: still reads level-0 into RAM (heavy). This mirrors the original behavior.
    """
    src_dir = Path(directory_path)
    dst_dir = Path(destination_path)

    # Gather slide files (keep full filenames with extensions)
    slides = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    print(f"Found {len(slides)} slides in {src_dir}")

    if not slides:
        return

    # Windows-friendly multiprocessing (top-level function)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Avoid lambdas/locals (not picklable on Windows - spawn start method).
        # Create iterables for the constant args so we can call the top-level _process_one
        # directly with executor.map(src_path, dst_dir, magnification).
        dst_iter = [dst_dir] * len(slides)
        mag_iter = [magnification] * len(slides)
        results = list(tqdm(
            executor.map(_process_one, slides, dst_iter, mag_iter),
            total=len(slides)
        ))

    failed = sum(1 for r in results if r == 2)
    print(f"Failed {failed} times")


if __name__ == "__main__":
    import time
    st = time.time()
    root_path = "Data"
    dest_path = "PreprocessedData"
    os.makedirs(dest_path, exist_ok=True)
    downsample_slides_to_img(root_path, dest_path, magnification=20.0, max_workers=4)
    print(f"Done. Took {time.time() - st:.2f}")
