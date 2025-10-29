'''
Split image preprocessing and patching
'''

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import openslide
import os
import pandas as pd
from PIL import Image
import numpy as np

def downsample_slides_to_img(directory_path : str, destination_path: str, magnification=20) -> list[openslide.OpenSlide]:
    '''
    MAKE SURE DESTINATION_PATH IS GITIGNORED

    Downsamples and saves images as preprocessing
    '''
    svs_names = []
    
    # not sure if it makes a difference if we load names from metadata_df instead
    # we could filter out unwanted images based on absence of metadata if we do that instead
    # maybe this is smth you could do @thomas
    for file in os.listdir(directory_path):
        if file.endswith(".tiff"):
            svs_names.append(file.strip(".tiff"))
    
    def helper(svs_name):
        '''
        Choose 20 because compute limitations (~1k-1.5k per dim after downsample)
        '''
        try:
            slide = openslide.OpenSlide(os.path.join(directory_path, f"{svs_name}.tiff"))
            # Read the full-resolution slide
            full_res = slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB")

            # Downsample manually to target magnification
            target_scale = 1 / magnification
            new_size = (int(full_res.width * target_scale), int(full_res.height * target_scale))
            downsamp_img = full_res.resize(new_size, Image.Resampling.LANCZOS)
            downsamp_img = np.array(downsamp_img)

            # save image
            im = Image.fromarray(downsamp_img)
            im.save(os.path.join(destination_path,f"{svs_name}.jpeg")) # save jpeg bc lossy

            print(f"Full slide size: {full_res.size}, Downsampled size: {downsamp_img.shape[:2]}")
            return 0

        except KeyError:
            print("Warning: Could not determine magnification. Using slide level 1.")
            downsamp_img = np.array(slide.read_region((0, 0), 1, slide.level_dimensions[1]).convert("RGB"))
            return 1
    # END HELPER FUNC

    # If Python 3.14 or 3.13 with special GIL disable, then threadpool should be much better
    with ProcessPoolExecutor(max_workers=4) as executor:
        res = list(tqdm(executor.map(helper, svs_names)))
        print(f"Failed {sum(res)} times")