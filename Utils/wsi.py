import numpy as np
import os
import tifffile as tiff


'''
Preprocessing:
- Load .tiff image (not of standard size)
- Find way to standardize image into tensor: TODO @aliu
'''

def load_images(directory_path : str, threshold : int | None = None):
    cnt = 0
    res = []
    for dirname, _, filenames in os.walk(directory_path):
        for filename in filenames:
            f_path = os.path.join(dirname, filename)
            img = tiff.imread(f_path)
            res.append(img)
            cnt+=1
            if cnt >= threshold:
                return res
    
    return res
