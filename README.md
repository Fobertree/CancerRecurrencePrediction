# GNNs for Breast Cancer Recurrence Prediction

## Model Approach

### Image to Graph Processing
- For now, random patch sampling
    - Later, we don't want uniform randomness. We want to weigh patch selection by how "interesting" they are
        - Possible approaches: traditional CV: edge detection (Canny), entropy/variance of intensity (want higher), color deviation from tissue mean, converting to HSV (hue, saturation, value) and applying some sort of threshold(s)
- Random patch to DinoV2 image segmentation
- Build graphs

``source .venv/bin/activate``

Graph construction in `build.py`

Training and testing in `train.py`

``du -sh *``

For Alex: downloaded zips 1-8
- Next one: 9