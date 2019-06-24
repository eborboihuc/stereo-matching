# Stereo Matching

**Keywords** : Semi-Global Matching, StereoSGBM_create

<p float="left">
  <img src="img/rgb_2011_09_26_0001_02_0070.png" width="49%" />
  <img src="img/rgb_2011_09_26_0001_03_0070.png" width="49%" />
</p>

### Requirements
* Python 3.6.4
* Numpy
* OpenCV

### Usage
- Show teaser image in 3D

![](img/stereo_image.png)

```python
python test.py img/rgb_2011_09_26_0001_02_0070.png img/rgb_2011_09_26_0001_03_0070.png
```

- Show disparity map

![](img/projected_disparity.png)

```python
python test.py img/rgb_2011_09_26_0001_02_0070.png img/rgb_2011_09_26_0001_03_0070.png --show_disp_map
```

- Show depth map

![](img/projected_depth.png)

```python
python test.py img/rgb_2011_09_26_0001_02_0070.png img/rgb_2011_09_26_0001_03_0070.png --show_depth_map
```

