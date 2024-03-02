<h1 align="center">
  Switch Picking Problem
</h1>

# Description:
[Provide a brief overview of your project, including its purpose and main features.]

I also provide Colab tutorial [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13E2_Rf_l4epgM6soDWsfcKHJEh05rt2i?usp=sharing) 

## Set up:
### Installation
Execute
```
bash requirements.sh
```
### Download Data
You can download [Google Drive files](https://drive.google.com/file/d/1i1UAWozECxv2IZbOwKEgLxJVIpGeUZQ3), unzip them and place them in the data/images directory. Or execute
```
gdown 1i1UAWozECxv2IZbOwKEgLxJVIpGeUZQ3
unzip image.zip
```

### Preprocess Data
I have already done the data preprocessing with Coco format. However, you can reproduce by
```
python preprocess/covert_to_coco_annotation.py
```
You can easily check by
```
python preprocess/check_coco_annotation.py
```
Run config file by
```
python preprocess/config.py
```
## Usage:
### Training
Execute
```
python mmdetection/tools/train.py mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_switch.py
```
Or you can download pretrained checkpoint at [Google Drive files](https://drive.google.com/file/d/1-_yCO7zStXzGlQYRGYla_KKfuywWqYeg/view?usp=sharing). Or execute
```
gdown 1-_yCO7zStXzGlQYRGYla_KKfuywWqYeg -O mmdetection/tutorial_exps/
```
### Picking Point Extraction
Execute
```
python picking_point.py
```
To visualize the result, run
```
python visualization/picking.py
```
### Pose Estimation 
Execute
```
python pose_estimation.py
```
To visualize the result, run
```
python visualization/pose.py
```
### Proposed Evaluation
- Qualitative Evaluation by Visualization
- Quantitative Evaluation using Average Reprojection Error

# Acknowledgements:
Source code was implemented based on the following source:
- [Mmdetection - OpenMM Lab](https://github.com/open-mmlab/mmdetection)
