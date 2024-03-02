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
unzip image.zip -t data
```

### Preprocess Data
I have already done the data preprocessing with Coco format. However, you can reproduce by
```
python covert_to_coco_annotation.py
```
You can easily check by
```
python check_coco_annotation.py
```
By this,


## Usage:
### Training

### Preprocess Data

### 


# Acknowledgements:
Source code was implemented based on the following sources:
- [Mmdetection - OpenMM Lab](https://github.com/open-mmlab/mmdetection)
