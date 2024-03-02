pip install -U openmim
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0rc4"

# Install mmdetection
# rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

pip install -e .
mim download mmdet --config mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco --dest ./mmdetection/checkpoints

# pip install future tensorboard 
# pip install --upgrade setuptools