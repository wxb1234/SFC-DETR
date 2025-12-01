# SFC-DETR:A Small Object Detection Model for UAVs Based on Spatial and Frequency Domain Collaborative Enhancement

# Description
This code is the algorithm implementation for article `SFC-DETR:A Small Object Detection Model for UAVs Based on Spatial and Frequency Domain Collaborative Enhancement`. The code is based on RT-DETR and is used to improve target detection in UAV images.

# Dataset Information
This paper uses two datasets for experiments.
## Visdrone
```
@article{zhu2021detection,
  title={Detection and tracking meet drones challenge},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={7380--7399},
  year={2021},
  publisher={IEEE}
}
```
## TinyPerson
```
@inproceedings{yu2020scale,
  title={Scale match for tiny person detection},
  author={Yu, Xuehui and Gong, Yuqi and Jiang, Nan and Ye, Qixiang and Han, Zhenjun},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={1257--1265},
  year={2020}
}
```

# Code Information
This code is built upon the RT-DETR codebase. The RT-DETR repository is located at: https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch.

# Usage Instructions
## Set the dataset path
### Visdrone
sfc_detr/configs/dataset/coco_detection_visdrone.yml
### TinyPerson
sfc_detr/configs/dataset/coco_detection-tp.yml
## Training on a Single GPU: 
```python tools/train.py -c sfc_detr/configs/sfc_detr/sfc_detr_r50vd_6x_coco.yml```
## Evaluation on a Single GPU:
```python tools/train.py -c sfc_detr/configs/sfc_detr/sfc_detr_r50vd_6x_coco.yml  -r path/to/checkpoint --test-only```

# Requirements
```
torch==2.0.1
torchvision==0.15.2
pycocotools
PyYAML
scipy
transformers
```