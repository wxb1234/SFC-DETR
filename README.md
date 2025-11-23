# SFC-DETR:A Small Object Detection Model for UAVs Based on Spatial and Frequency Domain Collaborative Enhancement
Train/test script examples
- sfc_detr/configs/sfc_detr/sfc_detr_r50vd_6x_coco.yml-c path/to/config &> train.log 2>&1 &`
- `-r path/to/checkpoint`
- `--amp`
- `--test-only`

## Data Preprocessing Instructions
- Training
First, the input image undergoes strong data augmentation, applying random photometric distortion to adjust brightness, contrast, and color saturation. Then, random scaling is performed, and black pixels are used to fill edges. Next, IoU-based intelligent cropping is applied to ensure valid target regions are preserved. After data augmentation, invalid bounding boxes and their corresponding labels are removed, and random horizontal flipping is applied to increase data diversity. The image is then uniformly scaled to a fixed size of 640x640 pixels, converted to tensor format, and undergoes data type conversion. Finally, the validity of the bounding boxes is verified again, invalid bounding boxes and their corresponding labels are removed, and the bounding box coordinates are converted from xyxy format to normalized cxcywh format. The entire process is processed in parallel using 4 threads, processing 4 images per batch, and the data order is shuffled in each training epoch.

- Inference 
The input image is scaled to a fixed size of 640x640 pixels, without any data augmentation. The images are then converted to tensor format and subjected to the corresponding data type conversion without disrupting the data order. Eight images are processed in batches for rapid processing, and four threads are used to ensure efficient data loading.

## Datasets
### Visdrone
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

### TinyPerson
@inproceedings{yu2020scale,
  title={Scale match for tiny person detection},
  author={Yu, Xuehui and Gong, Yuqi and Jiang, Nan and Ye, Qixiang and Han, Zhenjun},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={1257--1265},
  year={2020}
}