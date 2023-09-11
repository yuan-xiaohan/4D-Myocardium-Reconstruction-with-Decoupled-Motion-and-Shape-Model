# 4D-Myocardium-Reconstruction-with-Decoupled-Motion-and-Shape-Model
This repository contains the code for the ICCV'2023 paper "4D Myocardium Reconstruction with Decoupled Motion and Shape Model". 

Xiaohan Yuan, Cong Liu, [Yangang Wang](https://www.yangangwang.com/#me)  

\[[Paper](https://arxiv.org/pdf/2308.14083.pdf)\]
<div align=center><img src="/images/teaser.jpg" width="50%"></div>


## Introduction
Estimating the shape and motion state of the myocardium is essential in diagnosing cardiovascular diseases. However, cine magnetic resonance (CMR) imaging is dominated by 2D slices, whose large slice spacing challenges inter-slice shape reconstruction and motion acquisition. To address this problem, we propose a 4D reconstruction method that decouples motion and shape, which can predict the inter-/intra- shape and motion estimation from a given sparse point cloud sequence obtained from limited slices. Our framework comprises a neural motion model and an end-diastolic (ED) shape model. The implicit ED shape model can learn a continuous boundary and encourage the motion model to predict without the supervision of ground truth deformation, and the motion model enables canonical input of the shape model by deforming any point from any phase to the ED phase. Additionally, the constructed ED-space enables pre-training of the shape model, thereby guiding the motion model and addressing the issue of data scarcity. We propose the first 4D myocardial dataset (4DM Dataset) as we know and verify our method on the proposed, public, and cross-modal datasets, showing superior reconstruction performance and enabling various clinical applications.

## Dataset
Here, we release a new dataset consisting of 25 healthy subjects obtained from Jiangsu Province Hospital. Each subject includes multiple slices (8-10 slices) with a resolution of 1.25 × 1.25 × 10mm. Each slice covers the video sequence of the cardiac cycle (25 phases). The clinical experts manually delineated the left myocardium of all the phases and slices.

[4DM Dataset](https://drive.google.com/drive/folders/1027CUnLNoGiAiqBNI65f7pbAM5wn9Rih)  


## Usage
### Data Layout
```
${examples\demo}
|-- data
    |-- 000
        P.txt
        |-- mesh
            |-- 00.obj
            ...
            |-- 24.obj
        |-- points
            |-- 00.obj
            ...
            |-- 24.obj

    ...
```
### Preprocessing
    preprocess_data.py [-h] --data_dir DATA_DIR --source_dir SOURCE_DIR
                          --source_name SOURCE_NAME --class_name CLASS_NAME
                          --split SPLIT_FILENAME [--test]

### Training a Model
    train.py [-h] --experiment EXPERIMENT_DIRECTORY --data
                      DATA_SOURCE [--continue CONTINUE_FROM] [--debug]
                      [--quiet] [--log LOGFILE]

### Testing a Model
    reconstruct.py [-h] --experiment EXPERIMENT_DIRECTORY
                            [--checkpoint CHECKPOINT] --data DATA_SOURCE
                            --split SPLIT_FILENAME [--iters ITERATIONS]
                            [--seed SEED] [--resolution RESOLUTION] [--debug]
                            [--quiet] [--log LOGFILE]
    
## Citation
If you find our work is useful or want to use our dataset, please consider citing the paper.
```
@inproceedings{yuan2023myo4d,
title={4D Myocardium Reconstruction with Decoupled Motion and Shape Model},
author={Yuan, Xiaohan and Liu, Cong and Wang, Yangang},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
year={2023},
}
```

## Reference
Some of the code is based on the following works. We gratefully appreciate the impact they have on our work.

[mesh_to_sdf](https://github.com/marian42/mesh_to_sdf)

[Statistical-Shape-Model](https://github.com/UK-Digital-Heart-Project/Statistical-Shape-Model)

[DeepSDF](https://github.com/facebookresearch/DeepSDF)

[DeepImplicitTemplates](https://github.com/ZhengZerong/DeepImplicitTemplates)

[LoRD](https://github.com/BoyanJIANG/LoRD)
