# [IEEE Transactions on Intelligent Vehicles] Dynamic V2X Perception from Road-to-Vehicle Vision
[model.pdf](https://github.com/user-attachments/files/16104457/model.pdf)

## Requirements
- [PyTorch >= version 1.4](https://pytorch.org)

## Datasets
We follow [DiscoNet]([https://github.com/xyutao/fscil](https://github.com/ai4ce/DiscoNet)) setting to use the same data index_list for training. 
Please follow the guidelines in [DiscoNet]([https://github.com/icoz69/CEC-CVPR2021 (https://github.com/ai4ce/DiscoNet)) to prepare them.

## Train
python train_codet.py \
    --data  /path/to/training/dataset \
    --com disco \
    --log --batch 4 \
    --kd_flag 1 \
    --resume_teacher /path/to/teacher/checkpoint.pth \
    --auto_resume_path logs \
    --logpath logs \
    --nepoch 100 \
    -- rsu [0/1]
## Test
python test_codet.py \
    --data /path/to/testing/dataset \
    --com disco \
    --resume /path/to/teacher/checkpoint.pth \
    --tracking \
    --logpath logs \
    --visualization 1 \
    --rsu 1
    
## Acknowledgment
Our project references the codes in the following repos.
[DiscoNet]([https://github.com/xyutao/fscil](https://github.com/ai4ce/DiscoNet))
[V2x-sim]([https://github.com/ai4ce/V2X-Sim])
