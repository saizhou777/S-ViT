# S-ViT

## 🍀 1. Trianing Scripts

To train S-ViT-T on the ImageNet-1K dataset with two gpus, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29502 --use_env main.py --warmup-epochs 20 --model SVIT_S --data-path /data1/saizhou777/imagenet --num_workers 6 --batch-size 512 --drop-path 0.00 --epoch 300 --dist-eval --output_dir /data1/saizhou777/RALA2/output/svit_v1
```

To train S-ViT-DETR on the ImageNet-1K dataset with two gpus, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29510 main.py
```

To train S-ViT-Track on the GOT-10k dataset with two gpus, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script ostrack --config svit_t_128_32x4_got10k_ep100 --save_dir ./output --mode multiple --nproc_per_node 2 --use_wandb 0
```


## 👏 2. Acknowledgement
This repository is built using [RALA](https://github.com/qhfan/RALA), [Relation DETR](https://github.com/xiuqhou/Relation-DETR/tree/main), and [OSTrack](https://github.com/botaoye/OSTrack) repositories. We particularly appreciate their open-source efforts.


## 📖 3. Citation
If you find this repository helpful, please consider citing:
