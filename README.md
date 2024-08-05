
# JSLO: Joint Sparse and Localization Optimization for Object Detection

## Introduction

This repo contains PyTorch implementation for JSLO


Other papers related to lightweight model design:
- AMC: AutoML for Model Compression and Acceleration on Mobile Devices ([ECCV 2018](https://arxiv.org/abs/1802.03494))

- HAQ: Hardware-Aware Automated Quantization with Mixed Precision ([CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf))

- Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning([NeurIPS 2022](https://proceedings.neurips.cc//paper_files/paper/2022/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html))
## Dependencies
The dependencies list in the requirements.txt of the root directory

## Dataset
If you already have the VisDrone dataset for pytorch, you could create a link to data folder and use it:
```
# prepare dataset, change the path to your own
ln -s /path/to/visdrone/ data/
```
If you do not have the Visdrone yet, you can download and customize it later: 
[https://github.com/VisDrone/VisDrone-Dataset]




## Joint Sparse and Localization
- You can run the following script to search the quantization strategy(the default compressed size is set to 0.2):

```
python rl_sq.py
```
- For more details, run:
```
python rl_sq.py --help
```

## Compression result on customized Visdrone(BUS)

| Models                            | preserve ratio | mAP 0.1(%) | mAP 0.5 (%) | mAP 0.75 (%) |
|-----------------------------------|----------------|------------|-------------|--------------|
| Centernet+CSPDarknet53 (original) | 1.00           | 0.49       | 0.45        |0.13          |
| JSLO(14x compress)                | 0.07           | 0.49       | 0.42        |0.13          |
