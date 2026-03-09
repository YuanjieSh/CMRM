## Conformal Margin Risk Minimization: An Envelope Framework for Robust Learning under Label Noise (CMRM)

This repository contains a implementation of **CMRM**
corresponding to the follow paper:

Yuanjie Shi*, Peihong Li*, Zijian Zhang, Jana Doppa, Yan Yan.
*[Conformal Margin Risk Minimization: An Envelope Framework for Robust Learning under Label Noise](
https://openreview.net/forum?id=3wGVIAhxty)*.
AISTATS, 2026 .

## Overview

Training reliable classifiers under label noise is a challenging task. Existing methods often rely on restrictive assumptions about the noise distribution, model design, or access to clean data. Such assumptions rarely hold in practice, especially under severe or heterogeneous noise. We propose Conformal Margin Risk Minimization (CMRM), an uncertainty-aware envelope framework to improve the robustness of prior methods with noisy labeled data. Specifically, CMRM computes the confidence margin as the gap between confidence scores of observed label and other labels, and then a conformal quantile estimated over a batch of examples provides a statistically valid proxy for the set-level quantile. Minimizing the conformal margin risk allows the training to focus on low uncertainty (high margin) samples while filtering out high uncertainty (low margin) samples below the quantile, as mislabeled samples. We derive a learning bound for CMRM under arbitrary label noise with weaker assumptions than prior work. Experiments show that CMRM consistently improves accuracy and robustness of prior methods across different classification benchmarks without prior knowledge of noise.

## Running instructions

Please run the commands mentioned below to produce results:

# Multi-class Experiements on Datasets with Synthetic Noise
**Training commands**
```
Sh Multi_code_synthetic_noise/train.sh
```
**Evaluation commands**
```
sh Multi_code_synthetic_noise/eva.sh
```
# Multi-class Experiements on Datasets with Human Annotation Noise
**Training commands**
```
python main_cr2m.py --noise_mode aggre_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --noise_mode rand_1_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --noise_mode rand_2_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --noise_mode rand_3_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --noise_mode worst_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --dataset cifar100 --noise_mode worst_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
```
**Evaluation commands**
```
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode aggre_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode rand_1_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode rand_2_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode rand_3_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode worst_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar100 --noise_mode worst_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
```
# Binary class Experiements
**Training commands**
```
Sh binary_code/train_2.sh
```
**Evaluation commands**
```
sh binary_code/evaluate.sh
```