## Running instructions

Please run the commands mentioned below to produce results:

## Training commands
```
python main_cr2m.py --noise_mode aggre_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --noise_mode rand_1_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --noise_mode rand_2_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --noise_mode rand_3_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --noise_mode worst_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
python main_cr2m.py --dataset cifar100 --noise_mode worst_label --feature_type foundation_model --encoder_name dinov2_vitl14 --batch_size 128 --cache_mode load
```
## Evaluation commands
```
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode aggre_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode rand_1_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode rand_2_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode rand_3_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar10 --noise_mode worst_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva_cifarn.py --gpu 0 --dataset cifar100 --noise_mode worst_label --loss_type CR2M -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes

