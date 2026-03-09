

python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type CE --train_rule None --model resnet20 --epochs 200 --dataset cifar100 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type Focal --train_rule None --model resnet20 --epochs 200 --dataset cifar100 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type LDAM --train_rule None --model resnet20 --epochs 200 --dataset cifar100 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type GCE --train_rule None --model resnet20 --epochs 200 --dataset cifar100 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes

python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type CE_reg --train_rule None --model resnet20 --epochs 200 --dataset cifar100 --tr_alpha 0.85 --lr 0.05 --reg 0.1 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type Focal_reg --train_rule None --model resnet20 --epochs 200 --dataset cifar100 --tr_alpha 0.9 --lr 0.05 --reg 0.15 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type LDAM_reg --train_rule None --model resnet20 --epochs 200 --dataset cifar100 --tr_alpha 0.85 --lr 0.05 --reg 0.1 --start_epoch 100 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type GCE_reg --train_rule None --model resnet20 --epochs 200 --dataset cifar100 --tr_alpha 0.95 --lr 0.05 --reg 0.005 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 50 --bins 25 --all yes


python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type CE --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type Focal --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type LDAM --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type GCE --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --q 0.5 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes

python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type CE_reg --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --tr_alpha 0.8 --lr 0.05 --reg 0.15 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type Focal_reg --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --tr_alpha 0.85 --lr 0.05 --reg 0.15 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type LDAM_reg --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --tr_alpha 0.9 --lr 0.05 --reg 0.2 --start_epoch 100 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type GCE_reg --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --q 0.5 --tr_alpha 0.85 --lr 0.05 --reg 0.0005 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes


python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type CE --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type Focal --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type LDAM --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type GCE --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --q 0.3 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes

python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type CE_reg --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --tr_alpha 0.85 --lr 0.05 --reg 0.15 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type Focal_reg --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --tr_alpha 0.9 --lr 0.05 --reg 0.15 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type LDAM_reg --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --tr_alpha 0.85 --lr 0.05 --reg 0.05 --start_epoch 100 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes
python main_eva.py --gpu 0 --noise_rho 0.2 --loss_type GCE_reg --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --q 0.3 --tr_alpha 0.9 --lr 0.05 --reg 0.0001 --start_epoch 150 -score_functions HPS -methods MCP -seeds 0 1 2 3 4 5 6 7 8 9 --calibration_sampling balanced -avg_num_per_class 150 --bins 25 --all yes







