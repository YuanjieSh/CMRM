python train/CIFAR100_base_noise.py --gpu 0 --noise_rho 0.2 --loss_type CE --train_rule None --model resnet20 --epochs 200 --dataset cifar100 
python train/CIFAR100_base_noise.py --gpu 0 --noise_rho 0.2 --loss_type Focal --train_rule None --model resnet20 --epochs 200 --dataset cifar100 
python train/CIFAR100_base_noise.py --gpu 0 --noise_rho 0.2 --loss_type LDAM --train_rule None --model resnet20 --epochs 200 --dataset cifar100 
python train/CIFAR100_base_noise.py --gpu 0 --noise_rho 0.2 --loss_type GCE --q 0.7 --train_rule None --model resnet20 --epochs 200 --dataset cifar100 

python train/cifar10_reg_train.py --gpu 0 --noise_rho 0.2 --loss_type CE_reg --train_rule None --model resnet20 --epochs 200 --dataset cifar100 --tr_alpha 0.85 --lr 0.05 --reg 0.1 --start_epoch 150
python train/cifar10_reg_train.py --gpu 0 --noise_rho 0.2 --loss_type Focal_reg --train_rule None --model resnet20 --epochs 200 --dataset cifar100 --tr_alpha 0.9 --lr 0.05 --reg 0.15 --start_epoch 150 
python train/cifar10_reg_train.py --gpu 0 --noise_rho 0.2 --loss_type LDAM_reg --train_rule None --model resnet20 --epochs 200 --dataset cifar100 --tr_alpha 0.85 --lr 0.05 --reg 0.1 --start_epoch 100 
python train/cifar10_reg_train.py --gpu 0 --noise_rho 0.2 --loss_type GCE_reg --q 0.7 --train_rule None --model resnet20 --epochs 200 --dataset cifar100 --tr_alpha 0.95 --lr 0.05 --reg 0.005 --start_epoch 150 


python train/mini_noise_train.py --gpu 0 --noise_rho 0.2 --loss_type CE --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 
python train/mini_noise_train.py --gpu 0 --noise_rho 0.2 --loss_type Focal --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 
python train/mini_noise_train.py --gpu 0 --noise_rho 0.2 --loss_type LDAM --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 
python train/mini_noise_train.py --gpu 0 --noise_rho 0.2 --loss_type GCE --q 0.5 --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 

python train/mini_noise_reg.py --gpu 0 --noise_rho 0.2 --loss_type CE_reg --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --tr_alpha 0.8 --lr 0.05 --reg 0.15 --start_epoch 150 
python train/mini_noise_reg.py --gpu 0 --noise_rho 0.2 --loss_type Focal_reg --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --tr_alpha 0.85 --lr 0.05 --reg 0.15 --start_epoch 150 
python train/mini_noise_reg.py --gpu 0 --noise_rho 0.2 --loss_type LDAM_reg --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --tr_alpha 0.9 --lr 0.05 --reg 0.2 --start_epoch 100 
python train/mini_noise_reg.py --gpu 0 --noise_rho 0.2 --loss_type GCE_reg --q 0.5 --train_rule None --model resnet20 --epochs 200 --dataset mini --batch_size 512 --tr_alpha 0.85 --lr 0.05 --reg 0.0005 --start_epoch 150 


python train/food_noise_train.py --gpu 0 --noise_rho 0.2 --loss_type CE --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 
python train/food_noise_train.py --gpu 0 --noise_rho 0.2 --loss_type Focal --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 
python train/food_noise_train.py --gpu 0 --noise_rho 0.2 --loss_type LDAM --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 
python train/food_noise_train.py --gpu 0 --noise_rho 0.2 --loss_type GCE --q 0.3 --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 

python train/food_noise_reg.py --gpu 0 --noise_rho 0.2 --loss_type CE_reg --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --tr_alpha 0.85 --lr 0.05 --reg 0.15 --start_epoch 150 
python train/food_noise_reg.py --gpu 0 --noise_rho 0.2 --loss_type Focal_reg --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --tr_alpha 0.9 --lr 0.05 --reg 0.15 --start_epoch 150 
python train/food_noise_reg.py --gpu 0 --noise_rho 0.2 --loss_type LDAM_reg --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --tr_alpha 0.85 --lr 0.05 --reg 0.05 --start_epoch 100 
python train/food_noise_reg.py --gpu 0 --noise_rho 0.2 --loss_type GCE_reg --q 0.3 --train_rule None --model resnet20 --epochs 200 --dataset food --batch_size 512 --tr_alpha 0.9 --lr 0.05 --reg 0.0001 --start_epoch 150 

