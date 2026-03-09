
python main_evaluate_2.py --dataset adult --model MLP --method CE --lr 0.01 --epochs 300 --noise_rho 0.2
python main_evaluate_2.py --dataset adult --model MLP --method FOCAL --lr 0.01 --epochs 300 --noise_rho 0.2
python main_evaluate_2.py --dataset adult --model SVM --method SVM --lr 0.01 --epochs 300 --lambda_reg 0.01 --noise_rho 0.2
python main_evaluate_2.py --dataset adult --model MLP --method GCE --lr 0.01 --epochs 300 --noise_rho 0.2 

python main_evaluate_2.py --dataset adult --model MLP --method Margin_8 --lr 0.01 --epochs 150 --lambda_plus 0.4 --lambda_minus 0.5 --alpha_plus 0.3 --alpha_minus 0.1 --scheduler none --noise_rho 0.2 --finetune true 
python main_evaluate_2.py --dataset adult --model MLP --method Margin_9 --lr 0.01 --epochs 150 --lambda_plus 0.1 --lambda_minus 0.4 --alpha_plus 0.3 --alpha_minus 0.5 --scheduler none --noise_rho 0.2 --finetune true 
python main_evaluate_2.py --dataset adult --model SVM --method Margin_10 --lr 0.01 --epochs 150 --lambda_reg 0.001 --lambda_plus 0.15 --lambda_minus 1.5 --alpha_plus 0.2 --alpha_minus 0.5 --scheduler none --noise_rho 0.2 --finetune true 
python main_evaluate_2.py --dataset adult --model MLP --method Margin_11 --lr 0.001 --epochs 150 --lambda_plus 0.0 --lambda_minus 0.2 --alpha_plus 0.1 --alpha_minus 0.5 --scheduler none --noise_rho 0.2 --finetune true 


python main_evaluate_2.py --dataset email --model MLP --method CE --lr 0.01 --epochs 300 --noise_rho 0.2
python main_evaluate_2.py --dataset email --model MLP --method FOCAL --lr 0.01 --epochs 300 --noise_rho 0.2
python main_evaluate_2.py --dataset email --model SVM --method SVM --lr 0.01 --epochs 300 --lambda_reg 0.01 --noise_rho 0.2
python main_evaluate_2.py --dataset email --model MLP --method GCE --lr 0.01 --epochs 300 --noise_rho 0.2 

python main_evaluate_2.py --dataset email --model MLP --method Margin_8 --lr 0.01 --epochs 150 --lambda_plus 0.8 --lambda_minus 0.7 --alpha_plus 0.2 --alpha_minus 0.2 --scheduler none --noise_rho 0.2 --finetune true
python main_evaluate_2.py --dataset email --model MLP --method Margin_9 --lr 0.01 --epochs 150 --lambda_plus 0.4 --lambda_minus 0.6 --alpha_plus 0.1 --alpha_minus 0.1 --scheduler none --noise_rho 0.2 --finetune true
python main_evaluate_2.py --dataset email --model SVM --method Margin_10 --lr 0.01 --epochs 150 --lambda_reg 0.05 --lambda_plus 0.3 --lambda_minus 1.0 --alpha_plus 0.3 --alpha_minus 0.3 --scheduler none --noise_rho 0.2 --finetune true 
python main_evaluate_2.py --dataset email --model MLP --method Margin_11 --lr 0.01 --epochs 150 --lambda_plus 0.9 --lambda_minus 0.7 --alpha_plus 0.4 --alpha_minus 0.4 --scheduler none --noise_rho 0.2 --finetune true


python main_evaluate_2.py --dataset credit --model MLP --method CE --lr 0.005 --epochs 300 --noise_rho 0.2
python main_evaluate_2.py --dataset credit --model MLP --method FOCAL --lr 0.001 --epochs 300 --noise_rho 0.2
python main_evaluate_2.py --dataset credit --model SVM --method SVM --lr 0.001 --epochs 300 --lambda_reg 0.1 --noise_rho 0.2
python main_evaluate_2.py --dataset credit --model MLP --method GCE --lr 0.005 --epochs 300 --noise_rho 0.2 

python main_evaluate_2.py --dataset credit --model MLP --method Margin_8 --lr 0.005 --epochs 150 --lambda_plus 0.0 --lambda_minus 0.3 --alpha_plus 0.1 --alpha_minus 0.1 --scheduler none --noise_rho 0.2 --finetune true
python main_evaluate_2.py --dataset credit --model MLP --method Margin_9 --lr 0.001 --epochs 150 --lambda_plus 0.05 --lambda_minus 0.3 --alpha_plus 0.3 --alpha_minus 0.1 --scheduler none --noise_rho 0.2 --finetune true
python main_evaluate_2.py --dataset credit --model SVM --method Margin_10 --lr 0.001 --epochs 150 --lambda_reg 0.0 --lambda_plus 0.0 --lambda_minus 1.0 --alpha_plus 0.1 --alpha_minus 0.5 --scheduler none --noise_rho 0.2 --finetune true 
python main_evaluate_2.py --dataset credit --model MLP --method Margin_11 --lr 0.0005 --epochs 150 --lambda_plus 0.0 --lambda_minus 0.3 --alpha_plus 0.1 --alpha_minus 0.1 --scheduler none --noise_rho 0.2 --finetune true 
