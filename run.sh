python main.py --dataname Squirrel --seed 1234 --MLP_epoch 500 --MLP_hidden 64 --MLP_weight_decay 5e-4 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 300 --AE_latent_dim 200 --AE_weight_decay 5e-4 --AE_learning_rate 0.005 --GCN_epoch 500 --GCN_hidden 32 --GCN_weight_decay 5e-4 --GCN_learning_rate 0.01 --GCN_dropout 0.7 --top_k 20 --bottom_k 4 --with_latent True --device cuda:0 

# python main.py --dataname Chameleon --seed 0 --MLP_epoch 500 --MLP_hidden 128 --MLP_weight_decay 5e-4 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 200 --AE_latent_dim 128 --AE_weight_decay 5e-6 --AE_learning_rate 0.01 --GCN_epoch 500 --GCN_hidden 128 --GCN_weight_decay 5e-6 --GCN_learning_rate 0.01 --GCN_dropout 0.8 --top_k 40 --bottom_k 1 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0 | tee chameleon.txt

# python main.py --dataname Film --seed 1234 --MLP_epoch 500 --MLP_hidden 64 --MLP_weight_decay 5e-4 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 300 --AE_latent_dim 200 --AE_weight_decay 5e-4 --AE_learning_rate 0.01 --GCN_epoch 2500 --GCN_hidden 32 --GCN_weight_decay 5e-4 --GCN_learning_rate 0.01 --GCN_dropout 0.50 --top_k 70 --bottom_k 3 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0 | tee film.txt

# python main.py --dataname Cornell --seed 1234 --MLP_epoch 500 --MLP_hidden 64 --MLP_weight_decay 5e-5 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 300 --AE_latent_dim 200 --AE_weight_decay 5e-4 --AE_learning_rate 0.005 --GCN_epoch 1000 --GCN_hidden 32 --GCN_weight_decay 5e-4 --GCN_learning_rate 0.01 --GCN_dropout 0.5 --top_k 30 --bottom_k 3 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0 | tee cornell.txt

# python main.py --dataname Texas --seed 1234 --MLP_epoch 500 --MLP_hidden 64 --MLP_weight_decay 5e-5 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 300 --AE_latent_dim 200 --AE_weight_decay 5e-4 --AE_learning_rate 0.005 --GCN_epoch 1000 --GCN_hidden 32 --GCN_weight_decay 5e-4 --GCN_learning_rate 0.01 --GCN_dropout 0.5 --top_k 30 --bottom_k 3 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0 | tee texas.txt

# python main.py --dataname Wisconsin --seed 1234 --MLP_epoch 500 --MLP_hidden 64 --MLP_weight_decay 5e-5 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 300 --AE_latent_dim 200 --AE_weight_decay 5e-4 --AE_learning_rate 0.005 --GCN_epoch 500 --GCN_hidden 32 --GCN_weight_decay 5e-4 --GCN_learning_rate 0.01 --GCN_dropout 0.5 --top_k 30 --bottom_k 3 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0 | tee wisconsin.txt

# python main.py --dataname Cora --seed 1234 --MLP_epoch 500 --MLP_hidden 64 --MLP_weight_decay 5e-4 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 200 --AE_latent_dim 256 --AE_weight_decay 5e-6 --AE_learning_rate 0.01 --GCN_epoch 500 --GCN_hidden 64 --GCN_weight_decay 5e-8 --GCN_learning_rate 0.01 --GCN_dropout 0.7 --top_k 0 --bottom_k 4 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0 | tee cora.txt

# python main.py --dataname Citeseer --seed 1234 --MLP_epoch 500 --MLP_hidden 64 --MLP_weight_decay 5e-4 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 200 --AE_latent_dim 256 --AE_weight_decay 5e-6 --AE_learning_rate 0.01 --GCN_epoch 500 --GCN_hidden 64 --GCN_weight_decay 5e-8 --GCN_learning_rate 0.01 --GCN_dropout 0.7 --top_k 0 --bottom_k 4 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0 | tee citeseer.txt

# python main.py --dataname Pubmed --seed 1234 --MLP_epoch 500 --MLP_hidden 64 --MLP_weight_decay 5e-4 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 200 --AE_latent_dim 256 --AE_weight_decay 5e-6 --AE_learning_rate 0.01 --GCN_epoch 500 --GCN_hidden 64 --GCN_weight_decay 5e-8 --GCN_learning_rate 0.01 --GCN_dropout 0.7 --top_k 0 --bottom_k 4 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0 | tee pubmed.txt



# ================================================================================================================================================================


# python main.py --dataname Chameleon --seed 1234 --MLP_epoch 500 --MLP_hidden 128 --MLP_weight_decay 5e-4 --MLP_learning_rate 0.01 --MLP_dropout 0.6 --AE_epoch 200 --AE_latent_dim 128 --AE_weight_decay 5e-6 --AE_learning_rate 0.01 --GCN_epoch 500 --GCN_hidden 128 --GCN_weight_decay 5e-6 --GCN_learning_rate 0.01 --GCN_dropout 0.8 --top_k 0 --bottom_k 0 --alpha 3.0 --beta 1.5 --gamma 0.9 --with_latent True --device cuda:0