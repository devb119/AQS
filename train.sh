CUDA_VISIBLE_DEVICES=1 python train_linear.py \
--dataset "beijing" \
--group_name feature_selection \
--satellite_in_features 10 \
--num_epochs_stdgi 0 \
--decoder_epochs 50 \
--n_iterations 300 \
--name "miss_pressure" \
--lr_stdgi 0.001 \
--use_wandb
