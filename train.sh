CUDA_VISIBLE_DEVICES=0 python train_linear.py \
--dataset "beijing" \
--group_name attention \
--satellite_in_features 11 \
--num_epochs_stdgi 0 \
--decoder_epochs 50 \
--n_iterations 300 \
--name "attention_feature_lr_0.0005" \
--lr_stdgi 0.0005 \
--use_wandb
