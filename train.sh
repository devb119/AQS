CUDA_VISIBLE_DEVICES=0 python train_linear.py \
--dataset "beijing" \
--group_name station_experiments \
--satellite_in_features 11 \
--num_epochs_stdgi 50 \
--decoder_epochs 50 \
--n_iterations 300 \
--name "temporal_0_1_2_3_4" \
--lr_stdgi 0.001 \
--use_wandb
