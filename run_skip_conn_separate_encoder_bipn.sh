#source /home/moseslab/.bashrc
CUDA_VISIBLE_DEVICES=0
python train_skip_conn_separate_encoder_bipn.py --train_iters 100000 \
    --val_every 100 \
    --save_every 1000 \
    --plot_every 1000 \
    --experiment_name 'slack_20px_fluorescent_window_5' \
    --optimizer 'adam' \
    --learning_rate 1e-3 \
    --batch_size 32 \
    --loss 'l2' \
    --weight_decay 0.0 \
    --perceptual_loss_weight 0.0 \
    --perceptual_loss_endpoint 'conv4_3' \
    --model_name 'skip_conn_separate_encoder_bipn' \
    --starting_out_channels 8 \
    --additional_info '' \
    --debug 0
