#source /home/moseslab/.bashrc
CUDA_VISIBLE_DEVICES=0
out_channels=$1
n_IF=$2
exp_name=$3
python train_skip_conn_separate_encoder_bipn.py --train_iters 100000 \
    --val_every 100 \
    --save_every 10000 \
    --plot_every 1000 \
    --experiment_name $exp_name \
    --optimizer 'adam' \
    --learning_rate 1e-3 \
    --batch_size 16 \
    --loss 'l2' \
    --weight_decay 0.0 \
    --perceptual_loss_weight 0.001 \
    --perceptual_loss_endpoint 'conv5_3' \
    --model_name 'skip_conn_separate_encoder_bipn' \
    --starting_out_channels $out_channels \
    --n_IF $n_IF \
    --use_attention 0 \
    --spatial_attention 0 \
    --additional_info 'vminVmax' \
    --debug 0
