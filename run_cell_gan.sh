#source /home/moseslab/.bashrc
CUDA_VISIBLE_DEVICES=0
out_channels=$1
n_IF=$2
exp_name=$3
python train_cell_gan.py --train_iters 100000 \
    --disc_train_iters 1 \
    --val_every 100 \
    --save_every 10000 \
    --plot_every 1000 \
    --experiment_name $exp_name \
    --optimizer 'adam' \
    --learning_rate 0.0003 \
    --batch_size 32 \
    --reconstruction_loss_weight 1.0 \
    --perceptual_loss_weight 0 \
    --perceptual_loss_endpoint 'conv5_3' \
    --model_name 'gan' \
    --starting_out_channels $out_channels \
    --n_IF $n_IF \
    --use_attention 0 \
    --spatial_attention 1 \
    --additional_info 'l1-recLoss_vminVmax' \
    --discri_starting_out_channels 8 \
    --debug 0
