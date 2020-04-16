#source /home/moseslab/.bashrc
CUDA_VISIBLE_DEVICES=0
n_IF=$1
exp_name=$2
python train_slomo.py --train_iters 200000 \
    --val_every 100 \
    --save_every 5000 \
    --plot_every 1000 \
    --experiment_name $exp_name \
    --optimizer 'adam' \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --loss 'l2' \
    --n_IF $n_IF \
    --model_name 'slowmo' \
    --debug 0
