# cellular_video_interpolation

All experiments were executed on an NVIDIA Quadro P6000 24 GB GPU.

Default training hyperparameters (unless specified otherwise):
1. Batch size: 32
2. Training iterations: 100,000
3. Optimizer: Adam (with default values of beta1 and beta2)
4. Initial learning rate: 1e-3
5. Reconstruction loss: L2
6. Perceptual loss weight (if used): 1e-4
7. Perceptual loss end point (if used): conv5_3 from VGG16
7. L2 weight decay (if used): 1e-5

Training models:
1. Train W-Cell-Net-16 (k=16, IF=3): bash run_wnet.sh 16 3 slack_20px_fluorescent_window_5
2. Train BiPN (k-16, IF=3): python train_BiPN.py --n_IF 3 --experiment_name slack_20px_fluorescent_window_5 --batch_size 16
3. Train Super SloMo (IF=3): bash run_slowmo.sh 3 slack_20px_fluorescent_window_5

Testing models:
1. Test W-Cell-Net-16 (k=16, IF=3): python testing.py --model_name wnet --window_size 5 --out_channels 16
2. Test BiPN (k=16, IF=3): python testing.py --model_name bipn --window_size 5 --out_channels 16
3. Test Super SloMo (IF=3): python testing.py --model_name slomo --window_size 5
