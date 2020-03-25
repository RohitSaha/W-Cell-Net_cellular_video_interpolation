#source /home/moseslab/.bashrc

#loop through the possible experiments that can be executed

limit_IF=7
limit_out_channels=32

for (( n_IF=4; n_IF<=$limit_IF; n_IF++ ))
    do
        for (( out_channels=8; out_channels<=$limit_out_channels; out_channels=out_channels*2 ))
            do
                bash run_skip_conn_separate_encoder_bipn.sh $out_channels $n_IF
                echo "Program OC: $out_channels IF: $n_IF complete"
                python timer.py
            done
    done
