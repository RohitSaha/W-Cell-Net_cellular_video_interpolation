#source /home/moseslab/.bashrc

#loop through the possible experiments that can be executed

limit_IF=7
exp_name='slack_20px_fluorescent_window_'

for (( n_IF=5; n_IF<=$limit_IF; n_IF++ ))
    do
        bash run_slomo.sh $n_IF $exp_name$(($n_IF+2)) 
        echo "Program OC: $out_channels IF: $n_IF complete"
        python timer.py
    done


