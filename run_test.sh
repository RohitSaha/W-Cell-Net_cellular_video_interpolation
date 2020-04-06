#source /home/moseslab/.bashrc

#loop through the possible experiments that can be executed

limit_window_size=9
limit_out_channels=32

for (( window_size=5; window_size<=$limit_window_size; window_size++ ))
    do
        for (( out_channels=8; out_channels<=$limit_out_channels; out_channels=out_channels*2 ))
            do
                python testing.py --window_size $window_size --out_channels $out_channels
            done
    done
