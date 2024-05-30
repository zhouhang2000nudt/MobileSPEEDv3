#! bin/bash

start_time=$(date +%s)   #记录开始时间

export HF_ENDPOINT=https://hf-mirror.com

# renext
python3 train.py --backbone rexnet_100.nav_in1k         # 4.8M 0.4GMACsc
# python3 train.py --backbone rexnet_130.nav_in1k         # 7.6M 0.7GMACsc
# python3 train.py --backbone rexnet_150.nav_in1k         # 9.7M 0.9GMACsc
# python3 train.py --backbone rexnet_200.nav_in1k         # 16.4M 1.6GMACsc
python3 train.py --backbone rexnet_300.nav_in1k         # 34.7M 3.4GMACsc

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "build kernel time is $(($cost_time/60))min $(($cost_time%60))s"

/usr/bin/shutdown