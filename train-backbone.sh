#! bin/bash

start_time=$(date +%s)   #记录开始时间

# strid为10 不再往外分配
python3 train.py --backbone mobilenetv3_large_100.ra_in1k
python3 train.py --backbone rexnet_150.nav_in1k
python3 train.py --backbone resnet18.a1_in1k
python3 train.py --backbone resnet34d.ra2_in1k
python3 train.py --backbone efficientnet_b4.ra2_in1k
python3 train.py --backbone efficientnetv2_rw_s.ra2_in1k

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "build kernel time is $(($cost_time/60))min $(($cost_time%60))s"

/usr/bin/shutdown