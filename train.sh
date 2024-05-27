#! bin/bash

start_time=$(date +%s)   #记录开始时间

# strid为10 不再往外分配
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 0 --ratio 0

# strid为10 往外分配1个
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 1 --ratio 0.10
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 1 --ratio 0.20
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 1 --ratio 0.30

# strid为10 往外分配2个
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 2 --ratio 0.10
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 2 --ratio 0.20
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 2 --ratio 0.30

# strid为10 往外分配3个
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 3 --ratio 0.10
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 3 --ratio 0.20
python3 train.py --Rotatep 1.0 --stride 5 --neighbor 3 --ratio 0.30

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "build kernel time is $(($cost_time/60))min $(($cost_time%60))s"

/usr/bin/shutdown