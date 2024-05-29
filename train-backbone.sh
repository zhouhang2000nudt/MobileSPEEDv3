#! bin/bash

start_time=$(date +%s)   #记录开始时间

export HF_ENDPOINT=https://hf-mirror.com

# strid为10 不再往外分配

# resnet
python3 train.py --backbone resnet34.a1_in1k               # 21.8M 3.7GMACs
python3 train.py --backbone resnet50.a1_in1k               # 25.6M 4.1GMACs
python3 train.py --backbone resnet101.a1_in1k             # 44.5M 7.8GMACs

# resnetv2
python3 train.py --backbone resnetv2_50.a1h_in1k       # 25.5M 4.1GMACs
python3 train.py --backbone resnetv2_101.a1h_in1k     # 44.5M 7.8GMACsc

# renext
python3 train.py --backbone rexnet_100.nav_in1k         # 4.8M 0.4GMACsc
python3 train.py --backbone rexnet_130.nav_in1k         # 7.6M 0.7GMACsc
python3 train.py --backbone rexnet_150.nav_in1k         # 9.7M 0.9GMACsc
python3 train.py --backbone rexnet_200.nav_in1k         # 16.4M 1.6GMACsc
python3 train.py --backbone rexnet_300.nav_in1k         # 34.7M 3.4GMACsc

# efficientnetv2
python3 train.py --backbone tf_efficientnetv2_b0.in1k     # 7.1M 0.5GMACsc
python3 train.py --backbone tf_efficientnetv2_b1.in1k     # 8.1M 0.8GMACsc
python3 train.py --backbone tf_efficientnetv2_b2.in1k     # 10.1M 1.1GMACsc
python3 train.py --backbone tf_efficientnetv2_b3.in1k     # 14.4M 1.9GMACsc

# repvit
python3 train.py --backbone repvit_m0_9.dist_450e_in1k    # 5.5M 0.8GMACsc
python3 train.py --backbone repvit_m1_0.dist_450e_in1k    # 7.3M 1.1GMACsc
python3 train.py --backbone repvit_m1_1.dist_450e_in1k    # 8.8M 1.4GMACsc
python3 train.py --backbone repvit_m1_5.dist_450e_in1k    # 14.6M 2.3GMACsc
python3 train.py --backbone repvit_m2_3.dist_450e_in1k    # 23.7M 4.6GMACsc

# efficientformer
python3 train.py --backbone  timm/efficientformerv2_s0.snap_dist_in1k --local-dir pretrained/efficientformerv2_s0.snap_dist_in1k    # 3.6M 0.4GMACsc
python3 train.py --backbone  timm/efficientformerv2_s1.snap_dist_in1k --local-dir pretrained/efficientformerv2_s1.snap_dist_in1k    # 6.2M 0.7GMACsc
python3 train.py --backbone  timm/efficientformerv2_s2.snap_dist_in1k --local-dir pretrained/efficientformerv2_s2.snap_dist_in1k    # 12.7M 1.3GMACsc

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "build kernel time is $(($cost_time/60))min $(($cost_time%60))s"

/usr/bin/shutdown