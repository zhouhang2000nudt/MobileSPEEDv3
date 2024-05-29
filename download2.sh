#! bin/bash

wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh

export HF_ENDPOINT=https://hf-mirror.com

# resnet
./hfd.sh timm/resnet34.a1_in1k --tool aria2c -x 4               # 21.8M 3.7GMACs
./hfd.sh timm/resnet50.a1_in1k --tool aria2c -x 4               # 25.6M 4.1GMACs
./hfd.sh timm/resnet101.a1_in1k --tool aria2c -x 4             # 44.5M 7.8GMACs

# resnetv2
./hfd.sh timm/resnetv2_50.a1h_in1k --tool aria2c -x 4       # 25.5M 4.1GMACs
./hfd.sh timm/resnetv2_101.a1h_in1k --tool aria2c -x 4     # 44.5M 7.8GMACsc

# renext
./hfd.sh timm/rexnet_100.nav_in1k --tool aria2c -x 4         # 4.8M 0.4GMACsc
./hfd.sh timm/rexnet_130.nav_in1k --tool aria2c -x 4         # 7.6M 0.7GMACsc
./hfd.sh timm/rexnet_150.nav_in1k --tool aria2c -x 4         # 9.7M 0.9GMACsc
./hfd.sh timm/rexnet_200.nav_in1k --tool aria2c -x 4         # 16.4M 1.6GMACsc
./hfd.sh timm/rexnet_300.nav_in1k --tool aria2c -x 4         # 34.7M 3.4GMACsc

# efficientnetv2
./hfd.sh timm/tf_efficientnetv2_b0.in1k --tool aria2c -x 4     # 7.1M 0.5GMACsc
./hfd.sh timm/tf_efficientnetv2_b1.in1k --tool aria2c -x 4     # 8.1M 0.8GMACsc
./hfd.sh timm/tf_efficientnetv2_b2.in1k --tool aria2c -x 4     # 10.1M 1.1GMACsc
./hfd.sh timm/tf_efficientnetv2_b3.in1k --tool aria2c -x 4     # 14.4M 1.9GMACsc

# repvit
./hfd.sh timm/repvit_m0_9.dist_450e_in1k --tool aria2c -x 4    # 5.5M 0.8GMACsc
./hfd.sh timm/repvit_m1_0.dist_450e_in1k --tool aria2c -x 4    # 7.3M 1.1GMACsc
./hfd.sh timm/repvit_m1_1.dist_450e_in1k --tool aria2c -x 4    # 8.8M 1.4GMACsc
./hfd.sh timm/repvit_m1_5.dist_450e_in1k --tool aria2c -x 4    # 14.6M 2.3GMACsc
./hfd.sh timm/repvit_m2_3.dist_450e_in1k --tool aria2c -x 4    # 23.7M 4.6GMACsc

# efficientformer
./hfd.sh timm/efficientformerv2_s0.snap_dist_in1k --tool aria2c -x 4    # 3.6M 0.4GMACsc
./hfd.sh timm/efficientformerv2_s1.snap_dist_in1k --tool aria2c -x 4    # 6.2M 0.7GMACsc
./hfd.sh timm/efficientformerv2_s2.snap_dist_in1k --tool aria2c -x 4    # 12.7M 1.3GMACsc

