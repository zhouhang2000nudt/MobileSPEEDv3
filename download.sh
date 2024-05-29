#! bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# resnet
huggingface-cli timm/resnet34.a1_in1k --local-dir pretrained/resnet34.a1_in1k               # 21.8M 3.7GMACs
huggingface-cli timm/resnet50.a1_in1k --local-dir pretrained/resnet50.a1_in1k               # 25.6M 4.1GMACs
huggingface-cli timm/resnet101.a1_in1k --local-dir pretrained/resnet101.a1_in1k             # 44.5M 7.8GMACs

# resnetv2
huggingface-cli timm/resnetv2_50.a1h_in1k --local-dir pretrained/resnetv2_50.a1h_in1k       # 25.5M 4.1GMACs
huggingface-cli timm/resnetv2_101.a1h_in1k --local-dir pretrained/resnetv2_101.a1h_in1k     # 44.5M 7.8GMACsc

# renext
huggingface-cli timm/rexnet_100.nav_in1k --local-dir pretrained/rexnet_100.nav_in1k         # 4.8M 0.4GMACsc
huggingface-cli timm/rexnet_130.nav_in1k --local-dir pretrained/rexnet_130.nav_in1k         # 7.6M 0.7GMACsc
huggingface-cli timm/rexnet_150.nav_in1k --local-dir pretrained/rexnet_150.nav_in1k         # 9.7M 0.9GMACsc
huggingface-cli timm/rexnet_200.nav_in1k --local-dir pretrained/rexnet_200.nav_in1k         # 16.4M 1.6GMACsc
huggingface-cli timm/rexnet_300.nav_in1k --local-dir pretrained/rexnet_300.nav_in1k         # 34.7M 3.4GMACsc

# efficientnetv2
huggingface-cli timm/tf_efficientnetv2_b0.in1k --local-dir pretrained/tf_efficientnetv2_b0.in1k     # 7.1M 0.5GMACsc
huggingface-cli timm/tf_efficientnetv2_b1.in1k --local-dir pretrained/tf_efficientnetv2_b1.in1k     # 8.1M 0.8GMACsc
huggingface-cli timm/tf_efficientnetv2_b2.in1k --local-dir pretrained/tf_efficientnetv2_b2.in1k     # 10.1M 1.1GMACsc
huggingface-cli timm/tf_efficientnetv2_b3.in1k --local-dir pretrained/tf_efficientnetv2_b3.in1k     # 14.4M 1.9GMACsc

# repvit
huggingface-cli timm/repvit_m0_9.dist_450e_in1k --local-dir pretrained/repvit_m0_9.dist_450e_in1k    # 5.5M 0.8GMACsc
huggingface-cli timm/repvit_m1_0.dist_450e_in1k --local-dir pretrained/repvit_m1_0.dist_450e_in1k    # 7.3M 1.1GMACsc
huggingface-cli timm/repvit_m1_1.dist_450e_in1k --local-dir pretrained/repvit_m1_1.dist_450e_in1k    # 8.8M 1.4GMACsc
huggingface-cli timm/repvit_m1_5.dist_450e_in1k --local-dir pretrained/repvit_m1_5.dist_450e_in1k    # 14.6M 2.3GMACsc
huggingface-cli timm/repvit_m2_3.dist_450e_in1k --local-dir pretrained/repvit_m2_3.dist_450e_in1k    # 23.7M 4.6GMACsc

# efficientformer
huggingface-cli timm/efficientformerv2_s0.snap_dist_in1k --local-dir pretrained/efficientformerv2_s0.snap_dist_in1k    # 3.6M 0.4GMACsc
huggingface-cli timm/efficientformerv2_s1.snap_dist_in1k --local-dir pretrained/efficientformerv2_s1.snap_dist_in1k    # 6.2M 0.7GMACsc
huggingface-cli timm/efficientformerv2_s2.snap_dist_in1k --local-dir pretrained/efficientformerv2_s2.snap_dist_in1k    # 12.7M 1.3GMACsc

