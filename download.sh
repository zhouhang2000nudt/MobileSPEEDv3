#! bin/bash

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download timm/mobilenetv3_large_100.ra_in1k --local-dir pretrained/mobilenetv3_large_100
huggingface-cli download timm/rexnet_150.nav_in1k --local-dir pretrained/rexnet_150.nav_in1k
huggingface-cli download timm/resnet18.a1_in1k --local-dir pretrained/resnet18.a1_in1k
huggingface-cli download timm/efficientnet_b4.ra2_in1k --local-dir pretrained/efficientnet_b4.ra2_in1k
huggingface-cli download timm/efficientnetv2_rw_s.ra2_in1k --local-dir pretrained/efficientnetv2_rw_s.ra2_in1k
huggingface-cli download timm/resnet34d.ra2_in1k --local-dir pretrained/resnet34d.ra2_in1k
