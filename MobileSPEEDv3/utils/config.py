import yaml

def get_config(cfg_path: str = "MobileSPEEDv3/cfg/base.yaml"):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config
