import comet_ml
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from rich import print
from functools import partial

from MobileSPEEDv3.utils.config import get_config
from MobileSPEEDv3.utils.dataset import SpeedDataModule, prepare_Speed
from MobileSPEEDv3.module.Lightning_MobileSPEEDv3 import LightningMobileSPEEDv3

from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.plugins import MixedPrecision, DoublePrecision, Precision, BitsandbytesPrecision, HalfPrecision
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, ModelSummary, RichModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.profilers import SimpleProfiler


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # ====================配置====================
    config = get_config()
    dirpath = f"./result/{config['name']}-{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}"
    # 判断是否存在路径 若不存在则创建
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # 设置随机种子
    seed_everything(config["seed"], workers=True)


    # ===================训练器===================
    # =================callbacks=================
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # 进度条
    if config["bar"] == "rich":
        bar = RichProgressBar()
    elif config["bar"] == "tqdm":
        bar = TQDMProgressBar()
    # 保存模型
    checkpoint = ModelCheckpoint(dirpath=dirpath,
                                 filename="{epoch}-best",
                                 monitor="val/score",
                                 verbose=True,
                                 save_last=True,
                                 mode="min",
                                 every_n_train_steps="10",
                                 save_weights_only=True)
    # 监控设备状态
    device_monitor = DeviceStatsMonitor(cpu_stats=None)
    # 模型总结
    if config["summary"] == "rich":
        summary = RichModelSummary(max_depth=3)
    elif config["summary"] == "default":
        summary = ModelSummary(max_depth=3)
    callbacks = [lr_monitor, checkpoint, summary, bar]

    # ===================plugins=================
    plugins = []
    # 精度
    if config["precision"] == "mix":
        precision = MixedPrecision(precision="16-mixed", device="cuda" if config["accelerator"] == "gpu" else config["accelerator"])
    elif config["precision"] == "full":
        precision = Precision()
    elif config["precision"] == "double":
        precision = DoublePrecision()
    elif config["precision"] == "half":
        precision = HalfPrecision()
    elif config["precision"] == "int8":
        precision = BitsandbytesPrecision("int8")
    plugins = [precision]
    
    # ===================logger==================
    comet_logger = CometLogger(
        api_key=config["comet_api"],
        save_dir=dirpath,
        project_name="MobileSPEEDv3",
        experiment_name=config["name"] + "-" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        offline=config["offline"]
    )
    
    # =================profiler==================
    profiler = SimpleProfiler(dirpath=dirpath)


    # ===================trainer=================
    if config["debug"]:
        limit_train_batches, limit_val_batches = config["limit_train_val"]
    else:
        limit_train_batches, limit_val_batches = 1.0, 1.0
    trainer = Trainer(accelerator=config["accelerator"],        # 加速器
                      logger=comet_logger,
                      callbacks=callbacks,
                      max_epochs=config["epoch"],
                      limit_train_batches=limit_train_batches,
                      limit_val_batches=limit_val_batches,
                      accumulate_grad_batches=config["accumulate_grad_batches"],
                      deterministic=config["deterministic"],
                      benchmark=config["benchmark"],
                    #   profiler=profiler,
                      plugins=plugins,
                    #   precision=precision,
                      default_root_dir=dirpath,
                      num_sanity_val_steps=0)

    # ====================模型====================
    # TODO Efficient initialization
    with trainer.init_module():
        module = LightningMobileSPEEDv3(config)
    
    # ====================数据====================
    dataloader = SpeedDataModule(config=config)

    # ====================训练====================
    if config["train"]:
        trainer.fit(model=module, datamodule=dataloader)

    # ====================验证====================
    if config["val"]:
        module = LightningMobileSPEEDv3.load_from_checkpoint(module.trainer.callbacks[3].best_model_path)
        trainer.validate(model=module, datamodule=dataloader)
    
    # ====================剪枝====================
    # if config["pruning"]:
    #     module = LightningMobileSPEEDNet(config)
    #     module.model.load_state_dict(torch.load("BiFPN-GIoU-best.pt"))
    #     example_inputs = torch.randn(1, 3, 240, 384).to(module.device)
    #     iterative_steps = 1
    #     all_layer = list(module.model.modules())
        
    #     if config["pruning_method"] == "slim":
    #         imp = tp.importance.BNScaleImportance()
    #         pruner_entry = partial(tp.pruner.BNScalePruner, global_pruning=False)
    #     elif config["pruning_method"] == "l1":
    #         imp = tp.importance.MagnitudeImportance(p=1)
    #         pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=False)
    #     elif config["pruning_method"] == "group":
    #         imp = tp.importance.GroupNormImportance(p=2)
    #         pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=False)
    #     elif config["pruning_method"] == "grow":
    #         imp = tp.importance.GroupNormImportance(p=2)
    #         pruner_entry = partial(tp.pruner.GrowingRegPruner, global_pruning=False)
        
    #     pruner = pruner_entry(
    #             module,
    #             example_inputs,
    #             importance=imp,
    #             iterative_steps=iterative_steps,
    #             pruning_ratio=0.5,
    #             pruning_ratio_dict={},
    #             ignored_layers=all_layer[218:],
    #             unwrapped_parameters=[]
    #         )
        
    #     module.eval()
    #     original_ops, original_size = tp.utils.count_ops_and_params(module, example_inputs)
    #     trainer.validate(model=module, datamodule=dataloader)
    #     print("Pruning...")
    #     for i in range(iterative_steps):
    #         pruner.step()
    #         pruned_ops, pruned_size = tp.utils.count_ops_and_params(module, example_inputs)
    #         print(
    #             "Iter %d/%d, Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
    #             original_size / 1e6, pruned_size / 1e6, pruned_size / original_size * 100)
    #         )
    #         print(
    #             "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
    #             original_ops / 1e6,
    #             pruned_ops / 1e6,
    #             pruned_ops / original_ops * 100,
    #             original_ops / pruned_ops,
    #             )
    #         )
    #         module.eval()
    #         trainer.validate(model=module, datamodule=dataloader)
    #         print("========================================================")
    #     trainer = Trainer(accelerator=config["accelerator"],        # 加速器
    #                   logger=comet_logger,
    #                   callbacks=callbacks,
    #                   max_epochs=config["epoch"],
    #                   limit_train_batches=limit_train_batches,
    #                   limit_val_batches=limit_val_batches,
    #                   accumulate_grad_batches=config["accumulate_grad_batches"],
    #                   deterministic=config["deterministic"],
    #                   benchmark=config["benchmark"],
    #                 #   profiler=profiler,
    #                   plugins=plugins,
    #                 #   precision=precision,
    #                   default_root_dir=dirpath,
    #                   num_sanity_val_steps=0)
    #     trainer.fit(module, dataloader)
    # # git clone https://a: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@github.com/zhouhang2000nudt/Mobile-SPEEDNet.git