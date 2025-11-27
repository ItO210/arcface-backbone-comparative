import argparse
import logging
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

assert torch.__version__ >= "1.12.0"

rank = 0
local_rank = 0
world_size = 1

def main(args):

    cfg = get_config(args.config)

    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard")) if rank == 0 else None

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB key must be provided in config.")
            print(f"Error: {e}")
        run_name = datetime.now().strftime("%y%m%d_%H%M")
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                sync_tensorboard=True,
                resume=cfg.wandb_resume,
                name=run_name,
                notes=cfg.notes
            ) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB entity/project must be provided.")
            print(f"Error: {e}")

    train_loader = get_dataloader(
        root_dir=cfg.rec,
        local_rank=0,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        num_workers=cfg.num_workers
    )

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(device)
    backbone.train()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    module_partial_fc = PartialFC_V2(
        margin_loss, cfg.embedding_size, cfg.num_classes,
        cfg.sample_rate, False
    ).train().to(device)

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError("Unsupported optimizer: " + str(cfg.optimizer))

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step
    )

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        ckpt = torch.load(os.path.join(cfg.output, "checkpoint.pt"), map_location=device)
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]

        try:
            backbone.load_state_dict(ckpt["state_dict_backbone"])
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in ckpt["state_dict_backbone"].items()}
            backbone.load_state_dict(new_state)

        module_partial_fc.load_state_dict(ckpt["state_dict_softmax_fc"])
        opt.load_state_dict(ckpt["state_optimizer"])
        lr_scheduler.load_state_dict(ckpt["state_lr_scheduler"])
        del ckpt

    for key, value in cfg.items():
        logging.info(f"{key:<25}: {value}")

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger=wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent, total_step=cfg.total_step,
        batch_size=cfg.batch_size, start_step=global_step,
        writer=summary_writer
    )

    loss_meter = AverageMeter()
    amp = torch.amp.GradScaler(device="cuda", growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        for _, (img, labels) in enumerate(train_loader):
            global_step += 1
            img = img.to(device) if not isinstance(img, (list, tuple)) else [x.to(device) for x in img]
            labels = labels.to(device)

            embeddings = backbone(img)
            loss = module_partial_fc(embeddings, labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    lr_scheduler.step()
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    lr_scheduler.step()
                    opt.zero_grad()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step': loss.item(),
                        'Loss/Avg': loss_meter.avg,
                        'Step': global_step,
                        'Epoch': epoch
                    })
                loss_meter.update(loss.item(), 1)
                callback_logging(global_step, loss_meter, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, "checkpoint.pt"))

        if rank == 0:
            path_model = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.state_dict(), path_model)

            if wandb_logger and cfg.save_artifacts:
                artifact = wandb.Artifact(f"{run_name}_E{epoch}", type="model")
                artifact.add_file(path_model)
                wandb_logger.log_artifact(artifact)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Single-GPU ArcFace Training (No DALI)")
    parser.add_argument("config", type=str, help="Path to config file")
    main(parser.parse_args())
