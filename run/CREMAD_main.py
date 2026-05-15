import os
import shutil

import torch
import torch.nn as nn

from tasks.CREMAD_task import CREMAD_Task
from utils.checkpoint_tools import load_checkpoint_state_dict, resolve_checkpoint_path
from utils.function_tools import get_device, get_logger, save_config, set_seed, weight_init


def _forward_by_modality(model, modality, spec, image):
    if modality == "Audio":
        return model(spec)
    if modality == "Visual":
        return model(image)
    return model(spec, image)


def train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    logger,
    cfgs,
    epoch,
    device,
    label_smoothing=0.0,
):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model.train()
    model.mode = "train"
    total_correct = 0.0
    total = 0.0

    for spec, image, label in train_dataloader:
        spec = spec.to(device).unsqueeze(1)
        image = image.to(device)
        label = label.to(device)

        out = _forward_by_modality(model, cfgs.modality, spec, image)
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(out, dim=1)
        total_correct += (pred == label).sum().item()
        total += label.size(0)

    train_acc = total_correct / total
    if scheduler is not None:
        scheduler.step()

    logger.info(
        "[{:3d}/{:3d}] {} {} Acc: {:.2f}%".format(
            epoch, cfgs.epochs, "Train", cfgs.modality, train_acc * 100
        )
    )


@torch.no_grad()
def test(model, dataloader, logger, cfgs, epoch, device, mode="Train"):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    num_batch = len(dataloader)
    test_loss = 0.0
    total_correct = 0.0
    total = 0.0

    for spec, image, label in dataloader:
        spec = spec.to(device).unsqueeze(1)
        image = image.to(device)
        label = label.to(device)

        out = _forward_by_modality(model, cfgs.modality, spec, image)
        loss = loss_fn(out, label)
        pred = torch.argmax(out, dim=1)
        total_correct += (pred == label).sum().item()
        total += label.size(0)
        test_loss += loss.item() / num_batch

    test_acc = total_correct / total
    logger.info(
        "[{:3d}/{:3d}] {} {} Acc: {:.2f}% Loss: {:.4f}".format(
            epoch, cfgs.epochs, mode, cfgs.modality, test_acc * 100, test_loss
        )
    )
    return test_acc


def CREMAD_main(cfgs, seed):
    set_seed(seed, master_seed=getattr(cfgs, "master_seed", None))
    cfgs.seed = int(seed)
    save_dir = os.path.join(cfgs.expt_dir, cfgs.dataset, f"{cfgs.expt_name}", f"{seed}")
    is_test_mode = cfgs.mode == "test"
    if is_test_mode:
        os.makedirs(save_dir, exist_ok=True)
    else:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    if not is_test_mode:
        save_config(cfgs, save_dir)
    logger = get_logger("test_logger" if is_test_mode else "train_logger", logger_dir=save_dir)

    device = get_device()
    logger.info(vars(cfgs))
    logger.info(f"Processing ID:{os.getpid()}, Device:{device}, System-dependence Version:{os.uname()}")

    task = CREMAD_Task(cfgs)
    train_dataloader = task.train_dataloader
    valid_dataloader = task.valid_dataloader
    test_dataloader = task.test_dataloader

    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler

    model.apply(weight_init)
    model.to(device)

    if is_test_mode:
        checkpoint_path = resolve_checkpoint_path(cfgs, save_dir)
        load_checkpoint_state_dict(model, checkpoint_path, device)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        model.mode = "eval"
        model.eval()
        test_acc = test(model, test_dataloader, logger, cfgs, 0, device, mode="Test")
        return test_acc, test_acc

    best_epoch = {"epoch": 0, "acc": 0.0, "test_acc": 0.0}
    start_epoch = 1

    for epoch in range(start_epoch, cfgs.epochs + 1):
        logger.info(f"Training for epoch {epoch}...")
        train(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            logger,
            cfgs,
            epoch,
            device,
            label_smoothing=cfgs.label_smoothing,
        )

        logger.info(f"Validating for epoch {epoch}...")
        val_acc = test(model, valid_dataloader, logger, cfgs, epoch, device, mode="Valid")
        test_acc = test(model, test_dataloader, logger, cfgs, epoch, device, mode="Test")

        if val_acc > best_epoch["acc"]:
            best_epoch["acc"] = val_acc
            best_epoch["epoch"] = epoch
            best_epoch["test_acc"] = test_acc

            if cfgs.save_checkpoint:
                torch.save(
                    {
                        "epoch": best_epoch["epoch"],
                        "state_dict": model.state_dict(),
                        "best_acc": best_epoch["acc"],
                        "test_acc_at_best_valid": best_epoch["test_acc"],
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(save_dir, "ckpt_full_epoch_best.pth.tar"),
                )

        logger.info(
            "Best Epoch: {}, Validation Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(
                best_epoch["epoch"], best_epoch["acc"] * 100, best_epoch["test_acc"] * 100
            )
        )

    logger.info("Best Epoch: {}, Validation Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(best_epoch["epoch"], best_epoch["acc"] * 100, best_epoch["test_acc"] * 100))
    return test_acc, best_epoch["test_acc"]
