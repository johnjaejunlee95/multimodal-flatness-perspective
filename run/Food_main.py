import os
import shutil

import torch
import torch.nn as nn
from transformers import BertTokenizer

from tasks.Food_task import Food_Task
from utils.checkpoint_tools import load_checkpoint_state_dict, resolve_checkpoint_path
from utils.function_tools import get_device, get_logger, save_config, set_seed


def tokenize_batch(tokenizer, texts):
    return tokenizer(
        texts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )


def _forward_by_modality(model, modality, image, text_token):
    if modality == "Text":
        return model(text_token)
    if modality == "Visual":
        return model(image)
    return model(image, text_token)


def train(
    model,
    tokenizer,
    train_dataloader,
    optimizer,
    scheduler,
    epoch,
    cfgs,
    device,
    logger,
    label_smoothing=0.0,
):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model.train()
    total_correct = 0.0
    total = 0.0

    for image, text, label in train_dataloader:
        image = image.float().to(device)
        label = label.long().to(device)
        text_token = tokenize_batch(tokenizer, text).to(device)

        out = _forward_by_modality(model, cfgs.modality, image, text_token)
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
def test(model, tokenizer, test_dataloader, epoch, cfgs, device, logger, mode="valid"):
    loss_fn = nn.CrossEntropyLoss()
    if cfgs.modality == "Multimodal":
        model.mode = "eval"
    model.eval()

    test_loss = 0.0
    total_correct = 0.0
    total = 0.0

    num_batch = len(test_dataloader)
    for image, text, label in test_dataloader:
        image = image.float().to(device)
        label = label.long().to(device)
        text_token = tokenize_batch(tokenizer, text).to(device)

        out = _forward_by_modality(model, cfgs.modality, image, text_token)
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


def Food101Main(cfgs, seed):
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

    task = Food_Task(cfgs)
    train_dataloader = task.train_dataloader
    val_dataloader = task.val_dataloader
    test_dataloader = task.test_dataloader

    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        cache_dir="/nfs2/jjlee/model_cache",
        add_prefix_space=False,
    )

    if is_test_mode:
        checkpoint_path = resolve_checkpoint_path(cfgs, save_dir)
        load_checkpoint_state_dict(model, checkpoint_path, device)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        model.mode = "eval"
        model.eval()
        test_acc = test(model, tokenizer, test_dataloader, 0, cfgs, device, logger, mode="Test")
        return test_acc, test_acc

    best_epoch = {"epoch": 0, "acc": 0.0, "test_acc": 0.0}

    for epoch in range(1, cfgs.epochs + 1):
        logger.info(f"Training for epoch {epoch}...")
        train(
            model,
            tokenizer,
            train_dataloader,
            optimizer,
            scheduler,
            epoch,
            cfgs,
            device,
            logger,
            label_smoothing=cfgs.label_smoothing,
        )

        logger.info(f"Test for epoch {epoch}...")
        val_acc = test(model, tokenizer, val_dataloader, epoch, cfgs, device, logger, mode="Valid")
        test_acc = test(model, tokenizer, test_dataloader, epoch, cfgs, device, logger, mode="Test")

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
