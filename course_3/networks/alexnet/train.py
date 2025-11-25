import argparse
import os
import random
import time

import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader

from model import alexnet
from utils.adjust import adjust_learning_rate
from utils.datasets import load_datasets
from utils.eval import accuracy
from utils.misc import AverageMeter

# 解析命令行参数
parser = argparse.ArgumentParser(description="AlexNet training and evaluation")
parser.add_argument(
    "--dataroot",
    type=str,
    default="~/pytorch_datasets",
    help="download train dataset path.",
)
parser.add_argument(
    "--datasets",
    type=str,
    default="cifar100",
    help="cifar10/cifar100 datasets. default=`cifar100`",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Every train dataset size (CPU建议256-512).",
)
parser.add_argument(
    "--lr", type=float, default=0.0001, help="starting lr, every 10 epoch decay 10."
)
parser.add_argument("--epochs", type=int, default=50, help="Train loop")
parser.add_argument(
    "--phase", type=str, default="eval", help="train or eval? Default:`eval`"
)
parser.add_argument("--model_path", type=str, default="", help="load model path.")
parser.add_argument(
    "--resume", type=str, default="", help="path to checkpoint for resuming training."
)
parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=10,
    help="Stop training if no improvement for N epochs (0=disabled)",
)
opt = parser.parse_args()
print(opt)

try:
    os.makedirs("./checkpoints")
except OSError:
    pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

# setup gpu driver
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataloader, test_dataloader = load_datasets(
    opt.datasets, opt.dataroot, opt.batch_size
)

# Load model
if opt.datasets == "cifar100":
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(alexnet())
    else:
        model = alexnet()
else:
    model = ""
    print(opt)

model.to(device)
print(model)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)


def train(train_dataloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_dataloader):

        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute output
        output = model(inputs)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            print(
                f"Epoch [{epoch + 1}] [{i}/{len(train_dataloader)}]\t"
                f"Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {loss.item():.4f}\t"
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})",
                end="\r",
            )


def test(model):
    # switch to evaluate mode
    model.eval()
    # init value
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def save_checkpoint(state, filename="checkpoint.pth"):
    """保存完整的训练状态"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer):
    """加载训练状态"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint["epoch"]
        best_prec1 = checkpoint["best_prec1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            f"Loaded checkpoint (epoch {start_epoch}, best accuracy: {best_prec1:.2f}%)"
        )
        return start_epoch, best_prec1
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0.0


def run():
    # 检查是否从checkpoint恢复训练
    start_epoch = 0
    best_prec1 = 0.0
    patience_counter = 0  # Early stopping计数器

    if opt.resume:
        start_epoch, best_prec1 = load_checkpoint(opt.resume, model, optimizer)
    else:
        # 自动检查最新的checkpoint
        latest_checkpoint = f"./checkpoints/{opt.datasets}_latest.pth"
        if os.path.exists(latest_checkpoint):
            print(f"Found existing checkpoint: {latest_checkpoint}")
            response = input("是否从上次训练继续? (y/n): ")
            if response.lower() == "y":
                start_epoch, best_prec1 = load_checkpoint(
                    latest_checkpoint, model, optimizer
                )

    for epoch in range(start_epoch, opt.epochs):
        # train for one epoch
        print(f"\nBegin Training Epoch {epoch + 1}")
        train(train_dataloader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        print(f"Begin Validation @ Epoch {epoch + 1}")
        prec1 = test(model)

        # remember best prec@1 and save checkpoint if desired
        is_best = prec1 > best_prec1

        if is_best:
            best_prec1 = prec1
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1  # 没有提升，计数器+1

        print("Epoch Summary: ")
        print(f"\tEpoch Accuracy: {prec1:.2f}%")
        print(f"\tBest Accuracy: {best_prec1:.2f}%")

        # Early Stopping检查
        if opt.early_stop_patience > 0:
            print(
                f"\tNo improvement for {patience_counter} epochs (patience: {opt.early_stop_patience})"
            )
            if patience_counter >= opt.early_stop_patience:
                print(
                    f"\n⚠️  Early Stopping触发！连续{opt.early_stop_patience}个epoch无提升，停止训练。"
                )
                print(
                    f"最佳准确率: {best_prec1:.2f}% (Epoch {epoch + 1 - patience_counter})"
                )
                break

        # 保存checkpoint（包含完整训练状态）
        checkpoint_state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
        }

        # 保存最新的checkpoint
        save_checkpoint(checkpoint_state, f"./checkpoints/{opt.datasets}_latest.pth")

        # 每5个epoch保存一次独立的checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                checkpoint_state, f"./checkpoints/{opt.datasets}_epoch_{epoch + 1}.pth"
            )

        # 如果是最佳模型，单独保存
        if is_best:
            save_checkpoint(checkpoint_state, f"./checkpoints/{opt.datasets}_best.pth")


if __name__ == "__main__":
    if opt.phase == "train":
        run()
    elif opt.phase == "eval":
        if opt.model_path != "":
            print("Loading model...\n")
            model.load_state_dict(
                torch.load(opt.model_path, map_location=lambda storage, loc: storage)
            )
            print("Loading model successful!")
            accuracy = test(model)
            print(
                f"\nAccuracy of the network on the 10000 test images: {accuracy:.2f}%.\n"
            )
        else:
            print(
                "WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH"
            )
    else:
        print(opt)
