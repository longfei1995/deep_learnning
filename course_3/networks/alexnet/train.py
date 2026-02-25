import argparse
import os
import random
import time
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader

from model import alexnet
from utils.adjust import adjust_learning_rate
from utils.datasets import load_datasets
from utils.eval import accuracy
from utils.misc import AverageMeter


def create_parser():
    """创建并返回命令行参数解析器"""
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
        "--resume",
        type=str,
        default="",
        help="path to checkpoint for resuming training.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Stop training if no improvement for N epochs (0=disabled)",
    )
    return parser


def train(train_dataloader, model, criterion, optimizer, epoch, device):
    """训练一个epoch

    Args:
        train_dataloader: 训练数据加载器
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        epoch: 当前epoch编号
        device: 训练设备(cuda/cpu)
    """
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


def test(model, test_dataloader, device):
    """在测试集上评估模型

    Args:
        model: 模型
        test_dataloader: 测试数据加载器
        device: 评估设备(cuda/cpu)

    Returns:
        accuracy: 准确率(百分比)
    """
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
    """保存完整的训练状态

    Args:
        state: 包含训练状态的字典
        filename: 保存路径
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """加载训练状态

    Args:
        checkpoint_path: checkpoint文件路径
        model: 模型
        optimizer: 优化器
        device: 设备(cuda/cpu)

    Returns:
        start_epoch: 起始epoch
        best_prec1: 最佳准确率
    """
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


def run_training(
    model, train_dataloader, test_dataloader, criterion, optimizer, opt, device
):
    """执行完整的训练流程

    Args:
        model: 模型
        train_dataloader: 训练数据加载器
        test_dataloader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        opt: 命令行参数
        device: 训练设备(cuda/cpu)
    """
    # 检查是否从checkpoint恢复训练
    start_epoch = 0
    best_prec1 = 0.0
    patience_counter = 0  # Early stopping计数器

    if opt.resume:
        start_epoch, best_prec1 = load_checkpoint(opt.resume, model, optimizer, device)
    else:
        # 自动检查最新的checkpoint
        latest_checkpoint = f"./checkpoints/{opt.datasets}_latest.pth"
        if os.path.exists(latest_checkpoint):
            print(f"Found existing checkpoint: {latest_checkpoint}")
            response = input("是否从上次训练继续? (y/n): ")
            if response.lower() == "y":
                start_epoch, best_prec1 = load_checkpoint(
                    latest_checkpoint, model, optimizer, device
                )

    for epoch in range(start_epoch, opt.epochs):
        # train for one epoch
        print(f"\nBegin Training Epoch {epoch + 1}")
        train(train_dataloader, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        print(f"Begin Validation @ Epoch {epoch + 1}")
        prec1 = test(model, test_dataloader, device)

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


def run_evaluation(model, test_dataloader, opt, device):
    """执行模型评估

    Args:
        model: 模型
        test_dataloader: 测试数据加载器
        opt: 命令行参数
        device: 评估设备(cuda/cpu)
    """
    if opt.model_path != "":
        print("Loading model...\n")
        checkpoint = torch.load(opt.model_path, map_location=device)

        # 处理checkpoint格式（完整checkpoint vs 仅权重）
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print(
                f"Checkpoint info: Epoch {checkpoint.get('epoch', 'Unknown')}, "
                f"Best Accuracy: {checkpoint.get('best_prec1', 'Unknown')}"
            )
        else:
            state_dict = checkpoint

        # 处理DataParallel保存的模型
        if list(state_dict.keys())[0].startswith("module."):
            print("检测到DataParallel模型，正在转换...")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # 去掉'module.'前缀
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print("Loading model successful!")
        accuracy = test(model, test_dataloader, device)
        print(f"\nAccuracy of the network on the 10000 test images: {accuracy:.2f}%.\n")
    else:
        print(
            "WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH"
        )


def main():
    """主函数：所有执行逻辑的入口"""
    # 1. 解析命令行参数
    parser = create_parser()
    opt = parser.parse_args()
    print(opt)

    # 2. 创建checkpoints目录
    os.makedirs("./checkpoints", exist_ok=True)

    # 3. 设置随机种子
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # 4. 配置CUDA
    cudnn.benchmark = True

    # 5. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 6. 加载数据集
    train_dataloader, test_dataloader = load_datasets(
        opt.datasets, opt.dataroot, opt.batch_size
    )

    # 7. 创建模型H H
    if opt.datasets == "cifar100":
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(alexnet())
        else:
            model = alexnet()
    else:
        raise ValueError(f"Unsupported dataset: {opt.datasets}")

    model.to(device)
    print(model)

    # 8. 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # 9. 执行训练或评估
    if opt.phase == "train":
        run_training(
            model, train_dataloader, test_dataloader, criterion, optimizer, opt, device
        )
    elif opt.phase == "eval":
        run_evaluation(model, test_dataloader, opt, device)
    else:
        print(f"Unknown phase: {opt.phase}. Use 'train' or 'eval'.")
        print(opt)


if __name__ == "__main__":
    main()
