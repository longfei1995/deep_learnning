"""
使用训练好的AlexNet模型对单张图片进行分类
用法: python predict_image.py --image path/to/image.png
"""

import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import alexnet
from collections import OrderedDict

# CIFAR-100的100个类别名称
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

def load_model(model_path, device):
    """加载训练好的模型"""
    model = alexnet()
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理checkpoint格式
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "Unknown")
        accuracy = checkpoint.get("best_prec1", "Unknown")
        print(f"加载模型: Epoch {epoch}, 准确率 {accuracy}")
    else:
        state_dict = checkpoint
    
    # 处理DataParallel保存的模型
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # 去掉'module.'前缀
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path):
    """
    预处理图片，使其符合CIFAR-100的输入要求
    CIFAR-100图片是32x32，归一化参数为ImageNet标准
    """
    # 定义预处理转换（和训练时一致）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-100是32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100的均值
                           std=[0.2675, 0.2565, 0.2761])      # CIFAR-100的标准差
    ])
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    print(f"原始图片尺寸: {image.size}")
    
    # 预处理
    image_tensor = transform(image)
    
    # 添加batch维度 [C, H, W] -> [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image

def predict(model, image_tensor, device, top_k=5):
    """
    对图片进行预测
    返回top-k个最可能的类别及其概率
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        # 使用softmax获取概率
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # 获取top-k预测
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    return top_probs[0], top_indices[0]

def main():
    parser = argparse.ArgumentParser(description='使用AlexNet对图片进行分类')
    parser.add_argument('--image', type=str, required=True, help='要分类的图片路径')
    parser.add_argument('--model', type=str, default='./checkpoints/cifar100_best.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--top_k', type=int, default=5, help='显示前K个预测结果')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("=" * 70)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = load_model(args.model, device)
    print("模型加载成功！")
    print("=" * 70)
    
    # 加载和预处理图片
    print(f"加载图片: {args.image}")
    try:
        image_tensor, original_image = preprocess_image(args.image)
    except Exception as e:
        print(f"❌ 加载图片失败: {e}")
        return
    
    print("图片预处理完成")
    print("=" * 70)
    
    # 预测
    print("正在预测...")
    top_probs, top_indices = predict(model, image_tensor, device, args.top_k)
    
    # 显示结果
    print("\n🎯 预测结果:")
    print("=" * 70)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        class_name = CIFAR100_CLASSES[idx]
        probability = prob.item() * 100
        
        if i == 0:
            print(f"🏆 Top-{i+1}: {class_name:20s} - 置信度: {probability:6.2f}%")
        else:
            print(f"   Top-{i+1}: {class_name:20s} - 置信度: {probability:6.2f}%")
    
    print("=" * 70)
    
    # 显示最终预测
    best_class = CIFAR100_CLASSES[top_indices[0]]
    best_prob = top_probs[0].item() * 100
    print(f"\n✅ 最终预测: {best_class.upper()} (置信度: {best_prob:.2f}%)")

if __name__ == "__main__":
    main()

