"""
从CIFAR-100测试集中保存几张样本图片用于演示
"""

import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

# CIFAR-100类别名称
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

# 创建示例图片目录
os.makedirs("demo_images", exist_ok=True)

# 加载CIFAR-100测试集
print("加载CIFAR-100测试集...")
testset = torchvision.datasets.CIFAR100(
    root='~/pytorch_datasets',
    train=False,
    download=False
)

# 保存前5张图片作为演示
print("保存示例图片...")
for i in range(5):
    image, label = testset[i]
    class_name = CIFAR100_CLASSES[label]
    
    filename = f"demo_images/test_{i+1}_{class_name}.png"
    image.save(filename)
    
    print(f"保存图片 {i+1}: {filename} (真实类别: {class_name})")

print("\n✅ 完成！示例图片已保存到 demo_images/ 目录")
print("\n你可以使用以下命令测试:")
print("python predict_image.py --image demo_images/test_1_*.png")

