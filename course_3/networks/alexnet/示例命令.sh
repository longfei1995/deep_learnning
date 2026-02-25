#!/bin/bash
# AlexNet图片分类示例命令

echo "================================"
echo "AlexNet CIFAR-100 图片分类示例"
echo "================================"

# 1. 生成示例图片
echo -e "\n1. 生成示例图片..."
python demo_save_test_image.py

# 2. 对示例图片进行分类
echo -e "\n2. 对示例图片进行分类..."
for img in demo_images/test_*.png; do
    echo -e "\n--- 分类图片: $img ---"
    python predict_image.py --image "$img" --top_k 3
done

echo -e "\n================================"
echo "示例完成！"
echo "================================"
echo ""
echo "你现在可以使用自己的图片："
echo "  python predict_image.py --image your_image.png"

