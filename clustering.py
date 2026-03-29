import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.cluster import AffinityPropagation
from tqdm import tqdm
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class Food101Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        # 递归搜索所有 jpg 格式的图片
        self.image_paths = list(self.root_dir.rglob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 食物类别为图片所在的上级文件夹名称
        class_name = img_path.parent.name
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # 遇到损坏的图片时，返回一个全零张量以保证程序继续运行
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
            
        # 提取相对于 root_dir 的相对路径用于 metadata.jsonl
        rel_path = img_path.relative_to(self.root_dir)
        # 确保路径分隔符在不同系统下都是正斜杠
        return image, str(rel_path).replace('\\', '/'), class_name

def cluster_class(class_name, features, paths):
    """
    对单个食物类别的特征进行聚类。
    该函数会被多进程调用。
    """
    features_np = np.array(features)
    
    # 使用 AffinityPropagation 聚类
    # random_state=42 用于确保结果的可重复性 (需要 scikit-learn >= 0.23)
    try:
        clustering = AffinityPropagation(random_state=42)
    except TypeError:
        # 兼容旧版本 scikit-learn
        clustering = AffinityPropagation()
        
    cluster_labels = clustering.fit_predict(features_np)
    
    entries = []
    for path, label in zip(paths, cluster_labels):
        new_text = f"{class_name}_{label}"
        entries.append({
            "file_name": path,
            "text": new_text
        })
    return entries

def main():
    data_dir = './data'
    batch_size = 128
    num_workers = 4 # DataLoader 线程数
    
    print("Checking and downloading Food-101 dataset (this may take a while if not downloaded)...")
    try:
        from torchvision.datasets import Food101
        # 自动下载数据集。如果没有下载过，会自动下载 5GB 左右的压缩包并解压
        Food101(root=data_dir, split='train', download=True)
    except Exception as e:
        print(f"Failed to download Food-101 dataset using torchvision: {e}")
        return

    # torchvision 的 Food-101 会将图片解压到 root/food-101/images
    images_dir = os.path.join(data_dir, 'food-101', 'images')
    # metadata 文件保存在 images_dir 下，以适应 diffusers 的 ImageFolder 要求
    output_file = os.path.join(images_dir, 'metadata.jsonl')
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory '{images_dir}' not found after download.")
        return

    # 1. 设备配置 (支持无显卡在 CPU 运行)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 图像预处理和 DataLoader 实例化
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = Food101Dataset(images_dir, transform=transform)
    if len(dataset) == 0:
        print(f"No '.jpg' images found in {images_dir}. Please check your dataset.")
        return
        
    print(f"Found {len(dataset)} images in dataset.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3. 加载预训练的 ResNet-18
    print("Loading pre-trained ResNet-18...")
    # 为了兼容旧版本的 torchvision，这里使用 pretrained=True
    resnet18 = models.resnet18(pretrained=True)
    
    # 去掉最后的全连接层提取特征 (输出形状为: [B, 512, 1, 1])
    modules = list(resnet18.children())[:-1]
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()

    # 4. 批量处理特征提取
    class_data = {}
    print("Extracting features...")
    
    with torch.no_grad():
        for images, paths, class_names in tqdm(dataloader, desc="Feature Extraction"):
            images = images.to(device)
            
            # 前向传播提取特征
            features = feature_extractor(images)
            # 展平特征: [B, 512, 1, 1] -> [B, 512]
            features = features.squeeze(-1).squeeze(-1)
            features = features.cpu().numpy()
            
            for feat, path, cls_name in zip(features, paths, class_names):
                if cls_name not in class_data:
                    class_data[cls_name] = {'features': [], 'paths': []}
                class_data[cls_name]['features'].append(feat)
                class_data[cls_name]['paths'].append(path)

    # 5. 对每个类别的特征进行多进程聚类
    print("Clustering features per class (using multi-processing)...")
    metadata_entries = []
    
    # 使用 ProcessPoolExecutor 进行并行聚类加速
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(cluster_class, cls_name, data['features'], data['paths']): cls_name
            for cls_name, data in class_data.items()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Clustering Classes"):
            cls_name = futures[future]
            try:
                entries = future.result()
                metadata_entries.extend(entries)
            except Exception as e:
                print(f"Clustering failed for class {cls_name}: {e}")

    # 6. 生成 HuggingFace diffusers 格式的 metadata.jsonl
    print(f"Saving metadata to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')
            
    print("Done! Preprocessing completed.")

if __name__ == "__main__":
    main()
