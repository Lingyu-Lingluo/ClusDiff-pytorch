import os
import json
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
from peft import PeftModel  # 新增：引入 PEFT 库用于底层加载

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_metadata(metadata_path):
    """
    解析 metadata.jsonl，统计每个大类下的子类及其出现次数。
    返回结构: 
    {
        "steak": {
            "steak_0": 150,
            "steak_1": 50,
            ...
        },
        ...
    }
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    class_distributions = defaultdict(lambda: defaultdict(int))
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            # 标签格式通常为 classname_clusterID，例如 apple_pie_0
            fine_label = entry["text"]
            
            # 解析大类名称 (处理诸如 ice_cream_0 这样带有多个下划线的情况)
            # 假设最后一个下划线后是聚类 ID
            parts = fine_label.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                main_class = parts[0]
            else:
                # Fallback，如果格式不符，则大类就是细粒度标签本身
                main_class = fine_label
                
            class_distributions[main_class][fine_label] += 1
            
    return class_distributions

def sample_labels(subclasses, num_samples, method):
    """
    根据给定的策略从子类字典中采样 `num_samples` 个标签。
    subclasses: 字典，格式为 { "steak_0": 150, "steak_1": 50, ... }
    method: "uniform", "inverse_density", 或 "proportional"
    返回: 采样的细粒度标签列表
    """
    labels = list(subclasses.keys())
    counts = np.array(list(subclasses.values()), dtype=np.float32)
    
    if method == "uniform":
        # 方法1: 平均概率随机
        probabilities = np.ones(len(labels)) / len(labels)
    elif method == "inverse_density":
        # 方法2: 概率密度的倒数
        # 数量越少，概率越高
        inv_counts = 1.0 / counts
        probabilities = inv_counts / inv_counts.sum()
    elif method == "proportional":
        # 方法3: 按聚类占比
        # 原数据集中的数量占比，即正比于原始分布
        probabilities = counts / counts.sum()
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    # 使用 np.random.choice 进行采样（有放回）
    sampled_indices = np.random.choice(len(labels), size=num_samples, p=probabilities, replace=True)
    return [labels[i] for i in sampled_indices]

def main():
    # 1. 加载配置
    config_path = "./config/generate.yaml"
    config = load_config(config_path)
    
    # 2. 解析数据集分布
    print(f"Parsing metadata from {config['data']['metadata_path']}...")
    class_distributions = parse_metadata(config['data']['metadata_path'])
    print(f"Found {len(class_distributions)} main food classes.")

    # 3. 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    seed = config['generate'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 4. 加载基础模型 Pipeline
    pretrained_model_path = config['model']['pretrained_model_name_or_path']
    print(f"Loading base model: {pretrained_model_path}")
    
    # 推荐使用 fp16 以节省显存和加快速度（如果是 GPU 的话）
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=dtype,
        safety_checker=None
    )
    
    # ==========================================
    # 5. 加载 LoRA 权重 (使用 PEFT 底层 API，最稳定)
    # ==========================================
    unet_lora_path = config['model'].get('unet_lora_path')
    text_encoder_lora_path = config['model'].get('text_encoder_lora_path')
    
    # 【自动容错】如果你在 yaml 里填的是文件路径，自动帮你截取成它所在的文件夹路径
    if unet_lora_path and unet_lora_path.endswith('.safetensors'):
        unet_lora_path = os.path.dirname(unet_lora_path)
    if text_encoder_lora_path and text_encoder_lora_path.endswith('.safetensors'):
        text_encoder_lora_path = os.path.dirname(text_encoder_lora_path)

    # 5.1 精确加载并融合 UNet Adapter
    if unet_lora_path and os.path.exists(unet_lora_path):
        print(f"Loading UNet PEFT adapter from {unet_lora_path}...")
        try:
            pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_lora_path)
            pipeline.unet = pipeline.unet.merge_and_unload()
            print("UNet PEFT adapter loaded and fused successfully.")
        except Exception as e:
            print(f"Failed to load UNet PEFT adapter: {e}")
    else:
        print(f"Warning: UNet LoRA path '{unet_lora_path}' not found or not specified.")

    # 5.2 精确加载并融合 Text Encoder Adapter
    if text_encoder_lora_path and os.path.exists(text_encoder_lora_path):
        print(f"Loading Text Encoder PEFT adapter from {text_encoder_lora_path}...")
        try:
            pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, text_encoder_lora_path)
            pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()
            print("Text Encoder PEFT adapter loaded and fused successfully.")
        except Exception as e:
            print(f"Failed to load Text Encoder PEFT adapter: {e}")
    else:
        print(f"Warning: Text Encoder LoRA path '{text_encoder_lora_path}' not found or not specified.")

    # 将 pipeline 移动到 GPU
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True) # 关闭内层进度条，使用外层的大进度条

    # 6. 准备生成任务
    num_images_per_class = config['generate']['num_images_per_class']
    method = config['generate']['label_selection_method']
    batch_size = config['generate'].get('batch_size', 4)
    output_base_dir = Path(config['output']['output_dir'])
    
    print(f"\nConfiguration summary:")
    print(f"- Images per class: {num_images_per_class}")
    print(f"- Sampling method: {method}")
    print(f"- Output directory: {output_base_dir}")
    
    # 为每个大类生成对应的 prompt 列表
    generation_tasks = {}
    for main_class, subclasses in class_distributions.items():
        sampled_labels = sample_labels(subclasses, num_images_per_class, method)
        # 构建 prompt，与训练时保持一致
        prompts = [f"a photo of {label}" for label in sampled_labels]
        generation_tasks[main_class] = {
            "prompts": prompts,
            "labels": sampled_labels
        }
        
        # 创建大类对应的输出文件夹
        class_out_dir = output_base_dir / main_class
        class_out_dir.mkdir(parents=True, exist_ok=True)

    # 7. 开始生成
    print("\nStarting image generation...")
    total_classes = len(generation_tasks)
    
    # 外层循环遍历所有大类
    for i, (main_class, task_info) in enumerate(tqdm(generation_tasks.items(), desc="Classes")):
        prompts = task_info["prompts"]
        labels = task_info["labels"]
        class_out_dir = output_base_dir / main_class
        
        # 记录每个细粒度标签已经生成了多少张，用于文件命名
        label_counters = defaultdict(int)
        
        # 分批处理 prompt
        for j in range(0, len(prompts), batch_size):
            batch_prompts = prompts[j : j + batch_size]
            batch_labels = labels[j : j + batch_size]
            
            # 调用 Pipeline 生成
            with torch.no_grad():
                results = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=config['generate']['num_inference_steps'],
                    guidance_scale=config['generate']['guidance_scale'],
                    generator=torch.Generator(device=device).manual_seed(seed + i * 1000 + j) # 确保每批有不同的随机性，但整体可复现
                )
                images = results.images
                
            # 保存图片
            for img, label in zip(images, batch_labels):
                count = label_counters[label]
                # 命名格式: apple_pie_0_001.jpg
                filename = f"{label}_{count:03d}.jpg"
                img.save(class_out_dir / filename)
                label_counters[label] += 1

    print(f"\nGeneration complete! All images saved to {output_base_dir}")

if __name__ == "__main__":
    main()