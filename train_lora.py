import os
import json
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

# Hugging Face 库
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

# 数据集类
class FoodClusteringDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512, center_crop=True):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        
        # 解析 metadata.jsonl
        metadata_path = self.data_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found in {self.data_dir}")
            
        self.entries = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))
                    
        # 图像预处理
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1] for SD
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = self.data_dir / entry["file_name"]
        text = entry["text"]
        
        # 提示词
        prompt = f"a photo of {text}"

        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_transforms(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 【修复点 3】：避免使用纯黑图片投毒，递归返回下一张有效图片
            return self.__getitem__((idx + 1) % len(self.entries))

        # 处理文本
        inputs = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0) # 移除 batch 维度
        }

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def log_validation(pipeline, config, accelerator, step, unet, text_encoder):
    """用于在训练过程中进行采样并记录到 wandb，同时保存权重"""
    prompt = config['logging']['sampling']['prompt']
    num_inference_steps = config['logging']['sampling']['num_inference_steps']
    num_images = config['logging']['sampling']['num_images']
    
    print(f"\nRunning validation and saving weights at step {step}...")
    
    # 1. 保存当前权重
    import wandb
    output_dir = Path(config['output']['output_dir'])
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    
    unwrapped_unet.save_pretrained(checkpoint_dir / "unet_lora")
    unwrapped_text_encoder.save_pretrained(checkpoint_dir / "text_encoder_lora")
    
    # 将保存的权重作为 artifact 上传到 wandb
    if accelerator.is_main_process and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"lora-weights-step-{step}",
            type="model",
            description=f"LoRA weights at training step {step}"
        )
        artifact.add_dir(str(checkpoint_dir))
        wandb.log_artifact(artifact)
        print(f"Weights saved to {checkpoint_dir} and uploaded to wandb.")

    # 2. 生成验证图片
    # 将模型切换到 eval 模式
    pipeline.unet.eval()
    pipeline.text_encoder.eval()
    
    # 避免在推理时记录梯度
    with torch.no_grad():
        # 【修复点 1】：开启自动混合精度上下文，解决 Pipeline 精度冲突崩溃报错
        with accelerator.autocast():
            images = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images,
                generator=torch.Generator(device=accelerator.device).manual_seed(config['train']['seed']),
            ).images

    # 将图片记录到 wandb
    import wandb
    wandb_images = [wandb.Image(img, caption=f"{prompt} (step: {step})") for img in images]
    accelerator.log({"validation/images": wandb_images}, step=step)
    
    # 切换回 train 模式
    pipeline.unet.train()
    pipeline.text_encoder.train()
    
    # 清理缓存
    torch.cuda.empty_cache()

def main():
    # 1. 加载配置
    config_path = "./config/fine_tune.yaml"
    config = load_config(config_path)

    # 2. 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['train']['gradient_accumulation_steps'],
        mixed_precision=config['train']['mixed_precision'],
        log_with="wandb",
    )
    
    # 初始化 wandb
    wandb_config = config['logging']['wandb']
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_config['project'],
            config=config,
            init_kwargs={
                "wandb": {
                    "entity": wandb_config['entity'],
                    "name": wandb_config['name'],
                }
            }
        )

    set_seed(config['train']['seed'])

    # 3. 加载模型与 Tokenizer
    pretrained_model_name = config['model']['pretrained_model_name_or_path']
    
    # 【修复点 2】：提取获取权重的 dtype 数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    print(f"Loading pretrained model: {pretrained_model_name} in {weight_dtype}")
    
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")
    # 【修复点 2】：直接指定 torch_dtype，以半精度加载基础模型节省大量显存
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet", torch_dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

    # 冻结 VAE (我们不需要微调它)，并将其移动到对应设备
    vae.requires_grad_(False)
    vae.to(accelerator.device)
    
    # 4. 使用 PEFT 注入 LoRA (同时注入 UNet 和 Text Encoder)
    lora_config_unet = LoraConfig(
        r=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], # 针对 attention 层的常用配置
    )
    
    lora_config_text = LoraConfig(
        r=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], # 针对 text encoder 的 attention
    )

    print("Injecting LoRA into UNet and Text Encoder...")
    unet = get_peft_model(unet, lora_config_unet)
    text_encoder = get_peft_model(text_encoder, lora_config_text)
    
    # 确保只有 LoRA 层可以训练
    unet.print_trainable_parameters()
    text_encoder.print_trainable_parameters()

    # 5. 准备优化器
    # 收集两者的可训练参数
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters())) + \
                  list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
                  
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=float(config['train']['learning_rate']),
    )

    # 6. 数据加载
    dataset = FoodClusteringDataset(
        data_dir=config['data']['dataset_dir'],
        tokenizer=tokenizer,
        size=config['data']['resolution'],
        center_crop=config['data']['center_crop'],
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=8,
    )

    # 7. Accelerate 准备
    unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader
    )

    # 8. 计算真实的训练步数并配置学习率调度器
    import math
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['train']['gradient_accumulation_steps'])
    max_train_steps = config['train']['num_train_epochs'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_train_steps,
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    # 构建用于验证采样的 Pipeline
    if accelerator.is_main_process:
        # Pipeline 组装需要未被 prepare 包裹的基础模型
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            tokenizer=tokenizer,
            safety_checker=None,
        )
        pipeline.scheduler = noise_scheduler
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

    # 9. 训练循环
    num_train_epochs = config['train']['num_train_epochs']
    print(f"Starting training for {num_train_epochs} epochs...")
    global_step = 0
    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 外层循环 epoch
    unet.train()
    text_encoder.train()
    
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder):
                # 1. 编码图像到隐空间
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. 为隐变量添加噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. 提取文本特征
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # 4. U-Net 预测噪声
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 5. 计算 Loss (MSE)
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # 6. 反向传播与优化
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # 裁剪梯度，防止爆炸
                    accelerator.clip_grad_norm_(lora_layers, 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 同步更新全局步数
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 记录日志
                if global_step % config['train']['logging_steps'] == 0:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch
                    }, step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch}
            progress_bar.set_postfix(**logs)

        # 每个 epoch 结束后的验证采样
        if accelerator.is_main_process:
            if (epoch + 1) % config['logging']['sampling']['sample_every_n_epochs'] == 0:
                log_validation(pipeline, config, accelerator, global_step, unet, text_encoder)

    # 10. 训练结束，保存最终权重
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving LoRA weights to {output_dir}...")
        
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        
        # 使用 peft 自带的 save_pretrained
        unwrapped_unet.save_pretrained(output_dir / "unet_lora")
        unwrapped_text_encoder.save_pretrained(output_dir / "text_encoder_lora")
        
        print("Training completed successfully!")

    accelerator.end_training()

if __name__ == "__main__":
    main()