# 🧠 LLaVA 论文复现与微调步骤

复现的论文：[https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)

## ⚙️ 硬件条件

| 硬件 | 配置 |
|-------|------|
| GPU | RTX 4090 (24GB) × 1 |
| CPU | 16 vCPU Intel(R) Xeon(R) Gold 6430 |
| 内存 | 120 GB |
| 硬盘 | 系统盘 30 GB / 数据盘 免费 50 GB + 付费 25 GB |
| 来源 | AutoDL 平台 |

---

## 🧩 1. 下载源码

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

---

## 🧱 2. 创建虚拟环境

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
```

> 💡 建议使用清华源或 AutoDL 学术加速。

---

## ⚗️ 3. 安装训练依赖

```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

若报错，可改用以下命令解决 FlashAttention 与 4090 的兼容问题：

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

---

## 🔄 4. 更新最新代码

```bash
git pull
pip install -e .
```

若失败可跳过。

---

## 📦 5. 下载模型权重文件

分为两个模块：

- **LLM 模型**：[liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- **视觉编码器**：[openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)

> ⚠️ 建议先在本地下载后上传至服务器（AutoDL 提供公网传输更方便）。  
> 下载完成后请记住这两个文件的存放路径。

---

## 🧩 6. 修改配置文件

在下载的 LLM 文件夹中找到 `config.json`，修改以下字段：

```json
"mm_vision_tower": "/root/LLaVA/clip-vit-large-patch14-336"
```

这一步用于让语言模型（LLM）知道对应的视觉编码器位置。  
后续的 LoRA 微调将主要调整从 LLM 到 Vision Tower 的映射层参数。

---

## 🖥️ 7. 启动可视化界面

在 LLaVA 文件夹中打开三个终端。

### Terminal 1（控制器）

```bash
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

### Terminal 2（Web 服务）

```bash
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

### Terminal 3（模型加载）

```bash
python -m llava.serve.model_worker     --host 0.0.0.0     --controller http://localhost:10000     --port 40000     --worker http://localhost:40000     --model-path /root/autodl-tmp/llava-v1.5-7b     --load-4bit
```

然后打开本地 PowerShell 或 VSCode SSH，进入可视化界面。

> 作者测试模型为 **13B**，输入示例为“一只可爱的哈基米”。

---

## 🧾 8. 数据集准备

使用 **TextVQA** 的部分数据。

下载地址：  
👉 [https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)

### 解压

将数据解压至：

```
/root/autodl-tmp/train_images/textvqa/train_images/
```

### 过滤 JSON 标注

下载数据标注：  
👉 [https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

提取出只包含 TextVQA 的部分，并随机抽取 1000 条：

```python
import json, random

input_path = "llava_v1_5_mix665k.json"
output_path = "textvqa_1k.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

textvqa_data = [item for item in data if "textvqa" in item.get("image", "").lower()]
subset = random.sample(textvqa_data, min(1000, len(textvqa_data)))

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)

print(f"✅ 已保存到 {output_path}")
```

---

## 🎯 9. LoRA 微调

修改 `scripts/v1_5/finetune_task_lora.sh` 中参数：

```bash
model_name_or_path="/root/autodl-tmp/llava-v1.5-7b"
data_path="/root/autodl-tmp/textvqa_1k.json"
image_folder="/root/autodl-tmp/train_images/textvqa/train_images"
vision_tower="/root/LLaVA/clip-vit-large-patch14-336"
```

然后执行：

```bash
source ./scripts/v1_5/finetune_task_lora.sh
```

训练约 10 分钟完成。

---

## 🧩 10. 合并微调权重

```bash
python scripts/merge_lora_weights.py     --model-path "./checkpoints/llava-v1.5-7b-task-lora"     --model-base "/root/autodl-tmp/llava-v1.5-7b"     --save-model-path "/root/autodl-tmp/llava-v1.5-7b-task-lora-merged"
```

---

## 👀 11. 微调模型可视化

```bash
python -m llava.serve.model_worker     --host 0.0.0.0     --controller http://localhost:10000     --port 40000     --worker http://localhost:40000     --model-path /root/autodl-tmp/llava-v1.5-7b-task-lora-merged     --load-4bit
```

作者选择了 **7B 模型** 进行微调（13B 会显存爆炸），结果输出如图所示。

---

> 💬 本次微调到此为止，作者受限于资金，仅能完成该规模的实验。
