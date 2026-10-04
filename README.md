# PIDNet-FFC-DWT

基于 PIDNet 的城市场景语义分割项目，在负责上下文建模的 I 分支中引入快速傅里叶卷积（Fast Fourier Convolution，FFC），并使用 Haar 离散小波变换（Discrete Wavelet Transform，DWT）改进局部处理单元。项目使用 Cityscapes 数据集，完成 19 类像素级分类。

当前主模型保留 I 分支 Layer3 的普通卷积，在 Layer4、Layer5 中使用 FFC；FFC 的全局单元使用傅里叶变换，局部单元使用 Haar DWT、小波子带处理和逆变换。训练支持 CUDA 单卡／多卡和断点续训，评估支持 GPU 和 CPU。

## 目录

- [1. 项目结构](#1-项目结构)
- [2. 环境配置](#2-环境配置)
- [3. 数据集准备](#3-数据集准备)
- [4. 模型权重准备](#4-模型权重准备)
- [5. 训练](#5-训练)
- [6. 评估](#6-评估)
- [7. 图片推理与模型统计](#7-图片推理与模型统计)
- [8. 常见问题](#8-常见问题)
- [9. GitHub 发布规范](#9-github-发布规范)
- [10. 参考文献与致谢](#10-参考文献与致谢)

## 1. 项目结构

```text
PIDNet-FFC-DWT/
├── configs/
│   ├── default.py                          # 配置默认值与命令行覆盖
│   └── cityscapes/
│       ├── pidnet_small_cityscapes.yaml    # train 训练、val 验证
│       └── pidnet_small_cityscapes_trainval.yaml
├── data/
│   ├── cityscapes/                         # 本地存放数据集
│   └── list/cityscapes/                    # train / val / test / trainval 划分
├── datasets/                              # 数据读取、标签映射、数据增强
├── models/
│   ├── pidnet.py                          # 当前 FFC+DWT 主模型
│   ├── pidnet_s.py                        # 保留的普通 PIDNet 实现
│   ├── ffc_modules.py                     # 普通 FFC 模块
│   ├── ffc_modules_lfu_dwt.py              # 含 Haar DWT 局部单元的 FFC 模块
│   └── model_utils.py
├── pretrained_models/                     # 本地存放预训练权重
├── samples/                               # 自定义输入图片
├── tools/
│   ├── train.py                           # CUDA 训练
│   ├── eval.py                            # GPU 评估／测试集预测
│   ├── eval_cpu.py                        # CPU 评估／测试集预测
│   ├── precompute_edges.py                # 可选：离线生成边缘标签缓存
│   └── custom.py                          # 自定义图片可视化
├── utils/                                 # 损失、训练循环、日志与指标
├── output/                                # 本地训练输出及权重
├── log/                                   # TensorBoard 事件文件
├── summary.py                             # 参数量与可选复杂度统计
├── requirements.txt
└── README.md
```

**本文全部命令均从项目根目录执行。** 下载 GitHub 仓库后，先进入包含 `tools/`、`configs/` 和 `requirements.txt` 的目录。

### 当前模型入口

`tools/train.py` 和 `tools/eval.py` 固定调用 `models.pidnet`，因此它们运行的是当前 FFC+DWT 模型。`MODEL.NAME: pidnet_small` 用于选择 Small 规模。

`configs/default.py` 中的 `MODEL.VARIANT`、`MODEL.ENABLE_LFU` 当前没有接入模型构建逻辑，修改这两个配置不会切换普通 PIDNet、FFC 或 DWT。CPU 评估使用 `--model pidnet`；其帮助文本中的 `pidnet_ffc_34` 等模块不在当前目录中。

## 2. 环境配置

### 2.1 建议环境

以下安装示例采用课程报告记录的软件组合：

| 项目 | 配置 |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.3.0 |
| GPU 版本 | CUDA 12.1 对应的 PyTorch wheel |
| NumPy | 1.26.4 |
| 操作系统 | Windows 或 Linux；自定义图片脚本的路径处理见第 7 节 |

GPU 训练需要 NVIDIA GPU 和兼容驱动。训练脚本直接使用 CUDA；仅做精度评估时可以安装 CPU 版 PyTorch，使用 `tools/eval_cpu.py`。

创建独立环境：

```bash
conda create -n pidnet-ffc-dwt python=3.12 -y
conda activate pidnet-ffc-dwt
python -m pip install --upgrade pip
```

## 3. 数据集准备

### 3.1 下载 Cityscapes

在 [Cityscapes 官方网站](https://www.cityscapes-dataset.com/) 注册账号，登录 [下载页面](https://www.cityscapes-dataset.com/downloads/)，获取：

- `leftImg8bit_trainvaltest.zip`：训练、验证和测试图像。
- `gtFine_trainvaltest.zip`：精细标注包；本地训练和指标评估使用其中的 train、val 标签。

将解压后的 `leftImg8bit/`、`gtFine/` 放到 `data/cityscapes/` 中，保留城市子目录：

```text
data/
├── cityscapes/
│   ├── leftImg8bit/
│   │   ├── train/<city>/*_leftImg8bit.png
│   │   ├── val/<city>/*_leftImg8bit.png
│   │   └── test/<city>/*_leftImg8bit.png
│   └── gtFine/
│       ├── train/<city>/*_gtFine_labelIds.png
│       └── val/<city>/*_gtFine_labelIds.png
└── list/
    └── cityscapes/
        ├── train.lst
        ├── val.lst
        ├── test.lst
        └── trainval.lst
```

当前整理目录只附有少量图像与标注示例，运行完整训练／评估前仍需准备完整数据集。GitHub 发布版本应只保留数据划分清单，由使用者从官方渠道下载数据。

### 3.2 数据路径与标签格式

默认配置为：

```yaml
DATASET:
  ROOT: data/
  TRAIN_SET: list/cityscapes/train.lst
  TEST_SET: list/cityscapes/val.lst
  NUM_CLASSES: 19
```

`DATASET.ROOT` 是同时包含 `cityscapes/` 和 `list/` 的父目录。代码读取清单时使用 `ROOT + list_path`，读取图像和标签时使用 `ROOT + cityscapes + 清单中的相对路径`，因此不要将默认根目录直接设为 `data/cityscapes/`。

`train.lst`、`val.lst` 和 `trainval.lst` 每行包含图像与标签两个路径，例如：

```text
leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
```

`test.lst` 每行只有图像路径。当前清单的样本数为：

| 划分 | 清单 | 样本数 | 用途 |
| --- | --- | ---: | --- |
| train | `train.lst` | 2,975 | 训练 |
| val | `val.lst` | 500 | 本地验证、模型选择与指标报告 |
| test | `test.lst` | 1,525 | 生成预测并提交官方评估 |
| train + val | `trainval.lst` | 3,475 | 合并训练，面向官方测试集 |

**使用原始 `*_gtFine_labelIds.png`。** `datasets/cityscapes.py` 会将原始 label ID 映射到 0–18，并将其余类别映射为忽略标签 255。不要直接换成已经映射的 `labelTrainIds`，否则会发生重复映射。

训练集使用随机缩放、裁剪和水平翻转；图像按 RGB、ImageNet 均值和标准差归一化。验证集关闭随机增强，使用原始图像分辨率。

### 3.3 可选：预计算边缘标签

代码会自动查找标签同目录下的 `*_gtFine_labelIds_edge.png`，未找到时在线计算边缘，因此这一步不是必需条件。

```bash
python tools/precompute_edges.py --dataset cityscapes --data-root data/ --edge-size 4 --workers 4
```

该脚本读取 train、val、trainval 清单，去重后生成边缘缓存，保留原图像和原标签。缓存文件写入数据集标签目录；正式对比实验中，应保持边缘处理设置一致。

## 4. 模型权重准备

### 4.1 ImageNet 初始化权重

默认训练配置使用：

```text
pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar
```

可从 [PIDNet 官方仓库的模型下载入口](https://github.com/XuJiacong/PIDNet#models) 获取权重并放到上述路径。优先查看官方 README 当前提供的下载入口。

ImageNet 权重用于初始化与当前模型键名和形状匹配的层，新引入的 FFC／DWT 层使用自身初始化。加载日志中的参数项数量可用于确认初始化实际生效。

从零训练时，在训练命令末尾增加：

```text
MODEL.PRETRAINED '""'
```

这里的 `'""'` 向 YACS 传入可解析为空字符串的值，适用于本文的 Bash 与 PowerShell 示例。

### 4.2 项目分割权重

本地整理目录中的 `output/best_7820.pt` 是项目分割权重。第 6、7 节的命令以它为例；自行训练后，可将路径替换为对应输出目录下的 `best.pt`。

GitHub 若未包含该文件，需要先取得项目权重或完成训练。发布权重时应提供实际下载链接、模型结构、数据划分、评估设置和文件 SHA-256。

`pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt`、`PIDNet_S_Cityscapes_test.pt` 属于普通 PIDNet 权重。它们与当前 FFC+DWT 结构不同，不应当作当前模型的完整分割权重使用。

## 5. 训练

### 5.1 单 GPU 训练

**Windows PowerShell：**

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml --seed 304 GPUS "(0,)" WORKERS 4 TRAIN.BATCH_SIZE_PER_GPU 12 TEST.BATCH_SIZE_PER_GPU 1
```

**Linux Bash：**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml --seed 304 GPUS "(0,)" WORKERS 4 TRAIN.BATCH_SIZE_PER_GPU 12 TEST.BATCH_SIZE_PER_GPU 1
```

### 5.2 多 GPU 训练

脚本使用 `torch.nn.DataParallel`。下面以两张 GPU、每卡批量 6 为例，总训练批量为 12。

**Windows PowerShell：**

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1"
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml --seed 304 GPUS "(0,1)" WORKERS 4 TRAIN.BATCH_SIZE_PER_GPU 6 TEST.BATCH_SIZE_PER_GPU 1
```

Linux 可将环境变量写在同一行命令前：

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml --seed 304 GPUS "(0,1)" WORKERS 4 TRAIN.BATCH_SIZE_PER_GPU 6 TEST.BATCH_SIZE_PER_GPU 1
```

## 6. 评估

### 6.1 GPU 验证

准备完整的 500 张 val 图像及标签，使用与当前模型结构匹配的分割权重：

```bash
python tools/eval.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml MODEL.PRETRAINED '""' TEST.MODEL_FILE output/best_7820.pt WORKERS 4
```

`MODEL.PRETRAINED '""'` 关闭建模阶段的额外初始化加载；`TEST.MODEL_FILE` 指定本次真正评估的分割权重，因此评估不需要另外准备 ImageNet 权重。

### 6.2 CPU 验证

```bash
python tools/eval_cpu.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml --model pidnet MODEL.PRETRAINED '""' TEST.MODEL_FILE output/best_7820.pt
```

CPU 脚本使用 batch size 1、0 个数据加载进程，不需要 CUDA。关闭 `MODEL.PRETRAINED` 也能避免它在初始化阶段尝试以分割权重格式读取 ImageNet 文件。

两个评估脚本均可将 `TEST.MODEL_FILE` 替换为：

```text
output/cityscapes/pidnet_small_cityscapes/best.pt
```

当前 GPU 评估脚本会统一移除权重键名前 6 个字符，适配训练器保存的 `model.` 前缀。CPU 脚本同时兼容有、无 `model.` 前缀的权重，并检查形状。两者都可能保留未匹配层的初始化值，因此评估前应核对加载日志，确认使用了完整、匹配的分割权重。

## 7. 图片推理与模型统计

### 7.1 自定义图片可视化

将自己的 PNG 图片放入 `samples/`。当前 `tools/custom.py` 按 Windows 路径提取文件名，以下示例用于 Windows：

```powershell
python tools/custom.py --a pidnet-s --p output/best_7820.pt --r samples/ --t .png
```

- `--a pidnet-s`：选择当前主模型的 Small 规模。
- `--p`：匹配模型结构的分割权重；该脚本同样要求 `model.` 前缀。
- `--r`：输入目录，末尾保留路径分隔符。
- `--t`：文件扩展名；处理 JPEG 时可改为 `.jpg`。

该脚本默认在 CPU 上推理，生成 Cityscapes 19 类配色的分割图，保存到 `samples/outputs_ffc45/`。Linux 上使用前需要处理脚本现有的 Windows 路径假设；数据集评估可直接使用第 6 节入口。

`samples/README.md` 中提到的 `run.py infer` 属于旧说明，当前目录没有 `run.py`，请以本节命令为准。

### 7.2 参数量与复杂度

```bash
python summary.py
```

脚本统计当前 Small 模型的总参数量和可训练参数量。可选安装 `torchinfo` 和 `thop` 后查看模型结构和算子统计：

```bash
python -m pip install torchinfo thop
```

FFC 使用双流输入，优先使用 `torchinfo`。复杂度工具对 FFT、逆 FFT 和自定义小波运算的统计可能不完整；发布 MACs／FLOPs 时应注明输入尺寸、工具和统计口径，不能直接据此推断 FPS。

## 8. 参考文献

1. Xu, Jiacong; Xiong, Zixiang; Bhattacharyya, Shankar P. **PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers.** CVPR 2023. [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html)
2. Chi, Lu; Jiang, Borui; Mu, Yadong. **Fast Fourier Convolution.** NeurIPS 2020. [论文](https://papers.nips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html)
3. Cordts, Marius et al. **The Cityscapes Dataset for Semantic Urban Scene Understanding.** CVPR 2016. [项目主页](https://www.cityscapes-dataset.com/)