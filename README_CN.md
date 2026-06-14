**[English](README.md)**

## Med3D / MedPruner — 医学视觉语言模型评估与剪枝工具包

Med3D 是一个面向 3D 医学图像（CT/MRI）的视觉语言模型评估框架，内置 **MedPruner** 视觉 token 剪枝模块，在保持诊断精度的同时显著降低推理成本。

## 核心特性

- **3D 医学图像评估**: 支持 CT/MRI 等 3D 体积数据的端到端评估（Amos、M3D、3DRad）
- **视觉 Token 剪枝 (MedPruner)**: 支持切片级 (slice) 和 token 级两层剪枝策略，减少视觉 token 数量，降低推理显存和时间
- **多模型支持**: 统一接口接入 Qwen3-VL、HuluMed、MedGemma 三款医学视觉语言模型
- **多 GPU 并行**: 支持数据分片的多 GPU 并行评估

## 环境安装

```bash
# 创建 conda 环境
conda create -n med3d python=3.10
conda activate med3d

# 视频处理依赖
pip install decord ffmpeg-python imageio opencv-python
pip install qwen_vl_utils nltk rouge mathruler pylatexenc

# 3D 医学图像处理 (NIfTI 文件)
pip install nibabel

# 安装其他依赖
pip install -r requirements.txt

# 安装 medpruner 包
cd medpruner && pip install -e .
```

## 数据集

数据集下载: [Hulu Eval Dataset](https://modelscope.cn/models/Med-Team/Hulu-Med)

下载后将数据集放入 `data/Eval/` 目录，预期结构：
```
data/Eval/
├── Amos/
│   ├── amos_test_crop_32/
│   └── amos_val_mrg_sft.json
├── 3DRad/
│   └── ...
└── M3D/
    └── ...
```

## 支持的模型

| 模型 | 后端 | MedPruner 剪枝 |
|------|------|:---:|
| Qwen3-VL | transformers | ✓ |
| HuluMed Qwen2 | transformers | ✓ |
| MedGemma | transformers | ✓ |

## 支持的基准测试

| 数据集 | 模态 | 描述 |
|--------|------|------|
| Amos | 3D CT/MRI | 多器官分割与报告生成 |
| M3D | 3D CT | 全身 CT 影像问答 |
| 3DRad | 3D CT | 放射影像报告生成 |

## MedPruner 剪枝配置

MedPruner 通过两层剪枝减少视觉 token 数量：

- **切片级剪枝 (slice_compose)**: 基于锚点帧 L1 距离，丢弃低于阈值 `gamma` 的冗余帧
- **Token 级剪枝 (token_compose)**: 基于注意力权重，保留累计注意力权重达到 `tau` 的关键 token，其余按相似度聚合

配置文件 `config/config.json`:

```json
{
  "medpruner": {
    "slice_compose": true,
    "token_compose": true,
    "tau": 0.9,
    "gamma": 0.05
  }
}
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `slice_compose` | bool | 是否启⽤切片级帧间压缩 |
| `token_compose` | bool | 是否启⽤ token 级注意⼒选择 |
| `tau` | float | 累计注意力权重阈值 (0~1)，控制 token 保留⽐例。越⼩保留越少 |
| `gamma` | float | 锚点帧 L1 距离阈值，低于此值的帧被丢弃 |

剪枝率说明：评估完成后会输出 `compression_rate`（切片剪枝保留率）、`dyn_token_rate`（token 剪枝保留率）、`select_rate`（综合保留率）。

## 评估

```bash
cd MedUniEval

# 评估 Amos / 3DRad
bash ./eval.sh

# 评估 M3D（分片评估）
bash ./eval_M3D_chunked.sh
```

关键环境变量：

| 变量 | 描述 |
|------|------|
| `EVAL_DATASETS` | 数据集名称，如 `Amos`、`M3D`、`3DRad` |
| `MODEL_NAME` | 模型注册名，如 `Qwen3-VL`、`Hulumed_qwen2` |
| `MODEL_PATH` | 模型权重路径 |
| `COMPRESSION` | 剪枝⽅法，设为 `medpruner` 启⽤剪枝 |
| `CUDA_VISIBLE_DEVICES` | 可⽤ GPU，多个 GPU 时⾃动数据并⾏ |

## 许可证

本项目基于 [木兰宽松许可证 v2](http://license.coscl.org.cn/MulanPSL2) 开源。
