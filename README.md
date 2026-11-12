# 从零实现 BPE 分词器与 Transformer / LSTM 语言模型

基于 TinyStories 数据集的 NLP 课程作业。项目不使用 `torch.nn` 中的现成网络层，从零实现了 byte-level BPE 分词器、Transformer、LSTM，以及完整的训练和文本生成流程。

## 项目内容

### 1. Byte-level BPE 分词器
- 在 TinyStories 上训练 10,000 词表的 BPE
- 处理 `<|endoftext|>` 特殊 token，合并操作不跨文档边界
- 支持 `encode`、`decode`、`encode_iterable`，带预分词缓存
- 在 541M token 的训练集上完成编码并保存为 `uint16` 数组

### 2. Transformer 语言模型
- Linear、Embedding、RMSNorm
- SwiGLU 前馈网络
- RoPE 旋转位置编码
- 缩放点积注意力、因果多头自注意力
- Pre-Norm Transformer Block 与完整 LM

### 3. LSTM 语言模型
- LSTMCell：输入门、遗忘门、输出门、候选状态
- 多层 LSTM 与 LSTM LM

### 4. 训练与解码
- 交叉熵损失、AdamW、带 warmup 的余弦学习率、梯度裁剪
- 基于 `np.memmap` 的大规模数据加载
- checkpoint 保存与恢复
- temperature 缩放与 top-p (nucleus) 采样，支持 EOS 提前停止

## 实验结果

| 模型 | 参数量 | 训练步数 | 验证集 loss | 验证集 perplexity |
|---|---|---|---|---|
| Transformer | 28.9M | 40,000 | 1.304 | **3.68** |
| LSTM | 14.4M | 15,000 | 1.607 | **4.99** |

BPE 分词器：

- 词表大小：10,000
- 压缩比：约 4.11 bytes/token
- 训练耗时：约 15.7 分钟，峰值内存约 9.77 GB

## 生成样例

Prompt：

```text
Once upon a time, there was a little girl named Lily.
```

Transformer，`temperature=0.8, top_p=0.9`：

```text
Once upon a time, there was a little girl named Lily. She loved to ride her red bike. One day, she went to the park with her mom.
At the park, Lily saw a small, shivering dog. She asked her mom, "Mom, can we help the dog?" Her mom said, "Yes, let's give him a nice bath." They took the dog home and gave him a bath. The dog was happy and wagged his tail.
After the bath, the dog was not cold anymore. He was happy and warm. Lily and her mom went back to the park to ride their bikes. They had a fun day together
```

## 目录结构

```text
basics/
  BPE.py                    BPE 训练
  tokenizer.py              BPE 编码 / 解码
  model.py                  Transformer、LSTM、RoPE、注意力等
  optimizer.py              AdamW、交叉熵、学习率调度、梯度裁剪
  training.py               数据加载与 checkpoint
train_bpe.py                训练分词器
tokenizer_experiments.py    压缩率、吞吐量统计与数据集编码
train_lm.py                 语言模型训练脚本
tests/adapters.py           作业测试适配器
output/                     训练好的词表与合并规则
```

## 运行方式

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest

# 训练 BPE 并编码数据集
uv run python train_bpe.py
uv run python tokenizer_experiments.py --encode

# 训练 Transformer
uv run python train_lm.py \
  --train-data data/train_tokens.npy \
  --val-data data/valid_tokens.npy \
  --model transformer --device cuda \
  --vocab-size 10000 --context-length 256 \
  --d-model 512 --num-layers 6 --num-heads 8 --d-ff 1344 \
  --batch-size 64 --max-iters 40000 \
  --lr 3e-4 --min-lr 3e-5 --warmup-iters 2000 \
  --checkpoint-dir checkpoints/transformer
```

LSTM 训练把 `--model` 改成 `lstm`，并去掉 `--num-heads`、`--d-ff` 即可。

## 作业来源

本项目的作业要求、测试用例与初始代码框架来自：

- [WangXiangyv/NLPDL-2026Fall](https://github.com/WangXiangyv/NLPDL-2026Fall)

感谢课程提供的作业设计与测试用例。原始作业说明见 [ASSIGNMENT.md](ASSIGNMENT.md)。

## 说明

- `data/` 和 `checkpoints/` 未上传，需按原始作业说明自行下载 TinyStories 并训练
- 原始作业说明见 [ASSIGNMENT.md](ASSIGNMENT.md)
- 本项目为课程作业实现，仅供学习参考，请勿直接复制提交
