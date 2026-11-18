# LCF / LCF-X 独立复现仓库

> 这是 2026-11-18 的独立可迁移快照，不继承或依赖任何上游 Git
> remote。源码、正式配置、统一评测入口、提示词归档和结果文档均已纳入；模型、
> 数据、检查点和逐样本结果按 [`MIGRATION.md`](MIGRATION.md) 中的外部资产清单迁移。

六份完成训练的 final adapter 已通过 Git LFS 纳入 `weights/`。克隆后执行
`git lfs pull` 即可直接加载；文件校验和和参数量见 `weights/manifest.json`。

本目录对应 `/home/lewis/子问题四_统一复现与变量实验计划.md` 的 E4/E5，训练六份
`d=128` 通信模块权重。基础模型冻结，权重保存在独立目录，不改写共享资产。

| 配置 | Sender | Receiver | 任务 |
| --- | --- | --- | --- |
| M1 | Qwen3-0.6B | Qwen3-1.7B | LCF、LCF-X |
| M2 | Qwen3-0.6B | Qwen3-4B | LCF、LCF-X |
| M3 | Qwen3-4B | Qwen3-0.6B | LCF、LCF-X |

## 论文方法和训练配置

来源：[LCF v2](https://arxiv.org/html/2605.22863v2)，
[XKV 的异构 LCF-X 扩展](https://arxiv.org/html/2608.20617v1)。
LCF 原文的模型对、LCF-X 原文的同构模型对与本项目三组模型不同；这里按实验计划替换
基础模型，并采用 XKV 描述的可学习层轴映射。未找到作者公开的 LCF 实现，因此这是
依据论文和其引用的 C2C 代码重建的实现。详细实现选择见
`manifests/method_sources.json`；不宣称与未公开作者代码逐项一致。

| 项目 | LCF | LCF-X |
| --- | --- | --- |
| 数据 | OpenHermes 2.5 原始前 500K 候选，C2C 的完整对话长度过滤 | HotpotQA-distractor 训练集，仅 bridge |
| 输入 | 两端相同 prompt，token 对齐 | 共享问题，各一支持段落和四干扰段落 |
| 验证留出 | 1%，seed 42 | 2.8%，seed 0 |
| 训练长度 | 300 optimizer steps | 1 epoch，步数由最终训练条数计算 |
| 有效 batch | 260 | 256 |
| 物理 microbatch | 1，按原文 10 条逻辑 microbatch 加权累积 | 1，按原文 16 条逻辑 microbatch 加权累积 |
| AdamW | lr 1e-4，weight decay 0.01，clip 1.0 | 同左 |
| 线性学习率 warmup | 30 步 | 200 步 |
| gate temperature | 1 → 0.001 / 400 步 | 1 → 0.5 / 一个 epoch |
| 结构 | d=128，两个 4d FFN，dropout 0.1 | 同左，加两级 attention pool |
| 最大序列 | 2048 | 2048 |
| eval/save | 每 50 步 | 每 50 步 |
| 最终权重选择 | 固定 final，保留中间 checkpoint | 按原文采用 final |

显存优化保持冻结模型为 bf16，不量化；通信模块参数为 fp32，在 CUDA 上用 bf16
autocast。答案 vocabulary projection 分块重计算以减少内存。
两端先在 no-grad 下 prefill，更新 prefix cache 一次，再从 receiver 的最后一个
prompt token 开始 teacher forcing。Sender 的输入不包含监督答案。

LCF 的接收层采用 C2C `last_aligned` 源层映射。
LCF-X 的层数不同时，对源侧池化摘要使用可学习线性层轴映射，输出接收层数；其余投影
分别接受 source/receiver KV 宽度，输出 receiver 原生 KV。三个 Qwen3 模型均为
8 个 KV heads、head dim 128，但 4B 有 36 层，另两个模型各有 28 层。

## 模型与数据复用

`models/` 的三个目录通过符号链接复用 `/mnt/hd2/lewis/SwaTReproduction/models/qwen`
中的本地完整模型，不再访问 NFS；提交版本和原始 SHA-256 见
`manifests/reused_assets.json`。`scripts/verify_assets.py` 重新校验权重及 tokenizer，
输出 `manifests/verified_models.json`。文件已经完整时无需重新下载约 13.7 GB 权重。

OpenHermes 从 `/mnt/hd2/lewis/Subproblem4UnifiedReproduction/cache` 的本地 Arrow 原始缓存
读取；HotpotQA 从同一本地资产根目录中的固定版本 Parquet 读取。
OpenHermes 按固定提交的 C2C 代码去除 system 消息、裁去 user 文本两端空白，保留
assistant 原始文本，再进行完整对话长度过滤。
对模板渲染后 prompt token 前缀无法严格匹配完整对话的记录，单独统计并排除，防止
错误移动答案的监督位置。
所有派生数据、map cache、日志与权重写入本目录。准备过程生成独立 train/validation
划分、样本 ID 哈希、过滤计数。超长记录明确排除，不截掉支持证据或全部答案。
重复问题/对话不会跨 train/validation。预留 GSM8K test、HotpotQA validation、HumanEval
test 仅参与精确规范化文本重合检查；这个检查不能排除语义改写或子串层面的污染。

准备输出：`data/lcf`、`data/lcfx`；实际样本数和总步数见
`manifests/lcf_data.json`、`manifests/lcfx_data.json`。如果长度/重合检查改变了样本数，
LCF-X 的单 epoch 步数也随之变化，不硬凑论文约 277 步。

## 执行与查看

环境：Python 3.12，PyTorch 2.6.0+cu124，Transformers 4.51.3，Datasets 3.5.0。
安装版本快照保存在 `manifests/environment.txt`。
重建环境可执行 `bash scripts/install.sh`，优先复用 `wheels/` 中经 SHA-256 校验的安装包。

```bash
cd /home/lewis/LCFReproduction
export PYTHONPATH="$PWD/src"
.venv/bin/python scripts/status.py
tail -n 30 logs/bootstrap.log
cat runs/queue_status.json
```

`scripts/bootstrap.py` 校验模型、运行实现测试后进入 GPU 队列。队列分别检查两种方法
的数据，已就绪的方法先训练；尚未就绪的数据在一个独立 CPU 预处理进程中生成。
队列每次启动任务前扫描所有物理 GPU，选择利用率不超过 10%、可用显存充足的卡；
优先选择利用率更低、可用显存更多的卡。按用户 2026-10-21 的要求，所有可用 GPU
可以并行启动不同实验，每张卡同时只运行一组；被其他服务占用的卡等待释放后补入。
每个训练进程启动后保持在所选 GPU 上，后续任务重新选择空闲卡。M1 要求空闲
14,000 MiB，M2/M3 要求空闲 20,000 MiB；未满足时每 10 秒检查并记录等待状态。
每组实验仍是单 GPU 训练，各组之间并行，不使用分布式训练。按用户 2026-10-22 的要求，
不设置进程显存比例上限，按 GPU 实际可用显存分配。CPU 计算线程为 4，数据预处理进程为 2。
如果某一组失败，记录错误并继续其他组，不自动修改该组训练配方。
LCF M2 曾触及进程额度后退出，已从完整优化步断点恢复；当前运行也取消了该限制。
满足条件后，每组先做两个优化步的真实数据 smoke，再进入正式训练。
smoke 的有效 batch=1，结果与正式训练分开存放。

```bash
# 已有 bootstrap/queue 运行时不要重复启动，进程锁也会拒绝重复实例。
.venv/bin/python scripts/bootstrap.py
# 直接启动队列；缺失数据会自动准备，已就绪的方法先运行：
bash scripts/start_queue.sh --gpu auto
# 如需临时限制并发数：
bash scripts/start_queue.sh --gpu auto --max-workers 2
```

每组输出：

- `runs/formal/<配置>/status.json`：训练状态、步数、进程号。
- `runs/formal/<配置>/metrics.jsonl`：loss、梯度范数、耗时和显存；每 50 步完整验证。
- `runs/formal/<配置>/latest/`：每个优化步保存的恢复状态。
- `runs/formal/<配置>/checkpoint-*/`：每 50 步留存的权重与优化器状态。
- `runs/formal/<配置>/final/adapter.safetensors`：完成训练后的通信模块权重。
- `logs/<配置>.train.log`：控制台日志。

重启同一命令会从 `latest` 的完整优化步恢复 optimizer、scheduler、样本游标及随机数
状态；尚未完成的梯度累积会从该优化步开头重跑。最终权重已存在时队列跳过该配置。
单组失败会阻止其进入下一阶段，其余独立组继续运行；所有错误写入汇总状态。

本次只训练 E4/E5。统一测试、通信量/耗时测量与 d=64/256/512 重新训练属于计划后续阶段。
