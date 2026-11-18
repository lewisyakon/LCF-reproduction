# 独立仓库与迁移说明

本仓库是 LCF/LCF-X 当前复现状态的独立源码快照。它不包含父仓库、Git
submodule 或上游 remote；`vendor/C2C` 也是普通源码目录。克隆该仓库不需要访问原
LCF、C2C、Dense 或统一实验仓库。

## 仓库中包含的内容

- LCF/LCF-X 的训练实现、六组正式配置、测试和数据准备脚本；
- 统一协议评测脚本及其所需的评分依赖；
- 固定提示词 ZIP，SHA-256 为
  `830ea90cac0c4d415b0d883361cddce42cd8dfa3e11552a39bb9c9aac5d1ec17`；
- 正式结果、统一实验计划和跨方法对比文档；
- 当前环境、数据和方法来源 manifest。

模型、Arrow/Parquet 数据、训练检查点和逐样本结果不进入 Git。它们的当前位置、
建议恢复位置、用途和体积记录在 `migration/external_assets.json`。

## 迁移前检查

```bash
python tools/verify_external_assets.py --mode current --profile train
python tools/verify_external_assets.py --mode current --profile results
```

代码可制作成单文件离线 Git bundle：

```bash
bash tools/create_code_bundle.sh /path/to/export
```

资产应按 manifest 的 `current_path` 复制到新仓库下的 `recommended_path`。建议用
`rsync -aH --info=progress2`，不要把大文件提交到 Git。恢复后检查：

```bash
python tools/verify_external_assets.py --mode restored --profile train
python tools/verify_external_assets.py --mode restored --profile evaluate
```

## 新机器运行

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
export PYTHONPATH="$PWD/src"
export LCF_SOURCE_ASSETS_ROOT="$PWD"
.venv/bin/python -m pytest -q
```

`scripts/prepare_data.py` 默认从仓库内的 `cache/` 和 `datasets/` 读取原始资产；
`scripts/evaluate_unified.py` 默认使用仓库内固定提示词和已 vendored 的公共评分代码。
历史脚本中仍作为结果元数据保留的旧绝对路径，可先预览再替换：

```bash
python tools/relocate_snapshot.py \
  --map /home/lewis/LCFReproduction="$PWD" \
  --map /mnt/hd2/lewis/LCFReproduction="$PWD"
```

确认列表后追加 `--write`。环境路径与 GPU UUID 应通过 `LCF_PYTHON`、
`LCF_EXPECTED_GPU_UUID` 设置，不需要修改代码。

