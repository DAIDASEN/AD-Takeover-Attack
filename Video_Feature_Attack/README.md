# Video Feature Attack（视频特征攻击）

本目录包含多类针对 `LLaVA-NeXT-Video` 的视频输入对抗攻击脚本，并把每个视频的结果落盘到 `output-dir/<video_id>/`，方便断点续跑/复用结果。

## 1) Sponge Attack：`1_attack_sponge.py`

- 目标：把模型输出“拉长/灌水”为一段固定的 `SPONGE_TARGET`（类似 sponge/垃圾 token 注入效果），从而影响下游判断。
- 输入：`--data-root` 目录下的 `.mp4` 文件（默认 `BDDX/Sample`）。
- 输出：
  - `output-dir/<video_id>/adv_<原文件名>.mp4`：对抗后视频
  - `output-dir/<video_id>/log.json`：攻击前后回答与长度对比
  - `output-dir/final_summary.json`：把本次运行过程中“遇到的”（新攻击 + 复用的）视频日志汇总成一个列表

运行示例：

```bash
python 1_attack_sponge.py \
  --data-root BDDX/Sample \
  --output-dir results/Llama/results_sponge_sample \
  --limit 20 \
  --steps 150 \
  --num-frames 16
```

## 2) Auto-Flip / Misinfo Attack：`2_attack_misinfo.py`

- 目标：先用 `LLaVA-NeXT-Video` 对“是否需要人工接管(Yes/No)”做原始回答，再自动把目标翻转（Yes→No 或 No→Yes），并用 PGD 在视频输入上做定向优化。
- 额外组件：使用 `Qwen/Qwen2.5-0.5B-Instruct` 作为 Judge，把模型回答解析为 `requires_takeover=true/false`（JSON）。
- 输入：`--data-root/takeover/` 目录下的 `.mp4` 文件（默认 `BDDX/takeover`）。
- 输出：
  - `output-dir/<video_id>/adv_<原文件名>.mp4`
  - `output-dir/<video_id>/log.json`：包含原始回答、攻击后回答、目标、是否成功、提前停止 step 等
  - `output-dir/summary_all.json`：把本次运行过程中“遇到的”（新攻击 + 复用的）视频日志汇总成一个列表

运行示例：

```bash
python 2_attack_misinfo.py \
  --data-root BDDX \
  --output-dir results/Llama/results_auto_flip \
  --limit 20 \
  --steps 200 \
  --num-frames 16
```

## 3) Universal Sponge（通用扰动/通用 Patch）：`3_unified_sponge.py`

- 目标：学习一个“通用”的视频扰动（UAP 或 trigger patch），让不同视频在同一个提问下都倾向输出更长的 `SPONGE_TARGET`（可用于可用性/延迟型攻击的放大）。
- 支持模式：
  - `--attack-mode uap_delta`：整帧加性通用扰动（可选 `--delta-mode shared_time/full_time`）
  - `--attack-mode patch_delta`：局部 patch 的加性通用扰动
  - `--attack-mode patch_replace`：学习一个“替换式”trigger patch（默认）
- 输出：
  - `output-dir/universal_params.pt`：训练得到的通用参数（用于复用/跨数据集）
  - `output-dir/eval/<video_id>/log.json`：每个视频攻击前后回答与长度对比
  - `output-dir/timing_summary_eval.json`：推理耗时对比统计

训练 + 同数据集评测示例（默认行为）：

```bash
python 3_unified_sponge.py \
  --stage train_eval \
  --data-root BDDX/videos \
  --output-dir results/Llama/results_bddx_universal_sponge \
  --num-train-videos 200 \
  --num-eval-videos 100 \
  --uap-epochs 10 \
  --uap-iters-per-video 15 \
  --num-frames 16
```

如果要换用 Video-LLaVA + 自动驾驶微调 adapter（例如 BDD-X）：

```bash
python 3_unified_sponge.py \
  --model-family video_llava \
  --model-path LanguageBind/Video-LLaVA-7B-hf \
  --adapter-path saychuwho/videollava_BDD-X-v1 \
  ...
```

## 4) Cross-Dataset（跨数据集）评测：复用 `universal_params.pt`

跨数据集评测不需要重新训练：直接加载在 Source 数据集上训练好的 `universal_params.pt`，在 Target 数据集上跑 `eval_only` 即可。

示例：在 BDDX 上训练的通用参数 → 在 DD 上评测：

```bash
python 3_unified_sponge.py \
  --stage eval_only \
  --load-params results/Llama/results_bddx_universal_sponge/universal_params.pt \
  --eval-data-root DD \
  --output-dir results/Llama/results_cross_bddx_to_dd \
  --num-eval-videos 20 \
  --num-frames 16
```

反向：在 DD 上训练的通用参数 → 在 BDDX 上评测：

```bash
python 3_unified_sponge.py \
  --stage eval_only \
  --load-params results/Llama/results_DD_universal_sponge/universal_params.pt \
  --eval-data-root BDDX/videos \
  --output-dir results/Llama/results_cross_dd_to_bddx \
  --num-eval-videos 100 \
  --num-frames 16
```

对比两次跨数据集评测的 token 与延迟（读取各自的 `final_summary_eval.json` + `timing_summary_eval.json`）：

```bash
python summarize_cross_dataset_eval.py \
  --run bddx_to_dd results/Llama/results_cross_bddx_to_dd \
  --run dd_to_bddx results/Llama/results_cross_dd_to_bddx \
  --out results/Llama/results_cross_reports
```

## 断点续跑 / 复用结果（跳过已攻击视频）

两个脚本都默认开启 `skip existing`：当发现 `output-dir/<video_id>/log.json` 已存在时，会直接跳过该视频，并把旧的 `log.json` 读入汇总结果里。

- 默认行为：跳过已存在结果（省 GPU 时间）
- 如需强制重跑：加 `--no-skip-existing`

示例（强制重跑）：

```bash
python 1_attack_sponge.py --no-skip-existing ...
python 2_attack_misinfo.py --no-skip-existing ...
```

`--limit` 的含义：限制“本次新攻击”的视频数量（不包含跳过复用的那些）。

## 环境与依赖

脚本依赖 GPU 推理/反向传播，且会从 HuggingFace 下载模型权重。

```bash
pip install -r requirements_4060.txt
```

如果你是从零环境开始，通常还需要：

```bash
pip install av tqdm accelerate
```
