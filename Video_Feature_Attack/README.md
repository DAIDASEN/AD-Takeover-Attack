# Video Feature Attack（视频特征攻击）

本目录包含两类针对 `LLaVA-NeXT-Video` 的视频输入对抗攻击脚本，并把每个视频的结果落盘到 `output-dir/<video_id>/`，方便断点续跑/复用结果。

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
  --output-dir results_sponge_sample \
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
  --output-dir results_auto_flip \
  --limit 20 \
  --steps 200 \
  --num-frames 16
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

