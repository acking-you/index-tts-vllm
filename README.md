<a href="README.md">中文</a> ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

## 项目简介
该项目在 [index-tts](https://github.com/index-tts/index-tts) 的基础上使用 vllm 库重新实现了 gpt 模型的推理，加速了 index-tts 的推理过程。

推理速度（Index-TTS-v1/v1.5）在单卡 RTX 4090 上的提升为：
- 单个请求的 RTF (Real-Time Factor)：≈0.3 -> ≈0.1
- 单个请求的 gpt 模型 decode 速度：≈90 token / s -> ≈280 token / s
- 并发量：gpu_memory_utilization 设置为 0.25（约5GB显存）的情况下，实测 16 左右的并发无压力（测速脚本参考 `simple_test.py`）

## 更新日志

- **[2025-09-22]** 支持了 vllm v1 版本，IndexTTS2 正在兼容中

- **[2025-09-28]** 支持了 IndexTTS2 的 webui 推理，并整理了权重文件，现在部署更加方便了！ \0.0/ ；但当前版本对于 IndexTTS2 的 gpt 似乎并没有加速效果，待研究

- **[2025-09-29]** 解决了 IndexTTS2 的 gpt 模型推理加速无效的问题

- **[2025-10-09]** 兼容 IndexTTS2 的 api 接口调用，请参考 [API](#api)；v1/1.5 的 api 接口以及 openai 兼容的接口可能还有 bug，晚点再修

- **[2025-10-19]** 支持 qwen0.6bemo4-merge 的 vllm 推理

## TODO list
- V2 api 的并发优化：目前只有 gpt2 模型的推理是并行的，其他模块均是串行，而其中 s2mel 的推理开销大（需要 DiT 迭代 25 步），十分影响并发性能

- s2mel 的推理加速

## 使用步骤

### 0. 环境要求（重要）

- 建议：Linux / WSL2 + NVIDIA 显卡（vLLM 基本只支持 NVIDIA CUDA 环境）
- Python：3.12（本项目已提供 `.python-version`）
- 系统依赖：建议已安装 `ffmpeg`

### 1. git 本项目
```bash
git clone https://github.com/acking-you/index-tts-vllm.git
cd index-tts-vllm
```


### 2. 一键启动（推荐：uv）

默认启动 **IndexTTS-1.5 WebUI**，并自动完成：
- 安装/使用 `uv`
- `uv sync` 安装 Python 依赖
- 若未检测到模型文件则自动从 ModelScope 下载权重

```bash
./run_webui.sh
```

常用参数：

```bash
# 指定端口/host
./run_webui.sh --host 127.0.0.1 --port 6006

# 切换版本：1.0 / 1.5 / 2.0
./run_webui.sh --version 1.5

# IndexTTS-2：qwen 情感模型显存占用（遇到 “No available memory for the cache blocks” 时调大）
./run_webui.sh --version 2 --qwenemo-gpu-memory-utilization 0.20

# 关闭 BigVGAN CUDA 扩展编译（遇到 compute_120 / nvcc 版本偏旧时推荐）
./run_webui.sh --no-cuda-kernel

# 一键安装系统依赖（需要 sudo；ffmpeg 等）
./run_webui.sh --install-system-deps

# 安装/升级 CUDA Toolkit（需要 sudo；用于让 nvcc 支持 compute_120 并启用 CUDA 扩展）
./run_webui.sh --cuda-kernel --install-cuda-toolkit

# 关闭自动下载（只安装依赖 + 启动；模型需手动准备）
./run_webui.sh --no-download

# 指定模型目录
./run_webui.sh --model-dir ./checkpoints/Index-TTS-1.5-vLLM
```

### 3. 手动方式（uv）

如果你不想用一键脚本，也可以手动执行：

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖（优先使用 lockfile）
uv sync --frozen
```

然后参考下方步骤下载模型权重并启动 WebUI。

### 4. 下载模型权重

#### 自动下载（推荐）

选择对应版本的模型权重下载到 `checkpoints/` 路径下：

```bash
# Index-TTS
uv run modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM

# IndexTTS-1.5
uv run modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM

# IndexTTS-2
uv run modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
```

#### 手动下载

- ModelScope：[Index-TTS](https://www.modelscope.cn/models/kusuriuri/Index-TTS-vLLM) | [IndexTTS-1.5](https://www.modelscope.cn/models/kusuriuri/Index-TTS-1.5-vLLM) | [IndexTTS-2](https://www.modelscope.cn/models/kusuriuri/IndexTTS-2-vLLM)

#### 自行转换原权重（可选，不推荐）

可以使用 `convert_hf_format.sh` 自行转换官方权重文件：

```bash
bash convert_hf_format.sh /path/to/your/model_dir
```

### 5. webui 启动！

运行对应版本（第一次启动可能会久一些，因为要对 bigvgan 进行 cuda 核编译）：

```bash
# Index-TTS 1.0
uv run python webui.py

# IndexTTS-1.5
uv run python webui.py --version 1.5

# IndexTTS-2
uv run python webui_v2.py
```

> 说明：首次启动可能会尝试编译 BigVGAN 的 CUDA 扩展。如果你看到类似 `nvcc fatal: Unsupported gpu architecture 'compute_120'` 的报错，通常是 CUDA Toolkit（nvcc）版本偏旧导致；程序会自动回退到 torch 实现，一般不影响使用。你也可以通过 `./run_webui.sh --no-cuda-kernel` 或设置环境变量 `INDEXTTS_USE_CUDA_KERNEL=0` 来直接跳过编译。

## API

使用 fastapi 封装了 api 接口，启动示例如下：

```bash
# Index-TTS-1.0/1.5
uv run python api_server.py

# IndexTTS-2
uv run python api_server_v2.py
```

### 启动参数
- `--model_dir`: 必填，模型权重路径
- `--host`: 服务ip地址，默认为 `0.0.0.0`
- `--port`: 服务端口，默认为 `6006`
- `--gpu_memory_utilization`: vllm 显存占用率，默认设置为 `0.25`

### API 请求示例
- v1/1.5 请参考 `api_example.py`
- v2 请参考 `api_example_v2.py`

### OpenAI API
- 添加 /audio/speech api 路径，兼容 OpenAI 接口
- 添加 /audio/voices api 路径， 获得 voice/character 列表

详见：[createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## 新特性
- **v1/v1.5:** 支持多角色音频混合：可以传入多个参考音频，TTS 输出的角色声线为多个参考音频的混合版本（输入多个参考音频会导致输出的角色声线不稳定，可以抽卡抽到满意的声线再作为参考音频）

## 性能
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| model                   | zh    | en    |
| ----------------------- | ----- | ----- |
| Human                   | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm      | 1.12  | 1.987 |

基本保持了原项目的性能

## 并发测试
参考 [`simple_test.py`](simple_test.py)，需先启动 API 服务
