# Qwen 多模态客户端帧对齐预处理

这个工程用于解决一个特定问题：正式服务端不再接收原始视频文件，而是接收客户端已经抽好的视频帧列表和音频文件时，如何让处理结果尽量对齐“直接输入原始视频路径”的 Qwen 多模态预处理结果。

核心原则很简单：客户端帧列表路径不能再走原始 `fetch_video()` 的 list-frame 分支，因为那个分支通常会对每一帧先调用 `fetch_image()`，造成额外的逐帧 resize。对齐版路径会跳过 `fetch_image()`，只在视频帧 stack 成 `(T, C, H, W)` 后做一次视频级统一 resize。

## 适用场景

正式上线后，服务端接收如下格式的 `conversations`：

```python
[
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [frame_0, frame_1, frame_2],
                "sample_fps": 2.0,
                "raw_fps": 30.0,
                "frames_indices": [0, 15, 30],
                "total_num_frames": 300,
                "min_pixels": 3136,
                "max_pixels": 602112
            },
            {
                "type": "audio",
                "audio": "/path/to/audio.wav"
            },
            {
                "type": "text",
                "text": "请回答问题"
            }
        ]
    }
]
```

视频帧支持：

- `PIL.Image`
- `numpy.ndarray`
- `torch.Tensor`
- 单帧或批量形式，例如 HWC、CHW、THWC、TCHW

音频支持：

- 本地 wav 路径
- `file://` 本地 URI
- mono `numpy.ndarray`

## 文件说明

- `mm_aligned.py`
  - 正式运行代码。
  - `frames_to_video_tensor_no_fetch_image()`：把客户端帧转成 RGB uint8 CHW tensor，不做 resize。
  - `fetch_video_from_frames_aligned()`：stack 成视频 tensor，按 `FRAME_FACTOR` 补齐，然后只做一次视频级 resize。
  - `load_audio_aligned()`：加载 wav、`file://` 或 mono numpy 音频，并统一到 16000 Hz。
  - `process_mm_info_client_aligned()`：面向服务端使用的新入口，返回结构尽量兼容原始预处理。

- `test_alignment.py`
  - 测试和对比代码。
  - `simulate_client_payload_from_video()`：从本地视频模拟客户端上传的 frames + wav。
  - `compare_direct_video_vs_client_payload()`：比较直接视频路径和客户端 payload 路径的差异。

- `validate_qwen3_omni_alignment.py`
  - 按官方 Transformers 调用形态验证直接视频路径和客户端帧列表路径。

- `trace_utils.py`
  - 统一保存 JSON / JSONL trace，并用 shape、dtype、hash 摘要表示大 tensor。

- `trace_transformers_qwen3_omni.py`
  - 保存 Transformers Thinker 文本链路的预处理 trace、逐步 token、top-k logprob 和最终文本。

- `trace_vllm_qwen3_omni.py`
  - 保存 vLLM Python offline Thinker 文本链路的同格式 trace。

- `compare_service_outputs.py`
  - 离线比较 Transformers 和 vLLM 生成的 trace，定位 prompt、预处理输入或 decode step 的第一处分叉。

- `AGENTS.md`
  - 给后续协作人员或代码代理看的工程说明和维护边界。

## 正式运行示例

```python
from mm_aligned import process_mm_info_client_aligned

image_inputs, video_inputs, audio_inputs, video_kwargs, video_metadata = (
    process_mm_info_client_aligned(
        conversations,
        image_patch_size=14,
        return_video_kwargs=True,
        return_video_metadata=True,
        sample_rate=16000,
    )
)
```

返回值：

- `image_inputs`
- `video_inputs`
- `audio_inputs`
- 可选 `video_kwargs`
- 可选 `video_metadata`

## 对齐测试

测试时输入同一个本地视频文件，代码会跑两条路径：

- 路径 A：直接把 video path 交给原始 `fetch_video()`。
- 路径 B：先调用原始视频后端得到 pre-resize 帧，再把这些帧模拟成客户端 payload，交给 `fetch_video_from_frames_aligned()`。

运行示例：

```bash
python test_alignment.py --video /path/to/demo.mp4 --fps 2.0 --client-frame-format tensor
python test_alignment.py --video /path/to/demo.mp4 --fps 2.0 --client-frame-format pil
```

## 按官方 Transformers 服务方式验证

官方 Qwen3-Omni 示例里通常是：

```python
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=True,
)
```

本项目额外提供了 `validate_qwen3_omni_alignment.py`，用于按这个调用形态验证两条路径：

- 直接路径：`qwen_omni_utils.process_mm_info()` 处理原始 video path。
- 客户端路径：先模拟客户端 frames + wav，再用 `process_mm_info_client_aligned_qwen3_omni()` 返回 `(audios, images, videos)`。

只验证预处理输出：

```bash
python validate_qwen3_omni_alignment.py --video /path/to/demo.mp4 --fps 2.0 --client-frame-format tensor
```

同时验证 `Qwen3OmniMoeProcessor` 输出：

```bash
python validate_qwen3_omni_alignment.py \
  --video /path/to/demo.mp4 \
  --fps 2.0 \
  --client-frame-format tensor \
  --run-processor \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

如果模型和 processor 已经下载到本地，可以加：

```bash
--local-files-only
```

注意：这个验证脚本只需要加载 processor，不会加载 30B 模型，也不会执行 `model.generate()`。

`tensor` 模式通常最接近直接视频路径。  
`pil` 模式更接近真实客户端上传图片帧的形式，但可能因为 PIL 转换带来极小差异。

CLI 会输出类似如下字段：

```json
{
  "video_shape_equal": true,
  "direct_video_shape": [4, 3, 448, 448],
  "client_video_shape": [4, 3, 448, 448],
  "sample_fps_equal": true,
  "frames_indices_equal": true,
  "video_max_abs_diff": 0.0,
  "video_mean_abs_diff": 0.0,
  "audio_length_equal": true,
  "audio_max_abs_diff": 0.0,
  "audio_mean_abs_diff": 0.0,
  "passed": true
}
```

如果 shape 不一致，测试代码不会强行相减，会先报告 shape mismatch。

## 比较 Transformers 和 vLLM 服务输出

如果要比较两个框架的服务输出是否一致，建议先只比较 Qwen3-Omni Thinker 文本生成链路，不要一开始比较 hidden states 或 Talker 音频输出。流程是两边分别保存统一格式的 trace，然后离线比较：

1. `chat_template_text` 是否完全一致。
2. prompt token ids 是否完全一致。
3. image / audio / video 的数量、shape、dtype、hash 是否一致。
4. 每一步生成的 `chosen_token_id` 是否一致。
5. 第一次 token 不一致时，比较上一处或当前 step 的 top-k logprob。

准备一个 conversation JSON 文件，例如 `conversation.json`：

```json
[
  {
    "role": "user",
    "content": [
      {"type": "video", "video": "/path/to/demo.mp4"},
      {"type": "text", "text": "What can you see and hear? Answer in one short sentence."}
    ]
  }
]
```

Transformers 侧保存 trace：

```bash
python trace_transformers_qwen3_omni.py \
  --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --conversation-file conversation.json \
  --output-dir traces/hf \
  --local-files-only
```

vLLM Python offline 侧保存同格式 trace：

```bash
python trace_vllm_qwen3_omni.py \
  --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --conversation-file conversation.json \
  --output-dir traces/vllm \
  --local-files-only \
  --trust-remote-code
```

离线比较：

```bash
python compare_service_outputs.py \
  --hf-preprocess traces/hf/hf_preprocess_trace.json \
  --vllm-preprocess traces/vllm/vllm_preprocess_trace.json \
  --hf-steps traces/hf/hf_decode_steps.jsonl \
  --vllm-steps traces/vllm/vllm_decode_steps.jsonl \
  --hf-output traces/hf/hf_output.json \
  --vllm-output traces/vllm/vllm_output.json \
  --output service_compare_report.json
```

比较脚本会输出 `passed`、`prompt_ids_equal`、`raw_mm_equal`、`matching_prefix_steps`、`first_token_divergence` 等字段。若 prompt 或多模态输入摘要已经不同，后续生成文本不同通常是预期结果，应先修输入链路。

## 依赖

代码不访问网络，不使用 OpenCV。实际运行环境需要：

- PyTorch
- torchvision
- Pillow
- NumPy
- librosa
- soundfile
- 原始 Qwen 预处理模块，例如 `qwen_vl_utils.vision_process`

当前代码会优先复用原始 Qwen 模块里的常量和函数，例如 `smart_resize`、`FRAME_FACTOR`、`SPATIAL_MERGE_SIZE`、`VIDEO_READER_BACKENDS` 和 `get_video_reader_backend()`。如果这些模块不可 import，直接视频测试路径会给出明确错误。

## 维护注意事项

请保持以下边界：

- 不要在 `fetch_video_from_frames_aligned()` 中调用 `fetch_image()`。
- 不要对客户端每一帧提前 resize。
- 客户端帧只做 RGB、uint8、CHW 转换。
- 视频 resize 只能发生在 stack 成 `(T, C, H, W)` 之后。
- 帧数不是 `FRAME_FACTOR` 倍数时，复制最后一帧补齐。
- 测试模拟客户端帧时，必须复用原始视频后端抽帧结果，不要自己重新按 fps 抽帧。
- 音频提取需要尊重 `video_start`、`video_end` 和 16000 Hz 采样率。
