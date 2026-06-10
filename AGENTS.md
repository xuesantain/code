# AGENTS.md

## 项目说明

这个工程用于补充一套 Qwen 多模态预处理的“客户端帧列表 + 音频文件”对齐方案。

原始 Qwen 预处理在直接输入视频文件路径时，会由视频后端负责解码和抽帧，然后在 `fetch_video()` 的后半段统一做一次视频级 resize。  
但当输入 `video=[frame_0, frame_1, ...]` 这种帧列表时，原始逻辑通常会先对每一帧调用 `fetch_image()`，导致每帧提前 resize 一次，之后再进入视频级 resize。这样会让“同一个视频直接输入”和“先抽成帧列表再输入”的结果出现差异。

本工程的核心目标是：服务端正式接收客户端已经抽好的视频帧时，跳过原始 `fetch_image()` 的逐帧 resize，只保留最终的视频级统一 resize，从而尽量对齐直接输入视频文件路径时的处理结果。

## 关键文件

- `mm_aligned.py`
  - 正式运行代码。
  - 提供 `frames_to_video_tensor_no_fetch_image()`、`fetch_video_from_frames_aligned()`、`load_audio_aligned()`、`process_mm_info_client_aligned()`。
  - 重点原则是“不对每帧提前 resize，只做一次视频级 resize”。

- `test_alignment.py`
  - 测试和对比代码。
  - 提供 `simulate_client_payload_from_video()` 和 `compare_direct_video_vs_client_payload()`。
  - 测试时会复用原始 Qwen 视频后端抽出来的 pre-resize 帧，避免自己重新实现抽帧规则。

- `validate_qwen3_omni_alignment.py`
  - 按官方 Transformers 示例的调用形态做验证。
  - 直接路径使用 `qwen_omni_utils.process_mm_info()`。
  - 客户端路径使用 `process_mm_info_client_aligned_qwen3_omni()` 返回 `(audios, images, videos)`。
  - 可选加载 `Qwen3OmniMoeProcessor` 比较 processor 输出，但不会加载大模型或执行生成。

- `README.md`
  - 面向使用者的说明文档。
  - 包含正式运行方式、测试方式、对齐原则和依赖说明。

## 运行依赖

正式运行和测试需要放在包含原始 Qwen 多模态预处理代码的环境中。通常需要：

- PyTorch
- torchvision
- Pillow
- NumPy
- librosa
- soundfile
- 原始 Qwen 预处理模块，例如 `qwen_vl_utils.vision_process`

当前工程不会访问网络，也不依赖 OpenCV。

## 维护原则

修改这个工程时，请优先保持和原始 Qwen 逻辑兼容，不要大规模改写原始 `fetch_video()`。  
新增逻辑应尽量集中在新增函数里，尤其是客户端帧列表路径。

需要特别注意：

- `fetch_video_from_frames_aligned()` 不能调用原始 `fetch_image()`。
- 客户端帧只允许做格式转换：RGB、uint8、CHW、stack。
- 如果帧数不是 `FRAME_FACTOR` 的倍数，需要复制最后一帧补齐。
- resize 只能发生在视频 tensor stack 之后。
- 测试模拟客户端帧时，应复用原始 `VIDEO_READER_BACKENDS[get_video_reader_backend()]` 的抽帧结果。
- 音频提取要尊重 `video_start`、`video_end` 和 16000 Hz 采样率。

## 常用测试命令

在真实 Qwen 环境中运行：

```bash
python test_alignment.py --video /path/to/demo.mp4 --fps 2.0 --client-frame-format tensor
python test_alignment.py --video /path/to/demo.mp4 --fps 2.0 --client-frame-format pil
```

`tensor` 模式通常最接近直接视频路径。  
`pil` 模式更像普通客户端上传图片帧，但仍避免 JPEG 压缩损失。

## 给后续代码代理的提示

如果需要继续扩展，请先阅读 `mm_aligned.py` 中的视频路径实现。  
本项目最重要的行为边界是：客户端帧列表路径不能走原始 list-frame `fetch_video()` 分支，因为那个分支会触发逐帧 `fetch_image()` resize，从而破坏和直接视频路径的对齐。
