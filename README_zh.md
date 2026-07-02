# llama.cpp

[English](README.md) | [中文](README_zh.md)

分支 dev-unify_profiler

我们添加了一个内置的跨后端 profiler，可以捕获所有计算后端上每个操作的时序、数据传输成本和张量形状。它适用于任何基于 ggml 调度器构建的应用程序——无需修改源代码。

## 构建项目
```bash
chmod +x build-for-debug.sh && build-for-debug.sh
```

## 支持的后端

| 后端 | 状态 | 计时方法 |
|---------|--------|---------------|
| CPU     | 已支持 | Wall-clock  (`CLOCK_MONOTONIC_RAW`) |
| CUDA    | 已支持 | `cudaEvent` GPU 时间戳 |
| Vulkan  | 已支持 | GPU 时间戳查询 |
| BLAS    | 已支持 | Wall-clock  |
| Metal   | 暂不支持 | — |
| OpenCL  | 暂不支持 | — |

调度器还会分析后端之间的**数据拷贝**（H2D、D2H、D2D），无论哪些后端具有原生 profiler 支持。

## 启用 Profiler

有两种独立的方式启用 profiling。它们可以单独使用或一起使用。

### CLI 标志 (`--profile`, `--profile-output`)

在 `llama-cli`、`llama-completion`、`llama-server` 和 `debug` 中可用：

```bash
# 打印摘要到标准输出
llama-completion -m model.gguf --profile -p "Hello world"

# 导出到 JSON
llama-completion -m model.gguf --profile --profile-output profile.json -p "Hello world"

# 导出为纯文本
llama-completion -m model.gguf --profile --profile-output profile.txt -p "Hello world"
```

输出格式由文件扩展名决定：`.json` 表示 JSON，`.txt` 表示纯文本。其他扩展名默认为 JSON。

### 环境变量 (`GGML_PROFILE`)

`GGML_PROFILE` 环境变量在 ggml 调度器级别启用 profiling。这适用于**任何**使用调度器的应用程序——包括 `sd.cpp` 等第三方工具——无需 CLI 标志支持。

```bash
# 打印摘要到标准输出
GGML_PROFILE=1 llama-completion -m model.gguf -p "Hello world"

# 导出 JSON
GGML_PROFILE=profile.json llama-completion -m model.gguf -p "Hello world"

# 导出纯文本
GGML_PROFILE=profile.txt llama-completion -m model.gguf -p "Hello world"

# 适用于任何基于 ggml 的应用程序
GGML_PROFILE=1 sd -m model.gguf -p "a cat"
```

| 值 | 行为 |
|-------|----------|
| `1`、`stdout` 或空 | 打印摘要到标准输出 |
| `path.json` | 导出 JSON 到文件 |
| `path.txt` | 导出纯文本到文件 |
| 其他任何路径 | 导出 JSON 到文件 |

导出会在调度器被释放时自动进行（通常在程序退出时）。

## 输出格式

### 控制台摘要 (stdout)

当使用 `--profile` 但不使用 `--profile-output`，或 `GGML_PROFILE=1` 时的默认输出：

```
=== Profiling Summary ===
  [OP  ] backend 0 MUL_MAT                    45.2%  count=1200  total=  120.50 ms  avg=  100.42 us  ...  12.30 GB/s  [4096 x 4096]
  [OP  ] backend 1 MUL_MAT_ID                 30.1%  count= 600  total=   80.20 ms  avg=  133.67 us  ...   0.08 GB/s  [2688 x 1856 x 128]
  [COPY] backend 0 copy_H2D                    5.3%  count= 200  total=   14.10 ms  avg=   70.50 us  ...   2.50 GB/s
  ...
```

每行显示：事件类型（OP 或 COPY）、后端索引、操作名称、总时间百分比、调用次数、时序统计、带宽和代表性张量形状。

### 纯文本 (`.txt`)

包含三个部分的更详细报告：

1. **Profiling Summary** — 总时间、记录数、唯一操作数
2. **Per-Backend Summary** — 每个后端的操作和拷贝，以及聚合带宽
3. **Operations table** — 完整细分，包含所有源张量的带宽和张量形状

### JSON (`.json`)

适合 Python 分析工具的机器可读格式。包含：

- `version`: 格式版本（当前为 `2`）
- `backends[]`: 后端元数据（名称、设备、设备类型）
- `records[]`: 每个 profiling 事件，包含：
  - `type`: `0` = OP，`1` = COPY
  - `name`: 操作名称（例如 `"MUL_MAT"`、`"copy_H2D"`）
  - `backend_id`, `split_id`: 调度器索引
  - `start_ns`, `duration_ns`: 纳秒级计时
  - `bytes`: 输出张量大小（OP）或传输大小（COPY）
  - `extra`: 融合操作的融合名称，或 `null`
  - `ne_src0`, `ne_src1`, `ne_src2`: 源张量维度（4 元素数组）

`ne_src2` 仅在 `MUL_MAT_ID`（专家选择索引）时填充；对于所有其他操作，它为 `[0,0,0,0]`。

## Python 分析工具

`tools/profiler/profiler.py` 脚本读取 JSON 导出并生成分析报告和可视化。

### 基本用法

```bash
# 打印摘要
python -m tools.profiler.profiler profile.json

# 显示按时间排序的前 10 个操作
python -m tools.profiler.profiler profile.json --top-ops 10

# 显示前 10 个最长的单独内核
python -m tools.profiler.profiler profile.json --top-kernels 10

# 显示低效排名（最高时间/字节比）
python -m tools.profiler.profiler profile.json --inefficiency
```

### 导出可视化

```bash
# 交互式 HTML 时间线（自包含，无依赖）
python -m tools.profiler.profiler profile.json --html-viewer timeline.html

# Chrome Trace 格式（在 chrome://tracing 或 Perfetto 中打开）
python -m tools.profiler.profiler profile.json --chrome-trace trace.json

# 为 HTML 查看器降采样大 trace
python -m tools.profiler.profiler profile.json --html-viewer timeline.html --html-max-records 50000
```

可以在一次调用中组合多个导出：

```bash
python -m tools.profiler.profiler profile.json --html-viewer timeline.html --chrome-trace trace.json --top-ops 20
```

### CLI 参考

| 参数 | 描述 |
|----------|-------------|
| `profile` (位置参数) | Profiler JSON 文件路径 |
| `--chrome-trace FILE` | 导出 Chrome Trace Event 格式 |
| `--html-viewer FILE` | 导出交互式 HTML 时间线 |
| `--html-max-records N` | 限制 HTML 输出中的记录数（0 = 无限制） |
| `--top-ops N` | 显示按总时间排序的前 N 个操作 |
| `--top-kernels N` | 显示前 N 个最长的单独内核 |
| `--inefficiency` | 按时间/字节比排名操作（越高越差） |

### HTML 查看器功能

HTML 查看器是一个自包含文件，无外部依赖：

- **Canvas 时间线**，带有每个后端通道和颜色编码的操作
- **缩放控制**（1s / 100ms / 1ms / 100us）和鼠标拖拽导航
- **小地图**，显示完整 trace 和视口指示器
- **悬停工具提示**，显示操作名称、持续时间、形状和字节数
- **统计表格**，带有可折叠树：操作 → 后端 → 张量形状，显示时间百分比、计数、平均/最小/最大和带宽
- **图例**，显示最频繁的操作类型

## 测量内容

### OP 事件

每个张量操作（MUL_MAT、ADD、UNARY、FLASH_ATTN_EXT 等）记录：

- **Timing**: 开始/结束时间戳（纳秒精度）
- **Bytes**: 输出张量大小（`ggml_nbytes(node)`）
- **Tensor shapes**: `src[0]`、`src[1]` 和 `src[2]` 的维度（如适用）
- **Bandwidth**: 计算为 `bytes / duration` — 用于识别内存受限 vs 计算受限操作

### COPY 事件

后端之间的数据传输：

- **Direction**: `copy_H2D`（主机→设备）、`copy_D2H`（设备→主机）、`copy_D2D`（设备→设备）
- **Bytes**: 精确传输大小
- **Bandwidth**: 传输吞吐量

### MoE 权重拷贝

当使用 `--cpu-moe` 时，调度器选择性地仅拷贝活动专家。这些部分拷贝被记录为单独的 COPY 事件，带有实际传输的字节数。

## 编程 API

对于自定义应用程序，可以通过 `ggml/include/ggml-profiler.h` 中定义的 C API 控制 profiler：

```c
// 在调度器上启用 profiling
ggml_backend_sched_set_profiling(sched, true);

// ... 运行推理 ...

// 获取原始记录
const ggml_profile_record * records;
int n = ggml_backend_sched_get_profiling_records(sched, &records);

// 或直接导出
ggml_backend_sched_print_profiling(sched);                         // stdout
ggml_backend_sched_export_profiling_json(sched, "profile.json");   // JSON 文件
ggml_backend_sched_export_profiling_text(sched, "profile.txt");    // 文本文件
ggml_backend_sched_write_profiling_json(sched, fp);                // JSON 到 FILE*
ggml_backend_sched_write_profiling_text(sched, fp);                // 文本到 FILE*

// 重置以进行下一个测量窗口
ggml_backend_sched_reset_profiling(sched);
```
记录在多次 `graph_compute` 调用中累积，直到显式重置或调度器被释放。
## 提示
- **Prompt 评估 vs 生成**: Profiler 捕获所有图计算。在 prompt 评估期间，张量形状中会看到较大的批大小；在生成期间，批大小通常为 1-2。
- **Vulkan 并发模式**: 当 Vulkan 并发调度多个操作时，它们被报告为单个组合记录，跨越完整的 GPU 时间间隔。
- **带宽解释**: 对于计算操作，带宽 = `output_bytes / duration`。这不是内存带宽——它是吞吐量的代理。低带宽的 MUL_MAT 通常表示计算受限行为；高带宽表示内存受限。
- **大型 trace**: 对于长时间的推理运行，JSON 可能很大。使用 `--html-max-records` 对 HTML 查看器进行降采样，或使用 Chrome Trace 格式，它能很好地处理大文件。
- **多个后端**: 输出中的后端 ID 对应于调度器的优先级顺序（0 = 最高优先级，通常是 GPU；最后 = CPU）。
