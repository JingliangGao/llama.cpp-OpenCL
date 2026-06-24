#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// 性能分析器
//

// 性能分析事件类型
enum ggml_profile_event_type {
    GGML_PROFILE_EVENT_OP,    // 单个操作执行（计算内核）
    GGML_PROFILE_EVENT_COPY,  // 设备之间的数据传输
};

// 表示时间间隔的单个性能分析记录
typedef struct ggml_profile_record {
    enum ggml_profile_event_type type;
    const char *                 name;        // 算子名称（例如 "mul_mat", "copy_H2D"）
    int                          backend_id;  // 调度器的后端索引（0 = 最高优先级）
    int                          split_id;    // 属于哪个图分割（0..n_splits-1）
    uint64_t                     start_ns;    // 开始时间戳（纳秒）
    uint64_t                     end_ns;      // 结束时间戳（纳秒）
    uint64_t                     bytes;       // 传输的字节数（复制操作）或张量大小（运算操作）
    const char *                 extra;       // 融合操作的名称，或 NULL

    // 输出张量信息
    char                         tensor_name[GGML_MAX_NAME];           // 输出张量名称（例如 "ffn_out-0"），未命名则为空字符串
    int64_t                      ne[4];                                // 输出张量维度
    int                          out_type;                             // 输出张量类型（ggml_type），不适用则为 -1

    // 源张量（最多 GGML_MAX_SRC 个）。n_src 是实际填充的数量
    int                          n_src;
    int64_t                      ne_src[GGML_MAX_SRC][4];              // 每个源张量的维度
    int64_t                      nb_src[GGML_MAX_SRC][4];              // 每个源张量的步长（字节）
    int                          type_src[GGML_MAX_SRC];               // 每个源张量的 ggml_type，不存在则为 -1

    // 操作参数（从 ggml_tensor::op_params 复制的原始字节）
    int32_t                      op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

    int                          sub_op;      // 子操作（ggml_unary_op 或 ggml_glu_op），不适用则为 -1
} ggml_profile_record;

// 后端性能分析器接口 - 每个后端可选择实现此接口
// 用于提供细粒度的操作计时
struct ggml_backend_profiler {
    void * context;  // 后端特定的性能分析器上下文

    // 启用或禁用此后端的性能分析
    void (*enable)(void * context, bool enable);

    // 清除所有记录的数据
    void (*reset)(void * context);

    // 设置当前分割 ID（由调度器在 graph_compute 之前调用）
    void (*set_split_id)(void * context, int split_id);

    // 获取记录的性能分析数据
    // 返回记录数量；设置 *out 指向内部存储
    // 返回的指针在下次重置或禁用调用前保持有效
    int (*get_records)(void * context, const ggml_profile_record ** out);

    // 释放性能分析器上下文
    void (*free_context)(void * context);
};

typedef struct ggml_backend_profiler * ggml_backend_profiler_t;

// 从 ggml_tensor 节点填充 ggml_profile_record 的每个节点字段：
//   ne, out_type, n_src, ne_src, nb_src, type_src, op_params, sub_op。
// 所有其他字段（type/name/backend_id/split_id/timestamps/bytes/extra）必须
// 由记录事件的后端单独填充。
GGML_API void ggml_profile_record_from_tensor(struct ggml_profile_record * rec,
                                              const struct ggml_tensor *   node);

// 在后端上注册性能分析器（由后端在初始化期间调用）
// 性能分析器由后端拥有，将在后端释放时释放
GGML_API void ggml_backend_set_profiler(ggml_backend_t backend, ggml_backend_profiler_t profiler);

// 获取与后端关联的性能分析器（如果没有则返回 NULL）
GGML_API ggml_backend_profiler_t ggml_backend_get_profiler(ggml_backend_t backend);

//
// 调度器性能分析 API
//

// 启用或禁用调度器上的性能分析
// 启用时，调度器将：
//   - 计时后端之间的数据复制操作
//   - 在所有支持性能分析的后端上启用性能分析
//   - 在每次图计算后从所有后端收集性能分析记录
GGML_API void ggml_backend_sched_set_profiling(ggml_backend_sched_t sched, bool enable);

// 检查调度器上是否启用了性能分析
GGML_API bool ggml_backend_sched_get_profiling(ggml_backend_sched_t sched);

// 从上次图计算获取性能分析数据
// 记录由调度器拥有；在下次计算或重置前有效
// 返回记录数量
GGML_API int ggml_backend_sched_get_profiling_records(ggml_backend_sched_t sched, const ggml_profile_record ** records);

// 将上次性能分析运行的人类可读摘要打印到标准输出
// 按算子名称分组记录，并显示总计/计数/最小/最大/平均时间
GGML_API void ggml_backend_sched_print_profiling(ggml_backend_sched_t sched);

// 重置性能分析数据（清除所有记录的数据）
GGML_API void ggml_backend_sched_reset_profiling(ggml_backend_sched_t sched);

// 获取当前时间（纳秒）（如果需要手动性能分析）
GGML_API uint64_t ggml_profiler_time_ns(void);

// 将性能分析数据导出为 JSON 格式到文件
// 成功返回 0，失败返回 -1
GGML_API int ggml_backend_sched_export_profiling_json(ggml_backend_sched_t sched, const char * filepath);

// 将性能分析数据导出为 JSON 格式到 FILE 指针
GGML_API int ggml_backend_sched_write_profiling_json(ggml_backend_sched_t sched, FILE * fp);

// 将性能分析数据导出为纯文本统计信息到文件
// 成功返回 0，失败返回 -1
GGML_API int ggml_backend_sched_export_profiling_text(ggml_backend_sched_t sched, const char * filepath);

// 将性能分析数据导出为纯文本统计信息到 FILE 指针
GGML_API int ggml_backend_sched_write_profiling_text(ggml_backend_sched_t sched, FILE * fp);

#ifdef __cplusplus
}
#endif
