# kCacheKVTrim 设计说明

技术可行性参照 [ScissorHands](https://arxiv.org/pdf/2305.17118.pdf)，论文没有提供样例。lmdeploy 的实现和论文相去甚远。

## 原理和方案

论文认为 attention score 能够代表 kv cache 的重要性，分数低的 past_kv 不参与新 token 的生成，对精度影响不大。
这是我们基于 huggingface python，用 LLama-7B 在 gsm8k 上的测试结果：

| 版本 |                 说明                 | 精度 |
| :--: | :----------------------------------: | :--: |
|  A   |               baseline               | 10.1 |
|  B   | 保留 1024 token，每 128 token 剪一次 | 9.4  |
|  C   | 保留 512 token，每 128 token 剪一次  | 8.87 |
|  D   |        对称 kCacheKVInt8 量化        | 8.26 |
|  E   |             同时使用 B+D             | 8.04 |

B+D 方案同时开启，kv_cache 显存是 fp16 版的 25%（对称量化精度还有很大提升空间）

假设项目配置如下：

- 模型最长 token 2048
- 配置 kCacheKVTrim 最多使用 1024 长度 kv_cache
- 每生成 128 token 剪一次

lmdeploy 的实现：

- 用户第一轮对话，输入 1280 个字符。

  1. 字符被拆解成 10 段 128 字符

  2. Context 阶段。第 1 段执行 `unfusedMultiHeadAttention`，得到 128 个 kv_cache；第 8 段执行 mha，1024 个 kv_cache 占满 buffer，开始 trim：

     - 保留当前最近的 128 个 kv_cache，在剩余的历史 896 个选出 score 最高的 768 个跟当前合并，最终得到 896 个 kv_cache

  3. Generate 阶段。每次生成 1 个字符，直到 kv_cache 再次达到 1024，剪到 896 长度

- 第二轮对话。用户输入 200 字符。

  1. 同样拆成 128 + 72 两段，执行 Context 和 Generate。

因为最大 buf 约束在 1024，所以 Context 和 Generate 都需要判断是否溢出。

因为 trim 有很多 memmov 操作，实际上很耗时，所以不能每生成 1 个字符就 trim，必须“插帧”分担成本。

因为攒到 1024 才 trim，所以需要缓存很多 attention_score_mean 。这是有额外开销的。

// TODO 10 段 attention_score 怎么算一个大 score
