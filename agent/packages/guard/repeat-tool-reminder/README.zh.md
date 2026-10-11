---
description: "建议性循环卫生 guard：当 agent 重复完全相同的工具调用、或反复得到相同结果时提醒模型，供选择、配置或排查此插件的用户与维护者阅读。"
kind: "package-reference"
---

# @deepseek-ai/dsh-repeat-tool-reminder

[English](README.md) | 中文

## 概述

模型可能会卡在以相同参数调用同一工具上——反复运行失败的命令、反复读取未变化的文件——白白消耗时间和 token 却没有进展。`dsh-repeat-tool-reminder` 会发现这种模式并让模型停下来：在选定的重复次数上，它送出一条提醒，要求模型分析上一次结果并改用其他方法或结束任务。提醒只是建议，绝非阻止：合理的重复调用不会被延迟分毫，是否继续、改变方法或停止仍由模型决定。它分别跟踪每个 agent（智能体），一个 agent 的循环绝不会干扰另一个 agent 的工作，新的用户消息会清零计数。它随 `dsh` base 组合默认启用，在 3、5、8 次重复时提醒。循环也可能每次改变调用——每次换一个脚本名——却得到同样的答案；开启相同结果检测后（base 组合为 `pwsh` 与 `bash` 开启），guard 也能发现这种循环。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)
- [模型体验](#model-experience)
- [已知限制与延期工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用本包

当模型应当自行发现自己在相同工具调用上循环时，挂载此插件。无需学习或接线：`dsh` base 组合已经运行它，默认值适用于大多数会话——想更早、更晚或在更少的工具上收到提醒时，调优下面的阈值与工具范围即可。

### 何时选择

当模型长时间自主工作、且卡住的循环是你想用建议而非强制来打破的失败模式时，选择它。当相同的重复是合理且必须不受打扰地运行时——guard 只会提醒，提醒只是重复调用之后的一条小消息——以及必须捕获输出不同的近似调用时（重复链只匹配精确重复——同一工具、同一参数且与属性顺序无关——相同结果检测只匹配相同的结果），避免使用它。

### 设置阈值与范围

想改变提醒何时触发或覆盖哪些工具时，用配置挂载插件：

```yaml
- name: '@deepseek-ai/dsh-repeat-tool-reminder'
  config:
    thresholds: [3, 5, 8]        # remind at 3, 5, and 8 consecutive repeats
    include: []                  # track every tool; list patterns to track only some
    exclude: [todo_write]        # never track these tools
    argumentsPreviewChars: 500   # cap on arguments shown in the detailed reminder
    resultThresholds: [3, 5, 8]  # remind when one tool returns the same result 3, 5, 8 times
    resultInclude: [pwsh, bash]  # compare only these tools' results
    resultWindow: 12             # ...within the agent's last 12 such calls
    resultPreviewChars: 500      # cap on the result shown in the detailed reminder
    reminderSuffix: ''           # text appended to every reminder
```

| 字段 | 默认值 | 含义 |
|---|---|---|
| `thresholds` | `[3, 5, 8]` | 触发提醒的重复次数 |
| `include` | `[]` | 只跟踪这些工具；空表示所有工具 |
| `exclude` | `[]` | 绝不跟踪这些工具；对它们的调用既不计数也不重置 |
| `argumentsPreviewChars` | `500` | 详细提醒中显示多少字符的重复参数 |
| `resultThresholds` | `[]`（关闭） | 触发提醒的相同结果次数，与调用参数无关 |
| `resultInclude` | `[]` | 只比较这些被跟踪工具的结果；空表示所有被跟踪工具 |
| `resultWindow` | `12` | 在该 agent 最近多少次参与比较的调用中查找相同结果 |
| `resultPreviewChars` | `500` | 详细的相同结果提醒中显示多少字符的重复结果 |
| `reminderSuffix` | `''` | 追加到每条提醒末尾的文本，例如提高推理强度的聊天模板标记 |

无效配置会在启动时以清晰错误失败——空的 `thresholds` 列表、小于 2 的重复次数、重复值，或小于 `resultThresholds` 最大值的 `resultWindow`——绝不会静默改变行为。生成的[配置目录](../../../docs/config-catalog.zh.md#deepseek-aidsh-repeat-tool-reminder)记录每个受支持的值。

### 你会得到什么

按默认值，以相同参数重复同一调用的模型会在第三次重复时收到简短提醒——先分析上一次结果再调用——并在第五次和第八次收到详细提醒，列出工具与重复参数，使其决定改变方法、收集更多证据还是结束任务。新的用户消息会清零计数，因此全新指令绝不会被当作循环。提醒出现在重复调用的结果之后、归属于插件，模型像阅读任何其他消息一样阅读它。

开启相同结果检测后，某个工具在窗口内第三次返回相同输出时——即使每次调用运行的是不同的脚本——模型会收到简短提醒：停止尝试各种变体，重新审视自己的假设，包括任务中给出的预期，因为它本身可能是错的。第五次和第八次相同结果会收到引用该结果的详细提醒。在下一条用户消息之前，每个阈值对每个不同的结果只提醒一次。

-----

<a id="understand-the-implementation"></a>
## 理解实现

<details>
<summary>实现细节——点击展开</summary>

本节解释 guard 如何检测重复并投递提醒，并指出实现它的代码位置；可观察行为已在[使用本包](#use-this-package)中完整说明。

### 设计理念

guard 建立在四项承诺之上：

- **仅建议，不否决。** guard 用模型上下文丰富 post-execute 决策；它从不阻止或改写调用，因此 `PostToolDecision` 阻止仍是后续监听器的事。
- **在 post-execute 中计数。** 检测运行在 `tools/post-execute` 上，被拒绝的调用同样会经过它；在那里计数让一个监听器即可覆盖所有尝试，无需跨事件状态。
- **精确匹配规范化。** 参数以循环的 `JSON.parse` 输出（或畸形参数 JSON 的原始字符串回退）到达 guard，因此 JSON 的值域就是全部输入域，深度键排序加 `JSON.stringify` 是完整、确定性的同一性判定——不存在 bigint、循环引用或 `undefined` 处理，因为没有输入路径能产生它们。
- **加载时快速失败。** 两个阈值列表、预览上限与 `resultWindow` 在 `apply` 中校验并抛出错误，绝不回退到默认值。

### 检测：重复链

每个 agent 的链以「`(tool name, canonical arguments)`」为键——同一工具且规范化后参数相同（忽略属性顺序）的两次调用计为连续，换成另一条受跟踪调用则把计数重置为 1。链保存在 `WeakMap<Agent, Chain>` 中。

- **不受跟踪的调用对链透明。** 被 `include`／`exclude` 排除的调用既不递增也不重置计数器，因此 `grep X → todo_write → grep X` 在 `todo_write` 被排除时仍算作连续两次 `grep X`——穿插进循环的记录类工具不能掩盖循环。
- **被拒绝的调用也计数。** 检测位于 `tools/post-execute`，被 `tools/pre-execute` 监听器拒绝的调用同样会经过它；模型反复尝试被拒绝的调用，恰恰是需要打破的循环。
- **忽略没有 agent 的调用。** 直接调用 `ctx.tools.execute()` 的调用方没有需要提醒的模型，也没有可作为键的活跃 agent 对象。
- **按 agent 分键，用户提示词时重置。** 一个 agent 的重复绝不会触发另一个 agent 的提醒；用户提示词（`agent/pre-step`）会删除提交该提示词的 agent 链，对象生命周期限制弱引用条目的寿命，无需 dispose（资源释放）监听器。
- **仅驻留内存。** 从持久化恢复的会话以全新链开始——guard 是启发式提醒，而非记录在案的不变量，因此恢复后的提醒延后是可接受的代价。

### 检测：重复结果

在 `resultThresholds` 非空之前，相同结果检测处于关闭状态。对每个匹配 `resultInclude` 的被跟踪工具调用，guard 把 `(工具名, 结果内容块)` 记入该 agent 最近 `resultWindow` 次此类调用的历史，并统计相等的条目。以完整内容块为键可以区分两张不同的图片；提醒引用的是它们的文本渲染，非文本块记为 `[<type> content]`。每个阈值对每个不同的结果只提醒一次：旧调用离开窗口时计数可能回落、随后再次回升，第二次越过阈值不会再提醒。当重复链在同一次调用上也触发时，只送出精确重复提醒，因为它列出了重复的参数。用户提示会同时清空历史与链。

### 提醒传递

提醒随 post-execute 决策的 `additionalContexts`（来源为 `{kind: 'plugin', plugin: 'repeat-tool-reminder', form: 'notice', summary: '<tool> × <count>'}`，相同结果提醒为 `'<tool> same result × <count>'`；非空的 `reminderSuffix` 在空一行后跟在文本之后）传递，绝不替换 `content`：用于审计的 `tool/result` 事件仍保留工具自己的输出。循环会缓冲这段上下文，并在该步骤的工具结果之后作为注入的 `user/message` 追加，会话将其渲染为普通的合成用户消息——模型可见、带有来源归属，且无需新会话事件即可从会话日志重建。guard 始终通过 `next()` 委派，并把提醒放在下游决策的上下文数组之前，因此两种决策变体（包括被阻止的调用）都会收到提醒，同时每个条目保留自己的来源与元数据。

### 源码地图

| 文件 | 职责 |
|---|---|
| [`src/index.ts`](src/index.ts) | 插件入口：`Config` schema、快速失败校验、链监听器 |
| — | 不发布运行时不变式伴生入口；链私有于一个 post-execute 监听器。 |

</details>

-----

<a id="further-exploration"></a>
## 进一步探索

当包级约定不够用时阅读以下页面。它们从工具 waterfall 逐步进入穷尽式配置与 guard 组映射。

- [工具子系统参考](../../../docs/subsystems/tools.zh.md)——本 guard 消费的 `tools/execute` waterfall、`additionalContexts` 与决策形态。
- [生成配置目录](../../../docs/config-catalog.zh.md#deepseek-aidsh-repeat-tool-reminder)——每个受支持配置字段及其源声明。
- [guard 组映射](../README.zh.md)——同组的 guard 包与循环卫生家族。

-----

<a id="model-experience"></a>
## 模型体验

### 首个阈值的上下文消息

#### 模型看到什么

达到第一个配置的连续重复阈值时，对应 agent 会收到下面的提醒。不会添加工具 schema 或正常调用文本。

##### 首个阈值提醒

```markdown
You are repeating the exact same tool call with identical arguments. Carefully analyze the previous result before calling again: if the task is not complete, try a different approach or different arguments instead of repeating the call.
```

#### Token 影响

达到阈值前为零 token。提醒会作为该 agent 的历史记录保留。

#### KV Cache 影响

仅追加；新出现的内容位于可复用请求前缀之后，不会使现有 KV Cache 条目失效。

### 后续阈值的上下文消息

#### 模型看到什么

达到后续阈值时，agent 会收到下面的详细提醒模板。受上限约束的参数预览严格以 `… (+<omitted> more chars)` 结尾。

##### 后续阈值提醒

```markdown
Repeated tool call detected:
- tool: <toolName>
- consecutive_calls: <count>
- arguments: <canonicalArguments>
The repeated calls are not making progress. Do not call this tool with these exact arguments again. Inspect the latest result and choose a different action, different arguments, or finish the task if enough evidence has been gathered.
```

#### Token 影响

每条提醒都会作为历史记录保留；`argumentsPreviewChars` 限制随数据变化的参数文本长度，而各 agent 仍使用独立计数器。

#### KV Cache 影响

仅追加；新出现的内容位于可复用请求前缀之后，不会使现有 KV Cache 条目失效。

### 相同结果的上下文消息

#### 模型看到什么

开启相同结果检测后，`resultThresholds` 的第一个值送出下方的首条提醒，后续的值送出详细模板。被截断的结果预览恰好以 `… (+<omitted> more chars)` 结尾。

##### 首条相同结果提醒

```markdown
Your recent calls to this tool keep returning the same result. Running more variations of the same check will not change it. Before calling again, work out why the result differs from what you expected: re-check your assumptions, including any expectation stated in the task, which may itself be wrong. Then change approach, or report the discrepancy and finish.
```

##### 后续相同结果提醒

```markdown
Repeated tool result detected:
- tool: <toolName>
- same_result_count: <count> of the last <window> calls
- result: <result>
These calls are not making progress: the result is not going to change. Do not run another variation of this check. State what you expected and why the observed result differs, verify that expectation itself, then take a different action or finish the task and report the discrepancy.
```

#### Token 影响

检测关闭或未达到第一个阈值时为零 token。每条提醒都保留在历史中；`resultPreviewChars` 限制其中依赖数据的结果文本。

#### KV Cache 影响

与重复链提醒一样仅追加。若 `reminderSuffix` 会被聊天模板解释（例如推理强度标记），它可能改变提醒之前渲染的文本，从而使缓存的前缀失效一次。

## 已知限制与延期工作

<a id="known-limitations-and-deferred-work"></a>


这些限制说明 guard 何时不合适。它们是当前包约束，不是任务积压。

- **仅精确匹配检测**——规范化是深度键排序，因此近似变体（稍作修改的路径、值内多余的空白）会绕过链；在没有需求证据前，不采用模糊匹配。相同结果检测只在结果相等时才能捕获变化的调用，因此包含时间戳或临时路径的输出会绕过它。
- **相同的结果并不总是循环**——持续通过的测试、或一直没有输出的命令，也会收到相同结果提醒；可通过 `resultInclude` 与 `resultThresholds` 释放压力。
- **压缩（compaction）不会重置链**——跨越压缩检查点的链会继续计数。
- **仅提供建议**——尚未实现高阈值时升级为阻止形式，但 `PostToolDecision` 已支持阻止。
- **subagent 之间不共享链**——链始终按 agent 隔离；父 agent 与其 subagent 重复相同调用也绝不合并。
- **合理的幂等轮询超过阈值后仍会收到提醒**——可通过 `thresholds`／`exclude` 配置释放压力。
- **超过最高阈值后链不再提醒**——提醒只在精确达到所配置的次数时触发，超过后不会继续发送。

<a id="dev-note"></a>
### 开发备注

<details>
<summary>维护者的工作上下文——点击展开</summary>

本开发备注是维护者的工作上下文：开放问题与尚未决定的探索方向。它明确不具权威性——已交付的行为、限制与既定理由以上文、包代码和相关 Agent Note 为准。

[repeat-tool-guard Agent Note](../../../.agents/notes/archived/feature/2026-07-08-repeat-tool-guard.md) 以旧包名记录了原始设计与备选方案；[改名台账](../../../.agents/notes/implemented/architecture/2026-08-11-repository-naming-contract-and-rename-ledger.zh.md) 记录了改名为 `repeat-tool-reminder` 及其原因。

</details>
