# tau2-bench 评测交接（2026-10-20）

本包只实现 Qwen3-30B-A3B-Instruct-2507 的 tau2 文本评测；没有训练代码、作业提交或远端操作。现有 selfplay 和旧交接文件未修改。当前没有真实模型轨迹或 benchmark 分数，不能把模拟传输单元测试当作评测成功。

## 来源和正式协议

SPADE 官方代码固定 ebd40ec872fc5630cac299cb5e38e7c89743bef5；Sierra tau2 固定 a2c024725189473d2d7cea3a5cfdbcc67478e41f。两个子模块保持干净，未修改上游代码。tau2 的版本按论文公开时间选择，不是已核验的作者历史环境锁定版本。

直接沿用官方 tau2 task loader、run_tasks、SPADE config loader、每领域聚合函数及 Tau2EvalResult 指标。retail/base=114、telecom/base=114、airline/base=50，每任务4次，共1112条。并发48，最多200步；agent temperature=0/top_p=.8/top_k=20/max_tokens=16384；user temperature=0/max_tokens=2048；seed=300。报告严格分母 Pass@1，不计算四次取最好。缺少、重复、异常奖励、基础设施错误或聚合分母异常均退出且 complete=false。

GPU 服务入口使用官方 SGLang 形式，TP8/DP1、mem_fraction_static=.7、qwen tool parser。不会全局终止进程、安装依赖或操作 Ray。客户端与 GPU 服务可使用不同的现有 Python 环境，客户端需要 Python>=3.12。

## 显式差异与身份边界

用户模拟器和 NL-assertion 裁判按要求改用 deepseek/deepseek-flash，显式关闭 thinking；裁判要求 JSON。原始 GPT 用户/裁判分数与本设置不可直接声称等价。服务端返回的 model 字段记录在结果中，不能独立证明真实权重。API 别名可能更新。

run_tasks 的基础设施重采样显式设为0（原生默认3），API调用也不重试。agent显式禁用thinking，以匹配 Instruct-2507 路径；通过 extra_body 保留 top_k，避免被 LiteLLM drop_params 丢弃。实际加载模型名替代上游固定 sglang-local 名称。跳过上游训练式超时/空结果容错，完整性检查通过才标记有效。

模型架构、模板、分片索引、来源commit及任务清单会检查；执行时计算本地模型文件哈希并写manifest。模型元信息和远端/v1/models别名无法证明远端加载同一权重，执行者还须核对服务启动日志及其manifest。客户端不下载权重或修改环境。

## 边界与密钥

读取仓库根AGENTS.md。服务器执行只允许确认的ztwang目录内路径，缓存独立放在本次run下。默认仅预览，需要显式 --execute。使用全新输出目录，拒绝覆盖旧结果。子模块只能按固定gitlink初始化，不需要递归初始化Slime/Tinker。

真实密钥未写入本包、命令、环境或测试；执行者运行时交互输入DEEPSEEK_API_KEY。传给tau2持久化配置的user/judge参数不含密钥。API异常进入原生checkpoint前脱敏；客户端输出和失败报告脱敏；GPU子进程剔除API密钥环境。原始对话仍含业务轨迹信息，默认保留本地，交接者发布前自行审查，不把run目录自动提交Git。

## GPT + DeepSeek 审查记录

DeepSeek Flash完成provider草案、backend/config草案及独立只读审查；GPT逐项查源码、采用修正并运行测试。两个过大的初始草案因输出限制失败，未采用；provider任务传输失败但有完整草案，GPT检查完整diff后手工采用，未将失败任务当作验证通过。集成入口和路径身份检查由GPT完成。

独立复审提出的导入路径问题经核对不成立：BASE/upstream内部就有spade包，文档使用直接脚本入口（不承诺python -m spade.run_tau2）。NL judge读取模块全局值，未以默认参数捕获；固定tau2源码仅llm_utils导入completion，因此本路径的拦截有效。仍增加实际导入来源检查、字节码禁止、git只读检查、终止原因/奖励/聚合计数审查、API异常脱敏。固定官方聚合方法直接调用，不再存在猜测式fallback。

## 已完成的验证

- 新增本地测试：22 passed，包含模拟三角色传输、真实官方聚合函数、形式/冒烟隔离、全任务分母、损坏结果拒绝、凭据异常脱敏、目录边界、来源任务清单、TP8配置及默认预览。
- 固定SPADE官方相关测试：35 passed。
- CLI帮助入口及git diff --check通过。测试使用Python3.12，UTF-8和禁用字节码，未安装额外依赖。
- 未调用DeepSeek API、Qwen服务，未访问集群、启动GPU、执行冒烟或完整benchmark。因此不保证服务器依赖/显存/吞吐或真实分数已验证。

## 下一位执行者

按spade/README.md准备已有环境和固定子模块；先检查预览，再在已授权的八卡资源启动服务。先跑 --smoke 的三个领域各一条，人工核对工具调用、用户/裁判路由、终止理由、raw轨迹、服务权重manifest；确认后用全新目录运行完整1112条。任一基础设施错误停止并交接，不私自换模型或修改协议。收集manifest.json、endpoint_models.json、results.json、status/failure及三个领域raw.json；结果只有scope=formal且complete=true并包含全部领域和1112条时才进入正式表。
