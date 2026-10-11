# selfplay

selfplay 项目工作区。

## 目录约定

```
selfplay/
├── src/           源码
├── configs/       实验配置
├── scripts/       训练/评测入口脚本
├── logs/          运行日志（不入库）
└── checkpoints/   权重输出（不入库，建议软链到 ~/jhspoolers）
```

## 提交作业

直接用 `~/ztwang/tools/sbatch/` 下的模板，或在本目录放项目专属 sbatch：

```bash
cp ~/ztwang/tools/sbatch/gpu_a800x8.sbatch ./run_selfplay.sbatch
# 改 --job-name 和最后一行启动命令
sbatch ./run_selfplay.sbatch
```

## 注意

- 8 卡整机必须用 `i64m1tga800u8ka`
- checkpoint 写 `~/jhspoolers/selfplay/`，不要写家目录
- 日志路径用 `~/ztwang/logs/%x-%j.out`
