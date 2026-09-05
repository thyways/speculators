# Qwen3.6-35B-A3B 双节点在线训练

平台申请 **2 节点、每节点 8 卡**，每个节点执行一次相同的入口脚本。
脚本使用 `PET_NNODES=2`、`PET_NODE_RANK=0/1`、`PET_NPROC_PER_NODE=8`；
训练 rendezvous 优先使用 `MASTER_ADDR/MASTER_PORT`，缺失时使用 `PET_MASTER_ADDR/PET_MASTER_PORT`。

| 节点 | 推理卡 | 训练卡 | 训练 global rank |
| --- | --- | --- | --- |
| node 0 | 可见卡前 2 张，TP=1 / DP=2 | 可见卡后 6 张 | 0–5 |
| node 1 | 可见卡前 2 张，TP=1 / DP=2 | 可见卡后 6 张 | 6–11 |

每个训练进程从本机 `127.0.0.1:8000/v1` 获取 hidden states。训练器按 12 个 rank
分片数据、同步初始化模型，并在训练时通过 DDP all-reduce 同步梯度。
两台推理服务各自独立；hidden-state 路径和进程清理也按节点隔离。

## 启动

以下三个配方每次选择一个，在平台的两个节点运行同一条命令；直接运行 Bash，
脚本内部已启动 `torchrun --nnodes 2 --nproc_per_node 6`。

```bash
bash /inspire/sfs/project/inf-multimodal/public/wumengke/speculators/examples/train/nnode/domino_qwen3_6_35b_a3b_perfectblend_online_2node.sh
```

```bash
bash /inspire/sfs/project/inf-multimodal/public/wumengke/speculators/examples/train/nnode/dspark_qwen3_6_35b_a3b_perfectblend_online_2node.sh
```

```bash
bash /inspire/sfs/project/inf-multimodal/public/wumengke/speculators/examples/train/nnode/dflash2_qwen3_6_35b_a3b_perfectblend_online_2node.sh
```

需要 Bash 5.1+。`CUDA_VISIBLE_DEVICES` 的顺序决定 2+6 分卡；未设置时使用 `0..7`。
平台注入的外层 `WORLD_SIZE/RANK` 由子进程启动器重新确定：训练为 `WORLD_SIZE=12`，
本机推理是独立的 2 个 DP 副本。平台的 NCCL/GLOO 参数原样继承；沿用同集群
已有多机配方的 `NCCL_CROSS_NIC=0` 缺省值，适配后 6 卡训练的 RoCE rail 拓扑。

## 配方与产物

沿用对应单机脚本的模型参数：3 epochs、每 rank 8192 token 预算、block size 7、
2048 anchors、5 层 draft、target layers `1 10 19 28 37`。三个脚本各自独立，配置集中在文件顶部。
所有 draft 层使用 full attention，显式设置 `--full-attention-indices 0 1 2 3 4`。
学习率为 AdamW 组 `1e-4`、
Muon 组 `2e-4`，1% warmup 后 cosine 衰减。双机每步全局 token 预算为 `12 × 8192 = 98304`。

默认模型和数据与单机配方一致。checkpoint 保存到
`$WS/model_weights/<方法>_qwen3_6-35b-a3b-perfectblend_2node/runs/<RUN_NAME>`，
`RUN_NAME` 默认 `<方法>-base-fullattn-2node-<MASTER_ADDR>_<MASTER_PORT>`，两节点一致。
`WS` 默认工作区绝对路径，可通过 `ROOT` 覆盖。
可用 `RUN_DIR` 指定输出根目录，或用 `CHECKPOINT_DIR` 直接指定保存目录；
**两节点必须设置成同一路径**。新名称包含 `fullattn`，不会默认续训旧的 SWA 产物。
有 checkpoint 时沿用训练器的恢复机制，rank 0 保存模型及 `run.yaml/train_command.txt`。

每节点的正式日志在
`$REPO/logs/train/qwen3_6_35b_a3b/<方法>_2node/<RUN_NAME>/`，
分别为 `train_node0.log` / `train_node1.log`、`vllm_node0.log` / `vllm_node1.log`。
临时 hidden states 使用 `/tmp/<方法>_node<NODE_RANK>.XXXXXX`，由 `mktemp` 创建，退出时只清理本次目录。

`WANDB_MODE` 默认 `online`，优先使用环境里的 `WANDB_API_KEY`，否则读取
`$WS/.secrets/wandb_key`。离线运行可设 `WANDB_MODE=offline`。
DFlash2 保留完整词表要求，数据目录中存在 `d2t.npy/t2d.npy` 时会在启动前报错。
