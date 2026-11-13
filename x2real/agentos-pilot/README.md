# X2Real AgentOS 训练试验

这是将 [40 强赛构想](https://github.com/captainNemoCheng/seer-competition/tree/feature/40-team-proposal) 迁移到 X2Real 的第一轮组件实验，不是完整的通用机器人智能体，也尚无自变量平台分数。

保留原构想的职责划分：动作模型执行、状态预测模块预判、独立动作检查、有限回退、逐回合记忆和结果核验。当前只对白名单瓶子任务训练；不会把叉车演示的固定场景分支当成机械臂的实际推理能力。

## 当前实现和边界

| 组件 | 当前实现 | 尚未完成 |
| --- | --- | --- |
| 执行策略 | 官方 WALL-OSS-0.5，已微调动作专家，完成 64 步试验 | 闭环任务成功率、跨任务训练 |
| 学习型状态预测 | 三路 RGB 的冻结 ResNet18 特征、20 维机械臂状态和 32 帧动作，三个小网络预测 1.6 秒后的特征及状态 | 物体状态语义、抓取成功/碰撞标签、置信度校准 |
| 候选动作 | 对策略动作和保持姿态计算预测，记录诊断 | 经验证的收益/风险排序；目前预测不改变选出的动作 |
| 独立检查和回退 | 按官方仿真器规则截断预测夹爪到 [0,1]，再检查形状、有限数值、首行状态和范围；非法输出最多两次保持姿态，随后报错 | 碰撞检查、抓取失败后的实用恢复技能 |
| 记忆和验证 | 回合内最多 64 条事件；预测与 48 个仿真控制步之后的观测对比 | 可靠的任务完成判据、经过验证的跨任务失败记忆 |

`shadow_only` 表示预测只写诊断，不被当作成功概率，不授权开夹爪、移动到未知位置等动作。当前保持姿态回退只处理非法模型输出，**不能称为已实现的空抓恢复**。本目录与相邻 `bottle-recovery` 原型互相独立。

## 数据和可复现性

- HF 数据：`x-square-robot/X2Real-Dataset`，revision `5a741385f9f196073cfe163b0f25033025e383a6`，desktop-ee。
- 任务：`Pick up the bottle and place it on the plate, then return the arm home.`
- 300 段轨迹，56,511 帧，20 Hz。保留三路 RGB、20 维双臂末端状态/动作和归一化夹爪值。
- 转换为 LeRobot v3；所有动作窗口限制在本回合内，尾部重复最后一个动作。
- 固定种子 42 先打乱整段轨迹。动作策略用 270 段训练、30 段留出；预测模块用 240/30/30 段作训练/调参/测试。不能把预训练基础模型是否见过这些数据当成已知事实。
- 归一化只使用相应训练集。`results/split.json` 记录原始 episode ID。
- WALL-X revision `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`，Policy Space revision `6825ed0526692803c5bf9ac450780c27dcfb9d7b`。
- WALL-OSS-0.5 权重 revision `b7dc23a82dbf70a65859c9fdd596703234597028`。

## 已完成的状态预测实验

2026-11-13 在 A40 上训练，测试集为 30 段独立轨迹中的 313 个转换。原始结果在 `results/metrics.json`。

| 方法 | 视觉特征标准化 MSE ↓ | 末端位置 RMSE ↓ | 夹爪 MAE ↓ |
| --- | ---: | ---: | ---: |
| 假设场景和状态不变 | 1.13691 | 0.06664 m | 0.25723 |
| 场景不变、机械臂到达最后指令 | 1.13691 | **0.00209 m** | **0.00043** |
| 学习型预测器 | **0.68490** | 0.01278 m | 0.05073 |

视觉预测比不变基线改善约 39.8%；机械臂状态预测输给最后指令基线。因此当前不能用它代替已有运动学关系或根据它判断抓取是否成功。这些数字不是自变量任务分数，亦不是抓取成功率。

## 已完成的动作微调

A40 上仅训练动作专家的 399,860,736 个参数，BF16、单卡、batch 1、梯度累积 2，共 64 个优化步骤。在同一批 32 个固定留出帧上，验证损失从 0.1088739 降至 0.0244520。见 `results/action-training.json`。这只是短程试验，不等于瓶子任务成功率，更不能说明能完成全部平台任务。

导出后补回 `data.max_length=2048`，避免推理时截断图像 token；同时修正上游恢复训练所用的零基步骤计数。以上只修改导出配置和计数，不修改训练后的张量。

## 运行

默认目录是容器内 `/workspace/x2real-training`。先把本目录的脚本放到该位置，GPU 必须支持 BF16，并有足够磁盘保存基础权重及训练检查点。

```bash
bash setup_training.sh
.venv/bin/python prepare_training.py
.venv/bin/python latent_world_model.py
.venv/bin/python evaluate_world.py
PYTHONPATH=wall-x:policy-space/src:policy-space/packages/protocol/src .venv/bin/python check_pilot.py
PYTHONPATH=wall-x .venv/bin/torchrun --standalone --nproc_per_node=1 train_pilot.py
```

`train_pilot.py` 为上游加载器补充真正按整段轨迹划分的验证集，并保证达到步数上限后保存最终检查点。它在同一批固定留出帧上记录训练前后的损失；该损失仍不能代替闭环评估。

接入服务需要先安装本地 `policy-space`，设置 `WALLX_ROOT`、`WALLX_BASE_DEPLOY`、`WALLX_CHECKPOINT_PATH` 和 `AGENTOS_WORLD_CHECKPOINT`，并为微调权重设置匹配的 `WALLX_NORM_KEY`。服务使用 `exclusive`，不支持同一模型实例并发维护多个回合。

```bash
bash run_policy.sh check
bash run_policy.sh serve
```

接口检查必须返回真实策略动作，保持姿态回退不能算作模型推理成功。夹爪截断与 ManaEnv 的 `manaenv/policy/gripper_convention.py::to_sim_units` 一致；不改变末端位置、旋转或首行观测。真实平台评估还需要可达的服务地址、账户登录和正式任务运行记录。

`secure_serve.py` 可将相同服务绑定到本机 8002 端口，并要求随机生成的私有 URL 路径；路径保存在权限受限的 `.service/access-path`。配合外部 TLS 隧道使用时，完整 WSS 地址视同访问凭证，不提交到 Git。临时隧道的地址会随进程重启而变化，不是永久部署。诊断事件追加到 `outputs/agentos-events.jsonl`。

## 服务实测

2026-11-13：7 项数据边界、坐标变换、夹爪截断、回退及回合隔离检查通过。官方适配器检查通过；完整模型在一帧真实留出观测上返回有限 `[33,14]` 动作，未触发保持姿态回退，推理约 3.02 秒。随后从本机通过 WSS 外网地址完成协议 v2 的初始化、观测、动作推理和结束回合，往返约 4.92 秒。见 `results/interface-check.json` 和 `results/external-check.json`。

**已登记自变量模型 `AgentOS-bottle-pilot-64`（模型 94），尚无完成的正式评测成绩。** 旧评测已取消，进度 0/850；2026-11-13 已将平台地址更新至新 GPU 的服务，接口显示正常，用户已发起新的评测。临时 WSS 服务依赖 Pod 和隧道进程持续运行，不能停止 Pod 后继续评测。

## 权重备份和恢复

`pack_changes.py` 与固定 revision 的公开基础权重逐张量比较，保存全部不同张量及完整形状/类型清单；不仅按可训练参数名筛选，因此也保留训练初始化时扩展或改变的冻结张量。843 个张量逐一重建后与完整训练检查点完全相等。该备份为无损权重备份，不含优化器状态；旧 Pod 的完整训练检查点和优化器目前不可访问；新 GPU 只恢复推理权重，不声称恢复优化器或继续训练。

`changed.safetensors.zst` 解压后，可用以下命令生成完整推理权重；需要约 18 GB 的 CPU 内存及输出磁盘空间，并把原检查点的配置、tokenizer 和归一化文件放到同一目录。

```bash
zstd -d changed.safetensors.zst
python restore_changes.py --base weights/model.safetensors \
  --changes changed.safetensors --manifest manifest.json \
  --output restored/model.safetensors
```

脚本校验基础权重的 SHA256，不覆盖已有输出。公开 Git 只放代码和实验记录，模型权重与私有服务地址不提交到仓库。

## 后续验收

按原构想继续实现任务相关的语义状态核验、观测驱动的技能编排、失败数据采集、有限恢复与失败记忆，再比较基础策略、微调策略、恢复模块和预测模块的闭环消融结果。当前成果仅覆盖上述已验证组件，不承诺所有题目或所有硬件。

## RTX PRO 6000 迁移验证

2026-11-13 将上述 64 步检查点迁移至 RTX PRO 6000 Blackwell 96 GB。固定基础权重与无损差异备份恢复后，843 个张量逐一相等；没有新增训练步骤。Python 3.12、PyTorch 2.8.0+cu128、官方 Flash Attention 2.8.3.post2 二进制包；原生 sm120 CUDA 扩展与两种 head dimension 的 Flash Attention 检查通过。环境安装见 `setup_inference.sh`，备份散列和逐张量复核见 `restore_and_check.sh`。

恢复前需自行放置 `backup/{changed.safetensors.zst,SHA256SUMS,manifest.json}`、原检查点配置及 tokenizer、`TRAIN_COMPLETE.json`、预测模块权重和留出观测；这些大文件和服务访问凭证不提交到 Git。默认容器目录同上。完整恢复约需 43 GB 磁盘，须额外留出运行空间。

真实留出观测推理返回 `[33,14]`，有限数值、零回退，耗时 2.28 秒。公网协议 v2 完整初始化→观测→推理→结束回合通过，往返 2.85 秒。检查记录见 `results/pro6000/`。这是一帧观测的服务验证，不能当作机器人任务成功率；跨 GPU 数值一致性也未据此证明。
