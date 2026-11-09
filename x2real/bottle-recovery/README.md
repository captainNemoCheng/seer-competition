# X2Real 双臂空抓恢复原型

这套程序接在自变量的 WALL-OSS-0.5 策略服务前面，用于 ArtiXon Arm-6A 双臂桌面评测。它沿用官方模型、权重和动作格式。它只对「把瓶子放到盘子上」启动复查，其它指令由 WALL-OSS-0.5 处理。

每次规划动作前后，程序可以收到正面相机画面和机械臂状态。如果一侧原来张开的夹爪计划闭合，就在画面里记下唯一匹配的瓶子位置。收到后续画面时，如果同一侧仍闭合、末端高度比计划中的闭合点高至少 3.5 厘米，而且瓶子外观依然匹配原位，就记为**疑似空抓**。启用恢复后，程序会用一段动作保持当前姿态并松开该侧夹爪，再把新的观测交给原模型规划。每轮任务最多触发一次恢复。遇到重复目标、画面不匹配或缺少观测时沿用原模型。

瓶子模板受视角、尺寸、遮挡和外观变化影响。初版使用固定阈值和保守匹配，适合先观察触发情况。疑似空抓是诊断信号，任务完成情况以评测报告和回放为准。

## 部署到 GPU 服务器

1. 在 Linux NVIDIA GPU 服务器按 [Wall-X 安装说明](https://github.com/X-Square-Robot/wall-x)准备推理环境并下载 [WALL-OSS-0.5 权重](https://huggingface.co/x-square-robot/wall-oss-0.5)。在同一环境安装 [Policy Space](https://github.com/X-Square-Robot/policy-space)，并满足[官方 Arm-6A 适配器](https://github.com/X-Square-Robot/policy-space/tree/main/policies/wallx/wall_oss_05_artixon_arm_6a_eef)的依赖。原型本地用 Policy Space 提交 `6825ed0526692803c5bf9ac450780c27dcfb9d7b` 验证接口。官方适配说明给出的 Wall-X 验证版本是 `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`。
2. 解压此原型，在其目录下运行以下命令。把路径换成你的实际目录。

   ```bash
   export WALLX_ROOT=/path/to/wall-x
   export WALLX_CHECKPOINT_PATH=/path/to/wall-oss-0.5
   export WALLX_BASE_DEPLOY=/path/to/policy-space/policies/wallx/wall_oss_05_artixon_arm_6a_eef/deploy.yaml
   export WALLX_BACKEND=standard
   export BOTTLE_MODE=baseline

   policy-space check deploy.yaml
   policy-space serve deploy.yaml --host 127.0.0.1 --port 8001
   ```

   基线模式直接返回官方模型的动作。原型用单会话模式保存一次任务内的前后观测，本地仿真使用 `--num-parallel-envs 1`。上传评测前，按[官方云评测指南](https://x2robot.com/en/docs/quantax1pro/v1/simulation/evaluation-platform-guide)配置可达地址、登记 Arm-6A 并完成平台接口检查。

## 准备复查和恢复模式

从**独立的开发场景**录一帧抓取前的正面相机画面。画面按接口分辨率 640×480 保存，贴着瓶身裁一张 PNG，尽量避开盘子、机械臂和大块背景。裁图长宽分别需要在 12–192 像素之间。同目录可以放不同外观的几张模板。文件必须来自你有权用于开发的场景。

```bash
export BOTTLE_TEMPLATES=/path/to/bottle-crops
export BOTTLE_MODE=shadow
policy-space check deploy.yaml
policy-space serve deploy.yaml --host 127.0.0.1 --port 8001
```

从日志里找 `bottle_recovery`，对照回放确认 `suspected_empty_grasp` 是否属实。确认抓起瓶子的画面不会被频繁误判后，重启服务并设 `BOTTLE_MODE=recover`。日志中 `release_and_replan` 表示策略执行过松夹恢复，`episode_end` 汇总本轮的疑似空抓和恢复次数。

## 同一场景比较

在安装了 Isaac Sim 和 X2Real 资产的机器上，从 X2Real 仓库根目录先用下列命令跑基线。若模型和仿真分开部署，把服务器地址改成仿真机可达的地址。

```bash
python manaenv/scripts/x2real_eval.py \
  --board artixon_arm6a_tabletop \
  --env artixon_arm6a_H_put_bottle_on_plate_id_env \
  --seed 42 --num-episodes 20 --num-parallel-envs 1 \
  --server-address 127.0.0.1 --server-port 8001 \
  --save-path /path/to/results/baseline
```

换成恢复模式，用相同的场景、种子、轮数和时间预算再跑一遍，只把保存路径换成 `/path/to/results/recover`。先比官方报告中的成功率和平均得分，再看恢复触发次数和回放中人工确认的空抓次数。开发场景用独立种子。跑通单题以后，另用「具亮计划」评测实际指定的题集测整个服务的覆盖情况。

## 已验证范围

- 本地用替身模型通过 Policy Space 的 `check`。接口返回 `float32[33, 14]`，包括官方适配器要求的首行状态和 32 行动作。
- 本地验证了单次恢复、同一步重试、跨任务清空状态、瓶子已移动、重复瓶子、夹爪仍开、抬升不足和超时等分支。用公开 [X2Real 数据集](https://huggingface.co/datasets/x-square-robot/X2Real-Dataset) 第 1193 条演示的两帧额外验证了模板对原位和移位画面的区分。
- 官方评测脚本的 `--dry-run` 解析到了指定的瓶子任务。实际 WALL-OSS-0.5 GPU 推理和 X2Real 闭环评测待运行，恢复增益未知。

本原型代码使用 Apache-2.0。第三方代码和权重按各自上游条款使用。
