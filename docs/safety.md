## 1. 先把真实 Franka 的硬限制写进仿真

Franka FCI 文档明确要求关节空间命令满足位置、速度、加速度、jerk 限制；必要条件违规会触发错误并停止运动。文档还给出了 torque 和 torque-rate 限制，torque control 下 torque-rate 是必要条件。

旧 Panda / Franka Emika Robot，也就是 FCI 文档里的 FER，常用关节限制是：

| 项 | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_min rad | -2.8973 | -1.7628 | -2.8973 | -3.0718 | -2.8973 | -0.0175 | -2.8973 |
| q_max rad | 2.8973 | 1.7628 | 2.8973 | -0.0698 | 2.8973 | 3.7525 | 2.8973 |
| dq_max rad/s | 2.175 | 2.175 | 2.175 | 2.175 | 2.61 | 2.61 | 2.61 |
| ddq_max rad/s² | 15 | 7.5 | 10 | 12.5 | 15 | 20 | 20 |
| jerk_max rad/s³ | 7500 | 3750 | 5000 | 6250 | 7500 | 10000 | 10000 |
| tau_max Nm | 87 | 87 | 87 | 87 | 12 | 12 | 12 |

这些数值来自 Franka 官方 FCI robot limits；FR3 的 limits 不完全一样，例如 FR3 的基础 qdot_max 是 J1–J4 为 2.62 rad/s、J5/J7 为 5.26 rad/s、J6 为 4.18 rad/s，且 FR3 文档还给出了“随关节位置变化的速度限制”，靠近 joint limit 时会更严格。

在 Isaac Lab 里，至少要给 articulation actuator 写：

```
fromisaaclab.actuatorsimportImplicitActuatorCfg

# Panda / FER example. 关节名按你的 USD/URDF 调整。
PANDA_ACTUATORS= {
"panda_arm":ImplicitActuatorCfg(
joint_names_expr=["panda_joint.*"],

# PhysX solver 层硬限制，不只是 reward。
effort_limit_sim={
"panda_joint[1-4]":87.0,
"panda_joint[5-7]":12.0,
        },
velocity_limit_sim={
"panda_joint[1-4]":2.175,
"panda_joint[5-7]":2.61,
        },

# 这些不是厂家给的唯一真值，需要用真实机器人 step response / log 来调。
stiffness={
"panda_joint[1-4]":300.0,
"panda_joint[5-7]":80.0,
        },
damping={
"panda_joint[1-4]":30.0,
"panda_joint[5-7]":8.0,
        },

# 可选：加一点关节摩擦/armature，避免“无损无惯量”的超人表现。
# 注意 Isaac Sim 5.0+ 中 friction 是力/力矩意义，旧版本含义不同。
armature=0.01,
    )
}
```

Isaac Lab 里要优先用 `effort_limit_sim` 和 `velocity_limit_sim` 写进物理 solver；`effort_limit_sim` 会限制 PhysX 计算出的关节 effort，`velocity_limit_sim` 会限制关节速度，并且速度限制能否达到还取决于 effort limit 是否足够。

## 2. 不要让 policy 直接发“大跳变 target”

很多 Franka RL 环境看起来“无敌”，是因为 policy 每步可以把 target position 瞬间跳很远，然后 Isaac 的 PD drive 用很大 torque 去追。真实 FCI 不会让这种轨迹通过；它会检查位置、速度、加速度、jerk、torque-rate。

你可以在 action 到 actuator 之间加一个 safety filter：

```
importtorch

Q_MIN=torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
Q_MAX=torch.tensor([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
DQ_MAX=torch.tensor([2.175,2.175,2.175,2.175,2.61,2.61,2.61])
DDQ_MAX=torch.tensor([15.0,7.5,10.0,12.5,15.0,20.0,20.0])
JERK_MAX=torch.tensor([7500.0,3750.0,5000.0,6250.0,7500.0,10000.0,10000.0])

deffranka_action_filter(
q_target_raw:torch.Tensor,
q_target_prev:torch.Tensor,
qd_target_prev:torch.Tensor,
qdd_target_prev:torch.Tensor,
dt:float,
margin:float=0.03,
):
"""
    q_target_raw: policy 输出映射后的目标关节角, shape = [num_envs, 7]
    返回 rate-limited q_target, qd_target, qdd_target
    """

device=q_target_raw.device
q_min=Q_MIN.to(device)+margin
q_max=Q_MAX.to(device)-margin
dq_max=DQ_MAX.to(device)
ddq_max=DDQ_MAX.to(device)
jerk_max=JERK_MAX.to(device)

# 1) position hard clip
q_target=torch.clamp(q_target_raw,q_min,q_max)

# 2) velocity limit on target motion
dq_des= (q_target-q_target_prev)/dt
dq_des=torch.clamp(dq_des,-dq_max,dq_max)

# 3) acceleration limit
ddq_des= (dq_des-qd_target_prev)/dt
ddq_des=torch.clamp(ddq_des,-ddq_max,ddq_max)
dq_des=qd_target_prev+ddq_des*dt

# 4) jerk limit
jerk_des= (ddq_des-qdd_target_prev)/dt
jerk_des=torch.clamp(jerk_des,-jerk_max,jerk_max)
ddq_des=qdd_target_prev+jerk_des*dt
dq_des=qd_target_prev+ddq_des*dt

# 5) integrate back to feasible target
q_target=q_target_prev+dq_des*dt
q_target=torch.clamp(q_target,q_min,q_max)

returnq_target,dq_des,ddq_des
```

如果你用的是 Isaac Lab Manager-based action，可以把 action space 先缩小：`JointPositionActionCfg` 支持 `scale`、`offset`、`clip`；`JointPositionToLimitsActionCfg` 还能把 `[-1, 1]` action 重映射到关节上下限。

```
importisaaclab.envs.mdpasmdp

classActionsCfg:
arm_action=mdp.JointPositionToLimitsActionCfg(
asset_name="robot",
joint_names=["panda_joint.*"],
scale=0.7,# 不要一开始就给满行程
rescale_to_limits=True,
clip={"panda_joint.*": (-1.0,1.0)},
    )
```

更像真机的设置是：policy 不直接控制绝对关节角，而是输出**小的 delta q** 或末端 twist，然后通过安全过滤器 / IK / trajectory generator 变成关节目标。

## 3. 用 explicit actuator 模拟“电机没那么理想”

Isaac Lab 有两类 actuator：implicit actuator 由仿真内部 PD 处理；explicit actuator 会自己算 torque 并做 clipping。Ideal PD actuator 的公式就是 `tau = kp(q_des-q)+kd(dq_des-dq)+tau_ff`，然后按最大 torque 裁剪；DCMotor actuator 进一步用 torque-speed 曲线做速度相关饱和。

如果你的 policy 未来要上真机，我更建议训练时使用 explicit actuator，至少让 torque 先被裁掉：

```
fromisaaclab.actuatorsimportIdealPDActuatorCfg,DelayedPDActuatorCfg

PANDA_EXPLICIT_ACTUATORS= {
"panda_arm":DelayedPDActuatorCfg(
joint_names_expr=["panda_joint.*"],

# actuator model 内部 torque clipping
effort_limit={
"panda_joint[1-4]":87.0,
"panda_joint[5-7]":12.0,
        },

# solver 层再加一道保险
effort_limit_sim={
"panda_joint[1-4]":87.0,
"panda_joint[5-7]":12.0,
        },
velocity_limit_sim={
"panda_joint[1-4]":2.175,
"panda_joint[5-7]":2.61,
        },

stiffness={
"panda_joint[1-4]":250.0,
"panda_joint[5-7]":60.0,
        },
damping={
"panda_joint[1-4]":25.0,
"panda_joint[5-7]":6.0,
        },

# 1~3 个 physics step 的命令延迟，比“瞬时响应”更像真实链路。
min_delay=1,
max_delay=3,
    )
}
```

`DelayedPDActuator` 会把 actuator command 放进 buffer，然后延迟若干 physics step 再应用；这个对 sim-to-real 很有用，因为真实系统里有通信、状态估计、控制栈延迟。

## 4. 把“真机会停机”的情况做成 termination

真实 Franka 不是撞了还继续训，它会 contact/collision flag、stop、error recovery。libfranka 的 `setCollisionBehavior` 支持设置 joint torque 和 Cartesian force 的 contact/collision 阈值；超过上阈值会被注册为 collision，并导致机器人停止。

所以在 RL 里建议这样做：
defsafety_termination(q,qd,tau,ee_force,contact_force):
joint_limit_violation=torch.any((q<Q_MIN.to(q.device))| (q>Q_MAX.to(q.device)),dim=-1)
vel_violation=torch.any(torch.abs(qd)>DQ_MAX.to(q.device)*1.05,dim=-1)
torque_violation=torch.any(torch.abs(tau)>torch.tensor([87,87,87,87,12,12,12],device=q.device),dim=-1)

# 这些阈值需要按你的真实 Franka Desk/libfranka collision behavior 配置来设。
contact_violation=contact_force>40.0
ee_force_violation=torch.linalg.norm(ee_force[:, :3],dim=-1)>60.0

returnjoint_limit_violation|vel_violation|torque_violation|contact_violation|ee_force_violation