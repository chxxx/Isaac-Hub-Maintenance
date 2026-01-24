# ==============================================================================
# Full IsaacLab M20+Piper Cleanup Task
# ==============================================================================

import debugpy
try:
    debugpy.listen(("localhost", 5678))
    print(">>> Waiting for VS Code Debugger...")
except Exception:
    pass

# ==============================================================================
# Full IsaacLab M20+Piper Cleanup Task (Fixed & Complete)
# ==============================================================================

import argparse
from isaaclab.app import AppLauncher
import torch

# 1. 启动仿真 App (必须在导入其他 Isaac 模块前执行)
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. 导入 Isaac Lab 模块
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# ------------------------------------------------------------------------------
# Scene Definition
# ------------------------------------------------------------------------------
@configclass
class CleanupSceneCfg(InteractiveSceneCfg):
    # ==========================================================
    # [关键修复] 必须指定环境数量和间距
    # ==========================================================
    num_envs = 1
    env_spacing = 2.0

    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )
    
    # --- 机器人配置 ---
    
    # M20 底盘
    m20 = ArticulationCfg(
        prim_path="/World/M20",
        spawn=sim_utils.UsdFileCfg(
            # 请确保此路径在你的硬盘上真实存在
            usd_path="D:/study/usd/deep_robotics_model/M20/M20_usd/M20.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        actuators={
            "base_drive": ImplicitActuatorCfg(
                joint_names_expr=[".*"], 
                stiffness=20.0,
                damping=2.0,
            )
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # Piper 机械臂
    piper = ArticulationCfg(
        prim_path="/World/Piper",
        spawn=sim_utils.UsdFileCfg(
            # 请确保此路径在你的硬盘上真实存在
            usd_path="D:/study/usd/Piper.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=50.0,
                damping=5.0,
            )
        },
        init_state=ArticulationCfg.InitialStateCfg(
            # 根据 M20 高度调整 Z 值，防止机械臂卡在底盘里
            pos=(0.0, 0.0, 0.6), 
            joint_pos={f"joint{i}":0.0 for i in range(1,7)},
        ),
    )

    # Cloth rag
    rag = RigidObjectCfg(
        prim_path="/World/Rag",
        spawn=sim_utils.CuboidCfg(
            size=(0.4,0.4,0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0,2.0,0.1)),
    )
    
    # Trash Bin
    bin = AssetBaseCfg(
        prim_path="/World/TrashBin",
        spawn=sim_utils.CuboidCfg(
            size=(0.6,0.6,0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,1.0,0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0,-2.0,0.25)),
    )

# ------------------------------------------------------------------------------
# Run Loop
# ------------------------------------------------------------------------------
def run_simulator(sim, scene:InteractiveScene):
    print("Simulation Running...")
    sim_dt = sim.get_physics_dt()
    state = 0

    while simulation_app.is_running():
        sim.step()
        
        # 只有当 scene 有 update 方法时才调用
        if hasattr(scene, "update"):
            scene.update(dt=sim_dt)

        # 获取数据 (注意：必须要加上 env_ids 或索引 [0])
        m20_pos = scene["m20"].data.root_pos_w[0]
        rag_pos = scene["rag"].data.root_pos_w[0]
        
        dist = torch.norm(m20_pos - rag_pos)
        
        # 使用 end="\r" 让打印不刷屏
        print(f"Dist: {dist:.2f} | State: {state}", end="\r") 

        if state==0 and dist<1.0:
            state = 1
            print("\n>>> State 1: Approached!")
        if state==1:
            # 重置 Piper 关节位置
            scene["Piper"].set_joint_positions(torch.zeros((1,6),device=sim.device))
            state=2
        if state==2 and dist<0.5:
            pass

# ------------------------------------------------------------------------------
# Main Entry
# ------------------------------------------------------------------------------
def main():
    # 初始化仿真配置
    cfg = SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(cfg)
    
    # 初始化场景 (现在包含了 num_envs 参数，不会报错了)
    scene_cfg = CleanupSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    run_simulator(sim,scene)

if __name__=="__main__":
    main()
    simulation_app.close()