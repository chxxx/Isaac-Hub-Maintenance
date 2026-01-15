import debugpy
import os
import random
import numpy as np
from omni.isaac.kit import SimulationApp

# ==========================================
# 0. 调试器配置
# ==========================================
try:
    debugpy.listen(('localhost', 5678))
    print("【等待 VS Code 附加调试器...】请在 VS Code 按 F5")
    debugpy.wait_for_client() # 调试时取消注释此行
except Exception as e:
    print(f"调试器监听跳过: {e}")

# ==========================================
# 1. 启动 SimulationApp
# ==========================================
# 开启 anti_aliasing 以获得更好的视觉边缘
simulation_app = SimulationApp({"headless": False, "anti_aliasing": 1})

# ==========================================
# 2. 导入模块
# ==========================================
import omni.usd
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.world import World
from omni.isaac.core.prims import XFormPrim
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
import omni.kit.commands

# ==========================================
# 3. 场景构建函数
# ==========================================

def create_environment():
    print("Creating Environment (Solid Ground)...")
    # 视觉修正：实体地面，厚度 2cm，表面在 Z=0
    ground_path = "/World/Ground"
    prim_utils.create_prim(ground_path, "Cube", 
                           translation=(0, 0, -0.01), 
                           scale=(10, 10, 0.02))
    
    # 物理修正：添加碰撞 API
    stage = omni.usd.get_context().get_stage()
    ground_prim = stage.GetPrimAtPath(ground_path)
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    
    # 环境灯光
    if not stage.GetPrimAtPath("/World/DomeLight"):
        light_prim = stage.DefinePrim("/World/DomeLight", "DomeLight")
        light_prim.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(1000.0)

def create_cabinets_and_random_bin():
    print("Creating Cabinets and Randomized TrashBin...")
    # 抬高 5mm 以消除 Z-Fighting 视觉闪烁
    # 机柜 A (3.0, 3.0)
    prim_utils.create_prim("/World/Cabinet_A", "Cube", translation=(3.0, 3.0, 1.005), scale=(0.6, 0.6, 2.0))
    # 机柜 B (-3.0, 3.0)
    prim_utils.create_prim("/World/Cabinet_B", "Cube", translation=(-3.0, 3.0, 1.005), scale=(0.6, 0.6, 2.0))
    
    # 【随机任务逻辑】：废料桶位置三选一
    candidate_locations = [
        (0.0, -2.0, 0.255),  # 原位
        (-2.5, 0.0, 0.255),  # 侧方 A
        (2.5, 0.0, 0.255)    # 侧方 B
    ]
    chosen_pos = random.choice(candidate_locations)
    
    # 创建废料桶
    bin_path = "/World/TrashBin"
    prim_utils.create_prim(bin_path, "Cube", translation=chosen_pos, scale=(0.5, 0.5, 0.5))
    
    # 为废料桶设置黄色，便于识别
    omni.kit.commands.execute("ChangeProperty", prop_path=f"{bin_path}.primvars:displayColor", 
                             value=Gf.Vec3f(1.0, 1.0, 0.0), prev=None)
    
    print(f"TrashBin randomized to: {chosen_pos}")
    return chosen_pos

def create_deformable_rag(stage, prim_path, position):
    print(f"Creating Deformable Rag (Soft Object) at {prim_path}...")
    # 创建带网格的 Cube
    omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_type="Cube", prim_path=prim_path)
    
    prim = stage.GetPrimAtPath(prim_path)
    rag_prim = XFormPrim(prim_path)
    rag_prim.set_world_pose(position=position)
    rag_prim.set_local_scale((0.4, 0.4, 0.01)) # 抹布形状

    # 应用 Deformable 物理 API
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    PhysxSchema.PhysxDeformableBodyAPI.Apply(prim)

    # 属性注入（确保 GPU 仿真精细度）
    prim.CreateAttribute("physxDeformable:simulationResolution", Sdf.ValueTypeNames.Int).Set(20)
    prim.CreateAttribute("physxDeformable:collisionSimplification", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physxDeformable:youngsModulus", Sdf.ValueTypeNames.Float).Set(8000.0) # 较软
    prim.CreateAttribute("physxDeformable:damping", Sdf.ValueTypeNames.Float).Set(0.4)

def load_robot(usd_path):
    print(f"Loading Lite3 Robot from: {usd_path}")
    if not os.path.exists(usd_path):
        print(f"❌ Error: Robot USD not found at {usd_path}. Falling back to Mock.")
        prim_utils.create_prim("/World/Lite3", "Cube", translation=(0, 0, 0.3), scale=(0.4, 0.3, 0.2))
        return

    stage = omni.usd.get_context().get_stage()
    robot_prim_path = "/World/Lite3"
    
    # 使用 Reference 引用加载
    robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
    robot_prim.GetReferences().AddReference(usd_path)
    
    # 强制可见性并设置初始位姿
    imageable = UsdGeom.Imageable(robot_prim)
    if imageable: imageable.MakeVisible()
    
    # 放置在原点上方，利用物理落到地面
    XFormPrim(robot_prim_path).set_world_pose(position=(0.0, 0.0, 1.0))

# ==========================================
# 4. 主运行逻辑
# ==========================================

def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # --- 关键：配置 GPU 物理场景 ---
    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim:
        scene_prim = stage.DefinePrim("/physicsScene", "PhysicsScene")
    
    scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
    scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
    scene_prim.CreateAttribute("physics:gravityDirection", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene_prim.CreateAttribute("physics:gravityMagnitude", Sdf.ValueTypeNames.Float).Set(9.81)

    # --- 构建场景 ---
    create_environment()
    trash_bin_pos = create_cabinets_and_random_bin()
    
    # 在机柜 A 前放置抹布 (Z=0.5 掉落效果)
    create_deformable_rag(stage, "/World/Rag", position=(2.3, 3.0, 0.5))

    # 加载机器人
    robot_path = "D:/study/usd/deep_robotics_model/Lite3/Lite3_usd/Lite3.usd"
    load_robot(robot_path)

    # --- 启动仿真 ---
    world.reset()
    
    # 任务状态机
    # 0: 观测, 1: 移动到抹布, 2: 抓取, 3: 移动到桶, 4: 投放
    state = 0
    counter = 0

    print(">>> Logic Loop Started: Randomized Trashbin Task <<<")
    
    while simulation_app.is_running():
        world.step(render=True)
        counter += 1

        # 简单的任务逻辑演示（基于步数的状态切换）
        if counter % 100 == 0:
            # 获取抹布实时位置
            rag_pos, _ = XFormPrim("/World/Rag").get_world_pose()

            if state == 0 and counter > 200:
                print(f"[感知] 识别到抹布位置: {rag_pos}")
                state = 1
            
            elif state == 1:
                print(f"[动作] Lite3 机器人正在向机柜 A 移动...")
                state = 2
            
            elif state == 2:
                print(f"[动作] 已接触柔性体，正在执行抓取...")
                state = 3
            
            elif state == 3:
                print(f"[动作] 搬运中... 目标废料桶位置: {trash_bin_pos}")
                # 模拟抓取：将抹布附着到机器人或直接移动到桶上方
                XFormPrim("/World/Rag").set_world_pose(
                    position=(trash_bin_pos[0], trash_bin_pos[1], trash_bin_pos[2] + 0.6)
                )
                state = 4
            
            elif state == 4:
                print("[任务] 抹布已放入废料桶！任务成功完成。")
                state = 5

    simulation_app.close()

if __name__ == "__main__":
    main()