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
    # print("【等待 VS Code 附加调试器...】请在 VS Code 按 F5")
    # debugpy.wait_for_client() 
except Exception as e:
    pass

# ==========================================
# 1. 启动 SimulationApp
# ==========================================
simulation_app = SimulationApp({"headless": False, "anti_aliasing": 1})

# ==========================================
# 2. 导入模块
# ==========================================
import omni.usd
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.world import World
from omni.isaac.core.prims import XFormPrim, RigidPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

# ==========================================
# 3. 辅助功能函数
# ==========================================

def create_environment():
    print("Creating Environment (Solid Ground)...")
    ground_path = "/World/Ground"
    
    # 【修改 1】地面加厚到 0.1 (10cm)，防止穿透
    # 中心点下移到 -0.05，保证上表面依然是 Z=0
    prim_utils.create_prim(ground_path, "Cube", 
                           translation=(0, 0, -0.05), 
                           scale=(10, 10, 0.1))
    
    stage = omni.usd.get_context().get_stage()
    ground_prim = stage.GetPrimAtPath(ground_path)
    
    # 【修改 2】确保地面有碰撞属性
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    
    if not stage.GetPrimAtPath("/World/DomeLight"):
        light_prim = stage.DefinePrim("/World/DomeLight", "DomeLight")
        light_prim.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(1000.0)

def create_cabinets_and_random_bin():
    print("Creating Cabinets and Yellow TrashBin...")
    prim_utils.create_prim("/World/Cabinet_A", "Cube", translation=(3.0, 3.0, 1.005), scale=(0.6, 0.6, 2.0))
    prim_utils.create_prim("/World/Cabinet_B", "Cube", translation=(-3.0, 3.0, 1.005), scale=(0.6, 0.6, 2.0))
    
    candidate_locations = [(0.0, -2.0, 0.255), (-2.5, 0.0, 0.255), (2.5, 0.0, 0.255)]
    chosen_pos = random.choice(candidate_locations)
    
    bin_path = "/World/TrashBin"
    prim_utils.create_prim(bin_path, "Cube", translation=chosen_pos, scale=(0.5, 0.5, 0.5))
    
    stage = omni.usd.get_context().get_stage()
    # 垃圾桶也需要碰撞，否则抹布放上去会穿模（这里简单处理，实际应为中空）
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(bin_path))
    
    gprim = UsdGeom.Gprim(stage.GetPrimAtPath(bin_path))
    gprim.CreateDisplayColorAttr([(1.0, 1.0, 0.0)])
    
    print(f"TrashBin position set to: {chosen_pos}")
    return chosen_pos

def create_rag(stage, prim_path, position):
    print(f"Creating Rag (Rigid Body) at {prim_path}...")
    
    # 【修改 3】抹布厚度增加到 0.04 (4cm)，大幅增加物理稳定性
    prim_utils.create_prim(prim_path, "Cube", translation=position, scale=(0.4, 0.4, 0.04))
    
    rag_prim = stage.GetPrimAtPath(prim_path)

    # 【关键修改 4】显式应用 CollisionAPI！没有这行代码，物体就是幽灵。
    UsdPhysics.CollisionAPI.Apply(rag_prim)
    
    # 注册为刚体
    rag_rigid = RigidPrim(prim_path, mass=0.2)
    
    # 设置颜色
    gprim = UsdGeom.Gprim(rag_prim)
    gprim.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])

def load_robot(usd_path):
    print(f"Loading Lite3 Robot from: {usd_path}")
    stage = omni.usd.get_context().get_stage()
    robot_prim_path = "/World/Lite3"
    
    if not os.path.exists(usd_path):
        print(f"❌ Error: Robot USD not found. Falling back to Mock Cube.")
        # 如果是 Mock Cube，也要加 Collision 否则也会掉下去
        prim_utils.create_prim(robot_prim_path, "Cube", translation=(0, 0, 0.3), scale=(0.5, 0.3, 0.2))
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(robot_prim_path))
        RigidPrim(robot_prim_path, mass=5.0)
        return robot_prim_path

    robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
    robot_prim.GetReferences().AddReference(usd_path)
    XFormPrim(robot_prim_path).set_world_pose(position=(0.0, 0.0, 0.45)) # 稍微抬高机器人
    return robot_prim_path

# ==========================================
# 4. 主运行逻辑
# ==========================================

def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # 配置物理场景
    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim: scene_prim = stage.DefinePrim("/physicsScene", "PhysicsScene")
    scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)

    # --- 构建场景 ---
    create_environment()
    trash_bin_pos = create_cabinets_and_random_bin()
    
    # 【修改 5】生成高度设为 0.3，给它一个下落空间，避免卡在地面里
    create_rag(stage, "/World/Rag", position=(2.1, 2.1, 0.3))

    # --- 加载机器人 ---
    robot_path_str = "D:/study/usd/deep_robotics_model/Lite3/Lite3_usd/Lite3.usd"
    robot_prim_path = load_robot(robot_path_str)

    # 第一次 Reset
    world.reset()

    # --- 初始化控制器 ---
    lite3_robot = Articulation(robot_prim_path)
    lite3_robot.initialize()
    
    print("Wait for Physics View to initialize...")
    for _ in range(20):
        world.step(render=False)

    num_dofs = lite3_robot.num_dof
    default_joint_pos = None

    if num_dofs > 0:
        print(f">>> Robot loaded with {num_dofs} DOFs.")
        kps = np.full(num_dofs, 400.0)
        kds = np.full(num_dofs, 40.0)
        lite3_robot.get_articulation_controller().set_gains(kps=kps, kds=kds)
        default_joint_pos = np.zeros(num_dofs)

    # 任务状态机
    state = 0
    counter = 0

    print(">>> Starting Simulation Loop <<<")
    while simulation_app.is_running():
        world.step(render=True)
        
        # 机器人保持姿态
        if num_dofs > 0 and default_joint_pos is not None:
            if lite3_robot._articulation_view is not None and lite3_robot._articulation_view.initialized:
                try:
                    lite3_robot.apply_action(ArticulationAction(joint_positions=default_joint_pos))
                except Exception:
                    pass
        
        # 任务逻辑
        counter += 1
        if counter % 50 == 0: # 提高一点检测频率
            rag_prim = XFormPrim("/World/Rag")
            if not rag_prim.is_valid(): continue
            rag_pos, _ = rag_prim.get_world_pose()

            if state == 0 and counter > 100:
                print(f"[感知] 发现抹布位置: {rag_pos} (Z轴正常应 > 0)")
                state = 1
            elif state == 1:
                # print("[动作] 接近目标...") # 减少刷屏
                state = 2
            elif state == 2:
                # print("[动作] 抓取中...")
                state = 3
            elif state == 3:
                # print(f"[动作] 搬运至废料桶 {trash_bin_pos}")
                state = 4
            elif state == 4:
                print("[任务] 清理完成！")
                state = 5

    simulation_app.close()

if __name__ == "__main__":
    main()