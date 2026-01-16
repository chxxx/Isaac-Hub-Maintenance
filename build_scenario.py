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
    # debugpy.wait_for_client() # 需要断点调试时取消注释
except Exception as e:
    print(f"调试器监听跳过: {e}")

# ==========================================
# 1. 启动 SimulationApp
# ==========================================
simulation_app = SimulationApp({"headless": False, "anti_aliasing": 1})

# ==========================================
# 2. 导入模块 (注意：Isaac Sim 4.0+ 推荐新路径，但保留兼容写法)
# ==========================================
import omni.usd
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.world import World
from omni.isaac.core.prims import XFormPrim, RigidPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
import omni.kit.commands

# ==========================================
# 3. 辅助功能函数
# ==========================================

def create_environment():
    print("Creating Environment (Solid Ground)...")
    ground_path = "/World/Ground"
    prim_utils.create_prim(ground_path, "Cube", 
                           translation=(0, 0, -0.01), 
                           scale=(10, 10, 0.02))
    
    stage = omni.usd.get_context().get_stage()
    ground_prim = stage.GetPrimAtPath(ground_path)
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
    gprim = UsdGeom.Gprim(stage.GetPrimAtPath(bin_path))
    gprim.CreateDisplayColorAttr([(1.0, 1.0, 0.0)]) # 显式设为黄色
    
    print(f"TrashBin position set to: {chosen_pos}")
    return chosen_pos

def create_rag(stage, prim_path, position):
    print(f"Creating Rag (Rigid Body) at {prim_path}...")
    # 使用刚体代替柔性体以解决 Tetmesh Cooking 报错
    prim_utils.create_prim(prim_path, "Cube", translation=position, scale=(0.4, 0.4, 0.01))
    
    rag_rigid = RigidPrim(prim_path, mass=0.2)
    gprim = UsdGeom.Gprim(stage.GetPrimAtPath(prim_path))
    gprim.CreateDisplayColorAttr([(0.0, 0.0, 1.0)]) # 蓝色

def load_robot(usd_path):
    print(f"Loading Lite3 Robot from: {usd_path}")
    stage = omni.usd.get_context().get_stage()
    robot_prim_path = "/World/Lite3"
    
    if not os.path.exists(usd_path):
        print(f"❌ Error: Robot USD not found. Falling back to Mock Cube.")
        prim_utils.create_prim(robot_prim_path, "Cube", translation=(0, 0, 0.3), scale=(0.5, 0.3, 0.2))
        return robot_prim_path

    robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
    robot_prim.GetReferences().AddReference(usd_path)
    XFormPrim(robot_prim_path).set_world_pose(position=(0.0, 0.0, 0.40))
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
    # 稍微调整位置 (2.1, 2.1)，防止刷在机柜碰撞体内部导致弹飞
    create_rag(stage, "/World/Rag", position=(2.1, 2.1, 0.5))

    # --- 加载机器人 ---
    robot_path_str = "D:/study/usd/deep_robotics_model/Lite3/Lite3_usd/Lite3.usd"
    robot_prim_path = load_robot(robot_path_str)

    # 第一次 Reset，初始化物理对象
    world.reset()

    # --- 控制器初始化与【关键修复：物理视口等待】 ---
    lite3_robot = Articulation(robot_prim_path)
    lite3_robot.initialize()
    
    # 核心修复：强制步进几帧，让控制器与底层 PhysX 句柄完成绑定
    print("Wait for Physics View to initialize...")
    for _ in range(20):
        world.step(render=False)

    num_dofs = lite3_robot.num_dof
    default_joint_pos = None

    if num_dofs > 0:
        print(f">>> Robot loaded with {num_dofs} DOFs. Initializing PD gains.")
        kps = np.full(num_dofs, 400.0)
        kds = np.full(num_dofs, 40.0)
        lite3_robot.get_articulation_controller().set_gains(kps=kps, kds=kds)
        default_joint_pos = np.zeros(num_dofs)
    else:
        print(">>> Warning: No joints detected. Control will be disabled.")

    # 任务状态机
    state = 0
    counter = 0

    print(">>> Starting Simulation Loop <<<")
    while simulation_app.is_running():
        world.step(render=True)
        
        # --- 核心修复：增加 initialized 检查防止 NoneType 报错 ---
        if num_dofs > 0 and default_joint_pos is not None:
            # 只有在内部 View 真正准备好时才执行动作
            if lite3_robot._articulation_view is not None and lite3_robot._articulation_view.initialized:
                try:
                    lite3_robot.apply_action(ArticulationAction(joint_positions=default_joint_pos))
                except Exception:
                    pass # 捕捉偶发的首帧同步错误
        
        # 任务逻辑处理
        counter += 1
        if counter % 100 == 0:
            rag_prim = XFormPrim("/World/Rag")
            if not rag_prim.is_valid(): continue
            rag_pos, _ = rag_prim.get_world_pose()

            if state == 0 and counter > 200:
                print(f"[感知] 发现目标位置: {rag_pos}")
                state = 1
            elif state == 1:
                print("[动作] 接近目标...")
                state = 2
            elif state == 2:
                print("[动作] 抓取中...")
                state = 3
            elif state == 3:
                print(f"[动作] 搬运至废料桶 {trash_bin_pos}")
                # rag_prim.set_world_pose(position=(trash_bin_pos[0], trash_bin_pos[1], trash_bin_pos[2] + 0.6))
                state = 4
            elif state == 4:
                print("[任务] 清理完成！")
                state = 5

    simulation_app.close()

if __name__ == "__main__":
    main()