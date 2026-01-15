import debugpy
import os

# ==========================================
# 0. 调试器配置
# ==========================================
try:
    debugpy.listen(('localhost', 5678))
    print("【等待 VS Code 附加调试器...】请在 VS Code 按 F5")
    # debugpy.wait_for_client() # 调试时取消注释此行
except Exception as e:
    print(f"调试器监听跳过: {e}")

# ==========================================
# 1. 启动 SimulationApp
# ==========================================
from omni.isaac.kit import SimulationApp
# 开启抗锯齿，让视觉效果更好
simulation_app = SimulationApp({"headless": False, "anti_aliasing": 1})

# ==========================================
# 2. 导入模块
# ==========================================
import omni.usd
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.world import World
from omni.isaac.core.prims import XFormPrim
# 引入 Robot 类，为未来控制做准备
from omni.isaac.core.robots import Robot 
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
import omni.kit.commands

# ==========================================
# 场景构建函数
# ==========================================

def create_environment():
    print("Creating Environment (Solid Ground)...")
    # 【视觉修正】实体地面，厚度 2cm，中心下移 1cm => 表面在 Z=0
    ground_path = "/World/Ground"
    prim_utils.create_prim(ground_path, "Cube", 
                           translation=(0, 0, -0.01), 
                           scale=(10, 10, 0.02))
    
    # 【物理修正】添加碰撞
    stage = omni.usd.get_context().get_stage()
    ground_prim = stage.GetPrimAtPath(ground_path)
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    
    # 调暗一点环境光，让物体更有质感
    if not stage.GetPrimAtPath("/World/DomeLight"):
        light_prim = stage.DefinePrim("/World/DomeLight", "DomeLight")
        light_prim.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(800.0)

def create_cabinets():
    print("Creating Cabinets with clearance...")
    # 【Z-Fighting 修正】所有放在地上的物体，Z 坐标抬高 5mm (0.005)
    
    # 机柜 A (高 2.0 -> 半高 1.0 -> Z=1.005)
    prim_utils.create_prim("/World/Cabinet_A", "Cube", 
                           translation=(3.0, 3.0, 1.005), 
                           scale=(0.6, 0.6, 2.0))
    
    # 机柜 B
    prim_utils.create_prim("/World/Cabinet_B", "Cube", 
                           translation=(-3.0, 3.0, 1.005), 
                           scale=(0.6, 0.6, 2.0))
    
    # 垃圾桶
    prim_utils.create_prim("/World/TrashBin", "Cube", 
                           translation=(0.0, -2.0, 0.255), 
                           scale=(0.5, 0.5, 0.5))

def create_deformable_rag(stage, prim_path, position):
    print(f"Creating Deformable Rag at {prim_path}...")
    omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_type="Cube", prim_path=prim_path)
    
    prim = stage.GetPrimAtPath(prim_path)
    rag_prim = XFormPrim(prim_path)
    rag_prim.set_world_pose(position=position)
    # 抹布尺寸
    rag_prim.set_local_scale((0.4, 0.4, 0.01)) 

    # 应用物理 API
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    PhysxSchema.PhysxDeformableBodyAPI.Apply(prim)

    # 【底层属性注入】确保参数生效
    prim.CreateAttribute("physxDeformable:simulationResolution", Sdf.ValueTypeNames.Int).Set(20)
    prim.CreateAttribute("physxDeformable:collisionSimplification", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physxDeformable:youngsModulus", Sdf.ValueTypeNames.Float).Set(10000.0)
    prim.CreateAttribute("physxDeformable:damping", Sdf.ValueTypeNames.Float).Set(0.5)

def load_robot(usd_path):
    print(f"Attempting to load robot from: {usd_path}")
    
    # 1. 路径校验 (Isaac Sim 对路径非常敏感)
    if not os.path.exists(usd_path):
        print(f"❌ 路径不存在: {usd_path}")
        prim_utils.create_prim("/World/Lite3_Mock", "Cube", translation=(0, 0, 0.5), scale=(0.4, 0.2, 0.2))
        return

    # 2. 【关键修正】使用引用（References）方式加载
    # 很多时候直接 create_prim 会因为层级关系导致模型不可见
    stage = omni.usd.get_context().get_stage()
    robot_prim_path = "/World/Lite3"
    
    # 如果已存在则先删除，防止重复加载导致的冲突
    if stage.GetPrimAtPath(robot_prim_path):
        stage.RemovePrim(robot_prim_path)

    # 创建一个 Xform 节点作为容器
    robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
    # 将外部 USD 引用进来
    robot_prim.GetReferences().AddReference(usd_path)
    
    # 3. 设置坐标
    # 注意：有的 Lite3 USD 模型单位是厘米(cm)，如果是那样，Z=0.45 就在地板里
    # 我们先设高一点 (1.0m)，并重置缩放
    from omni.isaac.core.prims import GeometryPrim
    xform_robot = XFormPrim(robot_prim_path)
    xform_robot.set_world_pose(position=(0.0, 0.0, 1.0))
    xform_robot.set_local_scale((1.0, 1.0, 1.0)) # 强制 1:1 缩放，防止模型太小看不见

    print(f"✅ Robot loaded at {robot_prim_path}. If still invisible, check 'F' key in Viewport.")

# ==========================================
# 主运行逻辑
# ==========================================

def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # ========================================================
    # 【GPU 物理强制开启】(解决 Deformable 报错的唯一真理)
    # ========================================================
    scene_path = "/physicsScene"
    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim:
        scene_prim = stage.DefinePrim(scene_path, "PhysicsScene")
    
    scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
    scene_prim.CreateAttribute("physxScene:enableGPUPostSolver", Sdf.ValueTypeNames.Bool).Set(True)
    scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
    
    scene_prim.CreateAttribute("physics:gravityDirection", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene_prim.CreateAttribute("physics:gravityMagnitude", Sdf.ValueTypeNames.Float).Set(9.81)
    
    print(">>> Physics Scene Forced to GPU Mode <<<")

    # 构建环境
    create_environment()
    create_cabinets()

    # 放置抹布 (Cabinet A 前方)
    # 计算：3.0(中心) - 0.3(柜宽) - 0.4(距离) = 2.3
    create_deformable_rag(stage, "/World/Rag", position=(2.3, 3.0, 0.8))

    # 【加载 Lite3 机器人】
    # 假设 usd 文件在脚本同级目录，如果不是，请修改这里的路径
    # 比如: "C:/Assets/Lite3/Lite3_robot.usd"
    robot_path = os.path.abspath("D:/study/usd/deep_robotics_model/Lite3/Lite3_usd/Lite3.usd") 
    load_robot(robot_path)

    # 运行
    world.reset()
    print("Simulation starting... Watch Lite3 drop and the Rag deform.")
    
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()