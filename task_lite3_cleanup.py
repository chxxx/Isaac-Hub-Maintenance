# ==============================================================================
# Full IsaacLab M20 + Piper Cleanup Task
# FINAL FIX: correct 7D pose API usage and clone tensors
# ==============================================================================

import debugpy
try:
    debugpy.listen(('localhost', 5678))
    print(">>> [Debug] Waiting for VS Code attach (F5)...")
    debugpy.wait_for_client()
except Exception:
    pass

# ==============================================================================
# App launch
# ==============================================================================

import argparse
from isaaclab.app import AppLauncher
import torch

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
# Isaac Lab imports
# ==============================================================================

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# ==============================================================================
# Scene Config
# ==============================================================================

@configclass
class CleanupSceneCfg(InteractiveSceneCfg):

    num_envs = 1
    env_spacing = 2.0

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    # ---------------- M20 Base ----------------
    m20 = ArticulationCfg(
        prim_path="/World/M20",
        spawn=sim_utils.UsdFileCfg(
            usd_path="D:/study/isaac/usd/deep_robotics_model/M20/M20_usd/M20.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=20.0,
                damping=2.0,
            )
        },
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # ---------------- Piper Arm ----------------
    piper = ArticulationCfg(
        prim_path="/World/Piper",
        spawn=sim_utils.UsdFileCfg(
            usd_path="D:/study/isaac/usd/Piper.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=60.0,
                damping=6.0,
            )
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={f"joint{i}": 0.0 for i in range(1, 7)},
        ),
    )

    # ---------------- Rag ----------------
    rag = RigidObjectCfg(
        prim_path="/World/Rag",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.05)),
    )

    # ---------------- Trash Bin (Xform) ----------------
    bin = AssetBaseCfg(
        prim_path="/World/TrashBin",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.0, 0.25)),
    )

# ==============================================================================
# Helper function to write root pose
# ==============================================================================

def write_root_pos_quat(asset, pos, quat):
    """
    pos: (1,3)
    quat: (1,4)
    """
    pose = torch.cat([pos.clone(), quat.clone()], dim=-1)
    asset.write_root_pose_to_sim(pose)

# ==============================================================================
# Task Logic
# ==============================================================================

def run_simulator(sim: SimulationContext, scene: InteractiveScene):

    print(">>> Simulation running")
    dt = sim.get_physics_dt()

    state = 0
    grasped = False

    while simulation_app.is_running():
        sim.step()
        scene.update(dt)

        m20 = scene["m20"]
        piper = scene["piper"]
        rag = scene["rag"]

        # clone to avoid aliasing
        m20_pos = m20.data.root_pos_w[0].unsqueeze(0).clone()
        m20_quat = m20.data.root_quat_w.clone()
        piper_pos = piper.data.root_pos_w[0].unsqueeze(0).clone()
        piper_quat = piper.data.root_quat_w.clone()
        rag_pos = rag.data.root_pos_w[0].unsqueeze(0).clone()
        rag_quat = rag.data.root_quat_w.clone()

        bin_pos, _ = scene["bin"].get_world_poses()
        bin_pos = bin_pos[0]

        dist = torch.norm(m20_pos[0,:2] - rag_pos[0,:2])

        # --------------------------------------------------
        # State 0: Move base
        # --------------------------------------------------
        if state == 0:
            direction = rag_pos[0,:2] - m20_pos[0,:2]
            d = torch.norm(direction)

            if d > 0.1:
                direction = direction / d
                new_pos = m20_pos.clone()
                new_pos[0,0] += direction[0] * 0.01
                new_pos[0,1] += direction[1] * 0.01

                write_root_pos_quat(m20, new_pos, m20_quat)
            else:
                state = 1
                print("\n>>> State 1: Base arrived")

        elif state == 1:
            target = torch.tensor([[rag_pos[0,0], rag_pos[0,1], 0.8]], device=sim.device)
            write_root_pos_quat(piper, target, piper_quat)
            state = 2

        elif state == 2:
            target = torch.tensor([[rag_pos[0,0], rag_pos[0,1], rag_pos[0,2] + 0.05]], device=sim.device)
            write_root_pos_quat(piper, target, piper_quat)
            grasped = True
            state = 3

        elif state == 3 and grasped:
            ee_pos = piper.data.root_pos_w[0].unsqueeze(0).clone()
            ee_pos[0,2] += 0.01
            write_root_pos_quat(piper, ee_pos, piper_quat)
            write_root_pos_quat(rag, ee_pos, rag_quat)
            if ee_pos[0,2] > 1.0:
                state = 4

        elif state == 4:
            target = torch.tensor([[bin_pos[0], bin_pos[1], 1.0]], device=sim.device)
            write_root_pos_quat(piper, target, piper_quat)
            write_root_pos_quat(rag, target, rag_quat)
            state = 5

        elif state == 5:
            drop = torch.tensor([[bin_pos[0], bin_pos[1], bin_pos[2] + 0.3]], device=sim.device)
            write_root_pos_quat(rag, drop, rag_quat)
            state = 6
            print(">>> Task completed")

        print(f"State={state} | dist={dist:.2f}", end="\r")

# ==============================================================================
# Main
# ==============================================================================

def main():
    sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))
    scene = InteractiveScene(CleanupSceneCfg())
    sim.reset()
    run_simulator(sim, scene)
    simulation_app.close()

if __name__ == "__main__":
    main()
