"""Offline checks: tiny physics scene and GPU 0 rendering, no external assets."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--native', action='store_true')
parser.add_argument('--only', action='append', help='Rerun selected checks and preserve the prior report')
args = parser.parse_args()
assert sys.prefix == '/root/miniconda3/envs/RoboTwin'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
sys.path.insert(0, '/home/lmy/Robbyant/RoboTwin')
out = Path(__file__).parent
report = {'prefix': sys.prefix, 'python': platform.python_version(),
          'gpu': 'physical GPU 0 only', 'checks': {}, 'versions': {}}
if args.only:
    report = json.loads((out / ('verification-native.json' if args.native else 'verification-basic.json')).read_text())

def check(name, function):
    if args.only and name not in args.only:
        return
    try:
        detail = function()
        report['checks'][name] = {'passed': True, 'detail': detail}
        print('PASS', name, detail, flush=True)
    except Exception:
        detail = traceback.format_exc()
        report['checks'][name] = {'passed': False, 'detail': detail}
        print('FAIL', name, detail, flush=True)
    (out / ('verification-native.json' if args.native else 'verification-basic.json')).write_text(json.dumps(report, indent=2))

for package in ['torch', 'torchvision', 'numpy', 'sapien', 'mplib', 'scipy', 'open3d',
                'gymnasium', 'huggingface-hub', 'setuptools', 'pytorch3d', 'nvidia-curobo',
                'warp-lang', 'zarr', 'moviepy']:
    try:
        report['versions'][package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        report['versions'][package] = None

def cuda():
    import torch
    import numpy as np
    assert sys.version_info[:2] == (3, 10)
    assert torch.__version__.split('+')[0] == '2.4.1'
    assert np.__version__ == '1.26.4'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    a = torch.arange(64, dtype=torch.float32, device='cuda').reshape(8, 8)
    torch.testing.assert_close((a @ a.T).cpu(), a.cpu() @ a.cpu().T)
    torch.cuda.synchronize()
    return {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'device': torch.cuda.get_device_name(0)}

check('cuda_matmul', cuda)
for name in ['torchvision', 'sapien.core', 'mplib', 'scipy', 'gymnasium', 'open3d',
             'trimesh', 'cv2', 'h5py', 'zarr', 'imageio', 'av']:
    check('import:' + name, lambda name=name: str(importlib.import_module(name).__file__))

def simulation():
    import numpy as np
    import sapien.core as sapien
    import sapien.render
    from transforms3d.quaternions import mat2quat
    from PIL import Image

    # The legacy SapienRenderer wrapper ignores keyword arguments in 3.0.0b1.
    # Select the device on RenderSystem itself, before creating any render object.
    render_system = sapien.render.RenderSystem(device='cuda:0')
    assert render_system.device.cuda_id == 0, str(render_system.device)
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [1, 1, 1], shadow=True)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
    builder.add_box_visual(half_size=[0.05, 0.05, 0.05], material=[0.8, 0.15, 0.1])
    cube = builder.build(name='verification_cube')
    cube.set_pose(sapien.Pose([0, 0, 0.4]))
    for _ in range(200):
        scene.step()
    resting_z = float(cube.get_pose().p[2])
    assert 0.04 < resting_z < 0.065, resting_z

    position = np.array([0.6, 0.0, 0.4])
    forward = np.array([0., 0., 0.05]) - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0., 0., 1.], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    camera_pose = sapien.Pose(position, mat2quat(np.stack([forward, left, up], axis=1)))
    results = {}
    for shader in ['default', 'rt']:
        sapien.render.set_camera_shader_dir(shader)
        if shader == 'rt':
            sapien.render.set_ray_tracing_samples_per_pixel(4)
            sapien.render.set_ray_tracing_path_depth(4)
            sapien.render.set_ray_tracing_denoiser('oidn')
        camera = scene.add_camera('verification_' + shader, 64, 64, 1.0, 0.01, 10.0)
        camera.set_pose(camera_pose)
        scene.update_render()
        camera.take_picture()
        color = camera.get_picture('Color')
        assert color.shape == (64, 64, 4) and np.isfinite(color).all()
        variation = float(color[..., :3].std())
        assert variation > 0.01, variation
        image_path = out / ('render-' + shader + '.png')
        Image.fromarray((np.clip(color[..., :3], 0, 1) * 255).astype(np.uint8)).save(image_path)
        results[shader] = {'rgb_std': variation, 'image': str(image_path)}
    return {'resting_cube_z': resting_z, 'renderer_device': str(render_system.device),
            'renderer_pci': render_system.device.pci_string, 'renders': results}

check('headless_physics_and_rendering', simulation)

if args.native:
    for name in ['pytorch3d.ops', 'warp', 'curobo.wrap.reacher.motion_gen',
                 'curobo.curobolib.lbfgs_step_cu', 'curobo.curobolib.kinematics_fused_cu',
                 'curobo.curobolib.line_search_cu', 'curobo.curobolib.tensor_step_cu',
                 'curobo.curobolib.geom_cu', 'envs._base_task', 'envs.robot.planner']:
        check('import:' + name, lambda name=name: str(importlib.import_module(name).__file__))

    def mplib_plan():
        import mplib
        import numpy as np
        urdf = out / 'verification-slider.urdf'
        srdf = out / 'verification-slider.srdf'
        urdf.write_text('''<robot name="verification_slider">
<link name="base"/>
<link name="slider"><inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="slider"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint>
</robot>''')
        srdf.write_text('''<robot name="verification_slider">
<group name="arm"><chain base_link="base" tip_link="slider"/></group>
<disable_collisions link1="base" link2="slider" reason="Adjacent"/>
</robot>''')
        planner = mplib.Planner(urdf=urdf, srdf=srdf, move_group='slider')
        result = planner.plan_qpos([np.array([0.25])], np.array([0.0]), time_step=0.02, planning_time=1)
        assert result['status'] == 'Success', result['status']
        assert np.isfinite(result['position']).all()
        assert abs(float(result['position'][-1, 0]) - 0.25) < 1e-3
        return {'status': result['status'], 'trajectory_shape': list(result['position'].shape)}
    check('mplib_joint_motion_plan', mplib_plan)

    def fps():
        import torch
        from pytorch3d.ops import sample_farthest_points
        points = torch.randn(1, 32, 3, device='cuda')
        sampled, indices = sample_farthest_points(points, K=8)
        assert sampled.shape == (1, 8, 3) and indices.shape == (1, 8)
        assert torch.isfinite(sampled).all()
        torch.cuda.synchronize()
        return {'points': list(sampled.shape), 'unique_indices': int(indices.unique().numel())}
    check('pytorch3d_cuda_fps', fps)

    def fk():
        import torch
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
        from curobo.types.base import TensorDeviceType
        tensor_args = TensorDeviceType(device=torch.device('cuda:0'))
        config = CudaRobotModelConfig.from_robot_yaml_file('franka.yml', tensor_args=tensor_args)
        model = CudaRobotModel(config)
        q = torch.zeros((1, model.get_dof()), device='cuda')
        state = model.get_state(q)
        assert torch.isfinite(state.ee_position).all()
        assert state.ee_position.shape == (1, 3)
        torch.cuda.synchronize()
        return {'dof': model.get_dof(), 'ee_position': state.ee_position.detach().cpu().tolist()}
    check('curobo_cuda_forward_kinematics', fk)

    def curobo_plan():
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
        tensor_args = TensorDeviceType(device=torch.device('cuda:0'))
        config = MotionGenConfig.load_from_robot_config(
            'franka.yml', 'collision_table.yml', tensor_args,
            num_ik_seeds=4, num_graph_seeds=1, num_trajopt_seeds=1,
            use_cuda_graph=False, interpolation_steps=1000, interpolation_dt=0.02)
        planner = MotionGen(config)
        start = JointState.from_position(planner.get_retract_config().view(1, -1).clone())
        goal = start.clone()
        goal.position[0, 0] += 0.05
        result = planner.plan_single_js(start, goal, MotionGenPlanConfig(
            enable_graph=False, max_attempts=2, enable_finetune_trajopt=False))
        assert bool(result.success.item()), str(result.status)
        trajectory = result.get_interpolated_plan().position
        assert torch.isfinite(trajectory).all()
        torch.cuda.synchronize()
        return {'success': True, 'trajectory_shape': list(trajectory.shape),
                'solve_time_seconds': float(result.solve_time)}
    check('curobo_cuda_joint_motion_plan', curobo_plan)

report['passed'] = all(c['passed'] for c in report['checks'].values())
(out / ('verification-native.json' if args.native else 'verification-basic.json')).write_text(json.dumps(report, indent=2))
print('Overall passed:', report['passed'], flush=True)
sys.exit(0 if report['passed'] else 1)
