'''
Credit: @Dai Hang
'''

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import os
import signal
import sys
# import skvideo.io 
from omegaconf import OmegaConf
import scipy.spatial.transform as st
# from diffusion_policy.real_world.real_env import RealEnv #! replace to the eval agent
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.ensemble import EnsembleBuffer
from eval_agent import Agent
from termcolor import cprint
from tqdm import tqdm
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from pynput import keyboard
OmegaConf.register_new_resolver("eval", eval, replace=True)

import termios
import sys

def clear_input_buffer():
    termios.tcflush(sys.stdin, termios.TCIFLUSH)

def discretize_translation(pos_begin, pos_end, step_size):
    vector = pos_end - pos_begin
    distance = np.linalg.norm(vector)
    n_step = int(distance // step_size) + 1
    pos_steps = []
    for i in range(n_step):
        pos_i = pos_begin * (n_step - 1 - i) / n_step + pos_end * (i + 1) / n_step
        pos_steps.append(pos_i)
    return pos_steps

rot6d_quat_transformer = RotationTransformer(from_rep='rotation_6d', to_rep="quaternion")

interrupted = False

def _keyboard_signal_handler(key):
    if key.char == 't':
        global interrupted
        interrupted = True
        cprint("\nStopping...", "yellow")

keyboard_listener = keyboard.Listener(on_press=_keyboard_signal_handler)
keyboard_listener.start()

def run_single_episode(agent: Agent, policy, cfg, device, max_duration, gripper, output, use_all_joints=False):
    """Single episode running"""
    global interrupted
    global keyboard_listener
    try:
        agent.arm.home_robot()
        input("Press Enter to start.")
               
    except Exception as e:
        cprint(f"kill faild: {e}", "red")
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    print("Warming up policy inference")
    
    # Reset agent buffers at the start of each episode
    agent.reset_buffer()
    obs = agent.get_observation()
    with torch.no_grad():
        policy.reset()
        obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
        
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()
        if not use_all_joints:
            if gripper:
                assert action.shape[-1] == 10 # xyz + rot6d + gripper
            else:
                assert action.shape[-1] == 9 # xyz + rot6d
        del result
        del action

    ensemble_buffer = EnsembleBuffer(mode="avg")
    with torch.inference_mode():
        print("Start!")
        raw_imgs = []

        start_pose = agent.arm.get_tcp_position()
        obs = agent.get_observation()
        
        for t in tqdm(range(max_duration), desc="evaluating"):
            if not keyboard_listener.is_alive():
                keyboard_listener = keyboard.Listener(on_press=_keyboard_signal_handler)
                keyboard_listener.start()
            if interrupted:
                cprint("Interuppted. stop.", "yellow")
                break
                
            if t % 5 == 0:
                # run inference
                s = time.time()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                print("state", obs_dict_np["state"][:3])
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                result = policy.predict_action(obs_dict)
                # this action starts from the first obs step
                raw_action = result['action'][0].detach().to('cpu').numpy()
                ensemble_buffer.add_action(raw_action, t)
                
            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            
            if step_action is None:  # no action in the buffer => no movement.
                continue
            if use_all_joints:
                agent.set_joint_pose(np.array(step_action[:-1]))
            else:
                quat = rot6d_quat_transformer.forward(np.array(step_action[3:9]))
                step_tcp = np.concatenate([step_action[:3], quat])
                # print("action", step_action[:3])
                agent.set_tcp_pose(step_tcp)

            # control gripper
            if gripper:
                step_gripper = step_action[-1]
                cprint(step_gripper, "green") 
                agent.set_tcp_gripper(step_gripper)

            # agent.sleep()

            # update observation
            obs = agent.get_observation()
            if output is not None:
                raw_img = obs["raw_img"]
                raw_imgs.append(raw_img)
                
        if output is not None and raw_imgs:
            for idx_ in range(len(raw_imgs)):
                cv2.imwrite(os.path.join(output, f"rgb_{idx_:05d}.png"), raw_imgs[idx_])
            cprint(f"Image saved: {output}", "cyan")
            
        return not interrupted

@click.command()
@click.option('--ckpt', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', default=None, type=str, help='Directory to save recording')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=1000, help='Max duration in steps.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--gripper', '-g', is_flag=True, default=False, type=bool, help='Enable gripper control')
@click.option('--continuous', '-c', is_flag=True, default=True, type=bool, help='Enable continuous testing mode')
@click.option(
    "--use_all_joints", type=bool, default=False,
)
@click.option('--dim', type=int, default=None, help='Override low-dim state/action size at eval time')
def main(ckpt, output, match_dataset, match_episode,
    vis_camera_idx, steps_per_inference, max_duration, frequency, command_latency, gripper, continuous, use_all_joints, dim):
    global interrupted
    
    # load checkpoint
    ckpt_path = ckpt
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    assert 'diffusion' in cfg.name
    cprint("test diffusion!", "green")
    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    policy.eval().to(device)

    # set inference params
    policy.num_inference_steps = 16 # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    # setup experiment
    dt = 1/frequency

    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    agent = Agent(
        obs_num=n_obs_steps,
        gripper=gripper,
        use_all_joints=use_all_joints
    )

    if continuous:
        episode_count = 0
        cprint("Press T to terminate this test.", "cyan")
        while True:
            episode_count += 1
            interrupted = False
            cprint(f"\nTesting No. {episode_count} ", "blue")
            
            # create individual folder
            episode_output = None
            if output is not None:
                episode_output = os.path.join(output, f"episode_{episode_count:03d}")
                os.makedirs(episode_output, exist_ok=True)
            
            success = run_single_episode(agent, policy, cfg, device, max_duration, gripper, episode_output, use_all_joints)
            
            if success:
                cprint(f"Test No. {episode_count} finished", "green")
            else:
                cprint(f"Test No. {episode_count} stopped", "yellow")
            
            # Ask if continue
            try:
                clear_input_buffer()
                response = input("continue? (y/n)")
                response = response.strip().lower()
                if response in ['n', 'no', 'exit', 'quit']:
                    break
            except KeyboardInterrupt:
                cprint("\nExit.", "red")
                break
    else:
        # single episode
        episode_output = output
        if output is not None:
            os.makedirs(output, exist_ok=True)
        run_single_episode(agent, policy, cfg, device, max_duration, gripper, episode_output, use_all_joints)
    agent.arm.home_robot()
    print("Goodbye.")


# %%
if __name__ == '__main__':
    main()
