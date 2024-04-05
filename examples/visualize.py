import gymnasium as gym
import stompy_sim.tasks
import argparse
from mani_skill.utils.wrappers import RecordEpisode
def main(args):
    env = gym.make("Stand-v0", 
                   robot_uids=args.robot_uid, 
                   obs_mode=args.obs_mode, 
                   control_mode="pd_joint_delta_pos", 
                   render_mode="rgb_array", 
                   shader_dir=args.shader,
                   )
    # auto record videos to disk with the wrapper below
    if args.record_dir is not None:
        env = RecordEpisode(env, output_dir=args.record_dir, save_trajectory=True, save_video=True, trajectory_name="trajectory")
    env.reset(seed=0)
    N = 100
    if args.viewer:
        viewer=env.render_human()
        viewer.paused = True
        print("Simulation is paused, unpause on the top left.")
        env.render_human()
        N = 100000
    
    for i in range(N):
        print(f"Step {i}")
        action = env.action_space.sample()
        # action[:-4] = 0#env.unwrapped.agent.robot.qpos[0, :-4]
        # action[:] = 0
        # action[:-1] = env.unwrapped.agent.robot.qpos[0]
        env.step(action)
        if args.viewer: env.render_human()
    env.close()
    del env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="Stand-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("--shader", type=str, default="default", help="Can be 'default' or 'rt-fast'/'rt' for ray-tracing")
    parser.add_argument("-r", "--robot-uid", type=str, default="stompy", help="Robot setups supported are ['stompy']")
    parser.add_argument("--record-dir", type=str, default=None, help="directory to record videos rendered headlessly to. If None, video is not recorded")
    parser.add_argument("--viewer", action="store_true", help="Whether to show a GUI")
    args, opts = parser.parse_known_args()

    return args
if __name__ == "__main__":
    main(parse_args())

