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
                   sim_backend="gpu"
                   )
    # auto record videos to disk with the wrapper below
    # env = RecordEpisode(env, output_dir=args.record_dir, save_trajectory=True, save_video=True, trajectory_name="trajectory")
    env.reset(seed=0)
    viewer=env.render_human()
    viewer.paused = True
    print("Simulation is paused, unpause on the top left.")
    env.render_human()
    for _ in range(10000):
        action = env.action_space.sample()
        # action[:-4] = 0#env.unwrapped.agent.robot.qpos[0, :-4]
        # action[:] = 0
        # action[:-1] = env.unwrapped.agent.robot.qpos[0]
        env.step(action)
        env.render_human()
    env.close()
    del env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="Stand-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("--shader", type=str, default="default", help="Can be 'default' or 'rt-fast'/'rt' for ray-tracing")
    parser.add_argument("-r", "--robot-uid", type=str, default="stompy", help="Robot setups supported are ['stompy']")
    parser.add_argument("--record-dir", type=str, default="videos")
    args, opts = parser.parse_known_args()

    return args
if __name__ == "__main__":
    main(parse_args())

