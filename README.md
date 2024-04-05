# stompy-sim

Repo for simulating the Stompy robot from K-Scale Labs (and potentially others) with GPU sim and solving various tasks

## Installation

Recommended to use conda/mamba to setup

```bash
conda create -n "stompy-sim" "python==3.11"
pip install -e .
# install a version of torch that is compatible with your system
pip install torch torchvision torchaudio
# if you are on CUDA 11 you must also run
pip install fast_kinematics==0.1.11
```

This will install the [ManiSkill](https://github.com/haosulab/ManiSkill2/tree/dev) package, a open-source GPU state/visual parallelized robotics simulation framework. Star/watch the ManiSkill github repo to hear when it officially releases, it currently is in a beta phase (so do expect some bugs, you can post issues here).

For more instructions on installation / troubleshooting see https://maniskill.readthedocs.io/en/dev/user_guide/getting_started/installation.html. Vulkan is required for rendering.


## Scripts to run


To watch the simulation in a GUI (requires a display), run
```bash
python examples/visualize.py -e "Stand-v0" --viewer # sample random actions in the Stand-v0 environment
python examples/visualize.py -e "Stand-v0" --viewer --shader="rt-fast" # view the same but with fast ray-tracing on
python examples/visualize.py -e "Stand-v0" --viewer --shader="rt" # view the same but with high-quality ray-tracing on
```

To do the same but save a video headless (in case you don't have a display), just remove --viewer and add --record-dir="videos"
```bash
python examples/visualize.py -e "Stand-v0" --record-dir="videos"
```

Test CPU sim vs GPU sim FPS
```bash
python examples/fps.py -n 8 --cpu-sim # test cpu sim
python examples/fps.py -n 1024 # test gpu sim. beyond 1024 it might not work/run faster atm (probably a bug)

python examples/fps.py -n 8 -o rgbd --cpu-sim # test cpu sim with visual obs
python examples/fps.py -n 512 -o rgbd # test gpu sim with visual obs
```

Save a video showing 16 scenes at once in the GPU simulation
```bash
python examples/fps.py -n 16 --save-video --render-mode="rgb_array"
python examples/fps.py -n 16 --save-video --render-mode="sensors" # see what stompy sees
```

## Development Notes

- A sample task has been built in stompy_sim/tasks/stand.py
- Robot class implemented at stompy_sim/agents/stompy/stompy.py
- For working simulation a few things had to be changed to the URDF, see stompy_sim/agents/stompy/description/README.md
- To modify camera intrinsics/settings on the robot, see sensor_configs property in stompy_sim/agents/stompy/stompy.py
- To add other external cameras you can define them in the task itself (a base camera is defined already)
- Still optimizing this humanoid robot among several others, so simulation speed is not at its peak yet. There are likely some hacks/tricks that can speed it up without sacrificing simulation correctness.