# stompy-sim

Repo for simulating the Stompy robot (and potentially others) with GPU sim and solving various tasks

## Installation

Recommended to use conda/mamba to setup

```
conda create -n "stompy-sim" "python==3.11"
pip install -e .
# install a version of torch that is compatible with your system
pip install torch torchvision torchaudio
# if you are on CUDA 11 you must also run
pip install fast_kinematics==0.1.11
```

This will install the ManiSkill package. For more instructions on installation / troubleshooting see https://maniskill.readthedocs.io/en/dev/user_guide/getting_started/installation.html. Vulkan is required for rendering.

## Scripts to run


To watch the simulation in a GUI (requires a display), run
```bash
python examples/visualize.py -e "Stand-v0" # sample random actions in the Stand-v0 environment
python examples/visualize.py -e "Stand-v0" --shader="rt-fast" # view the same but with ray-tracing on
```

You can also save a video like so without a display:
```bash
python examples/fps.py -n 1 --save-video --render-mode="rgb_array" # render from a base camera
python examples/fps.py -n 1 --save-video --render-mode="sensors" # render from all sensors attached to the robot
```

Test CPU sim vs GPU sim FPS
```bash
python examples/fps.py -n 8 --cpu-sim # test cpu sim
python examples/fps.py -n 1024 # test gpu sim. beyond 1024 it might not work/run faster atm (probably a bug)

python examples/fps.py -n 8 -o rgbd --cpu-sim # test cpu sim with visual obs
python examples/fps.py -n 512 -o rgbd # test gpu sim with visual obs
```

Save a video of the GPU simulation
```bash
python examples/fps.py -n 64 --save-video --render-mode="rgb_array"
```