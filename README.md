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