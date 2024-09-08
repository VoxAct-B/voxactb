# VoxAct-B: Voxel-Based Acting and Stabilizing Policy for Bimanual Manipulation

[[Project website](https://voxact-b.github.io/)] [[Paper](https://arxiv.org/abs/2407.04152)]

This project is a PyTorch implementation of VoxAct-B: Voxel-Based Acting and Stabilizing Policy for Bimanual Manipulation, published in CoRL 2024.

Bimanual manipulation is critical to many robotics applications. In contrast to single-arm manipulation, bimanual manipulation tasks are challenging due to higher-dimensional action spaces. Prior works leverage large amounts of data and primitive actions to address this problem, but may suffer from sample inefficiency and limited generalization across various tasks. To this end, we propose VoxAct-B, a language-conditioned, voxel-based method that leverages Vision Language Models (VLMs) to prioritize key regions within the scene and reconstruct a voxel grid. We provide this voxel grid to our bimanual manipulation policy to learn acting and stabilizing actions. This approach enables more efficient policy learning from voxels and is generalizable to different tasks. In simulation, we show that VoxAct-B outperforms strong baselines on fine-grained bimanual manipulation tasks. Furthermore, we demonstrate VoxAct-B on real-world 𝙾𝚙𝚎𝚗 𝙳𝚛𝚊𝚠𝚎𝚛 and 𝙾𝚙𝚎𝚗 𝙹𝚊𝚛 tasks using two UR5s.

## Installation

### Prerequisites

VoxAct-B is built-off the [PerAct repository](https://github.com/peract/peract) by Shridhar et al. The prerequisites are the same as PerAct.

#### 1. Environment

```bash
# setup a virtualenv with whichever package manager you prefer
conda create -n voxactb python=3.8
conda activate voxactb
pip install --upgrade pip==24.0
```

#### 2. PyRep and Coppelia Simulator

Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)
- [Ubuntu 22.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces.

Finally install the python library:

```bash
cd PyRep
pip install -r requirements.txt
pip install .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).

#### 3. RLBench

```bash
cd ../RLBench
pip install -r requirements.txt
python setup.py develop
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

#### 4. YARR

```bash
cd ../YARR
pip install -r requirements.txt
python setup.py develop
```

Common Issues:
```
pkg_resources.extern.packaging.requirements.InvalidRequirement: .* suffix can only be used with `==` or `!=` operators
    PyYAML (>=5.1.*)

# Solution
pip install setuptools==66
```

#### 5. [Optional] urx for real-robot experiments
```bash
cd urx
python setup.py develop
```


### PerAct Repo

Install:
```bash
cd ../peract
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt

export PERACT_ROOT=$(pwd)  # mostly used as a reference point for tutorials
python setup.py develop

pip install transformers==4.40.0
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 # feel free to ignore "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts."
```

Create data folder:
```bash
# make sure you're inside the `voxactb/peract` folder
mkdir data
cd data
```

Download this [zip file](https://drive.google.com/file/d/1nSi1DEk1RUDek6N42izIMrE9RZxZLGwE/view?usp=sharing), unzip it, and place `clip_rn50.pth` and `segment_anything_vit_h.pth` inside the `voxactb/peract/data` folder.

#### VoxPoser

Install:
```bash
# make sure you're inside the `voxactb/peract/voxposer` folder
pip install -r requirements.txt
```

### [Optional] Running on a Headless Computer

Install:
```
sudo apt-get install mesa-utils x11-xserver-utils xvfb dbus-x11 x11-utils libxkbcommon-x11-0 
```

Export display variable:
```
export DISPLAY=:0.0
```

For each training/evaluation script, insert the following line before the python command:
```
xvfb-run -a --server-num=611 python <script>
```


### Gotchas

#### OpenGL Errors

GL errors are probably being caused by the PyRender voxel visualizer. See this [issue](https://github.com/mmatl/pyrender/issues/86) for reference. You might have to set the following environment variables depending on your setup:

```bash
export DISPLAY=:0
export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl
```

#### Unpickling Error

If you see `_pickle.UnpicklingError: invalid load key, '\x9e'`, maybe one of the replay pickle files got corrupted when quitting the training script. Try deleting files in `replay.path` and restarting training.

## How to run experiments

You can download the train/val/test datasets from [here](https://drive.google.com/drive/folders/1FOvFVElbKFhIHV5BOnsKlUDnejk1U2-5?usp=drive_link) (place the train, val, and test folders in the `voxactb/peract/data` folder) or generate the datasets by following [these instructions](#how-to-generate-datasets). 

To reproduce our results, you can download the [checkpoints](https://drive.google.com/drive/folders/1GJ1mFXCkRCeiMokoefG7s1BrmFEFoSGI?usp=sharing) and place them in the `voxactb/peract/log` folder.

Go to `voxactb/peract/scripts`.

Open Jar
```
# training
./train_open_jar_ours_vlm_10_demos_v2_11_acting.sh
./train_open_jar_ours_vlm_10_demos_v2_11_stabilizing.sh

# validation/test
./eval_open_jar_ours_vlm_10_demos_v2_vlm_11.sh
```

Open Drawer
```
# training
./train_open_drawer_ours_vlm_10_demos_v2_11_acting.sh
./train_open_drawer_ours_vlm_10_demos_v2_11_stabilizing.sh

# validation/test
./eval_open_drawer_ours_vlm_10_demos_v2_vlm_11.sh
```

Put Item in Drawer
```
# training
./train_put_item_in_drawer_ours_vlm_10_demos_v2_11_acting.sh
./train_put_item_in_drawer_ours_vlm_10_demos_v2_11_stabilizing.sh

# validation/test
./eval_put_item_in_drawer_ours_vlm_10_demos_v2_11.sh
```

Hand Over Item
```
# training
./train_hand_over_item_ours_vlm_10_demos_v1_11_acting.sh
./train_hand_over_item_ours_vlm_10_demos_v1_11_stabilizing.sh

# validation/test
./eval_hand_over_item_ours_vlm_10_demos_v1_11.sh
```

## How to generate datasets

First, make sure you're inside the `voxactb/RLBench/tools` folder.

Open Jar
```
# training data
python dataset_generator_two_robots.py --tasks=open_jar \
                            --save_path=$PERACT_ROOT/data/train \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=10 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=open_jar_noises_starting_states_dominant_assistive

# val data
python dataset_generator_two_robots.py --tasks=open_jar \
                            --save_path=$PERACT_ROOT/data/val \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=open_jar_close_to_jar_dominant_assistive \
                            --seed=43

# test data
python dataset_generator_two_robots.py --tasks=open_jar \
                            --save_path=$PERACT_ROOT/data/test \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=open_jar_close_to_jar_dominant_assistive \
                            --seed=88
```

Open Drawer
```
# training data
python dataset_generator_two_robots.py --tasks=open_drawer \
                            --save_path=$PERACT_ROOT/data/train \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=10 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=open_drawer_noises_starting_states_dominant_assistive

# val data
python dataset_generator_two_robots.py --tasks=open_drawer \
                            --save_path=$PERACT_ROOT/data/val \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=open_drawer_close_to_drawer_dominant_assistive \
                            --seed=43

# test data
python dataset_generator_two_robots.py --tasks=open_drawer \
                            --save_path=$PERACT_ROOT/data/test \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=open_drawer_close_to_drawer_dominant_assistive \
                            --seed=88
```

Put Item in Drawer
```
# training data
python dataset_generator_two_robots.py --tasks=put_item_in_drawer \
                            --save_path=$PERACT_ROOT/data/train \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=10 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=put_item_in_drawer_noises_starting_states_dominant_assistive											
# val data
python dataset_generator_two_robots.py --tasks=put_item_in_drawer \
                            --save_path=$PERACT_ROOT/data/val \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=put_item_in_drawer_noises_starting_states_dominant_assistive \
                            --seed=43

# test data
python dataset_generator_two_robots.py --tasks=put_item_in_drawer \
                            --save_path=$PERACT_ROOT/data/test \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=put_item_in_drawer_noises_starting_states_dominant_assistive \
                            --seed=88
```

Hand Over Item
```
# training data
python dataset_generator_two_robots.py --tasks=hand_over_item \
                            --save_path=$PERACT_ROOT/data/train \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=10 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=hand_over_item_noises_starting_states_dominant_assistive

# val data
python dataset_generator_two_robots.py --tasks=hand_over_item \
                            --save_path=$PERACT_ROOT/data/val \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=hand_over_item_noises_starting_states_dominant_assistive \
                            --seed=43

# test data
python dataset_generator_two_robots.py --tasks=hand_over_item \
                            --save_path=$PERACT_ROOT/data/test \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=25 \
                            --processes=1 \
                            --all_variations=True \
                            --mode=hand_over_item_noises_starting_states_dominant_assistive \
                            --seed=88
```

## Citations

**VoxAct-B**
```
@inproceedings{liu2024voxactb,
    title={VoxAct‐B: Voxel‐Based Acting and Stabilizing Policy for Bimanual Manipulation},
    author={I-Chun Arthur Liu and Sicheng He and Daniel Seita and Gaurav S. Sukhatme},
    booktitle={Conference on Robot Learning},
    year={2024}
}
```

**PerAct**
```
@inproceedings{shridhar2022peract,
    title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
    author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
    booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
    year      = {2022},
}
```

**C2FARM**
```
@inproceedings{james2022coarse,
    title={Coarse-to-fine q-attention: Efficient learning for visual robotic manipulation via discretisation},
    author={James, Stephen and Wada, Kentaro and Laidlow, Tristan and Davison, Andrew J},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={13739--13748},
    year={2022}
}
```

**PerceiverIO**
```
@article{jaegle2021perceiver,
    title={Perceiver io: A general architecture for structured inputs \& outputs},
    author={Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and others},
    journal={arXiv preprint arXiv:2107.14795},
    year={2021}
}
```

**RLBench**
```
@article{james2020rlbench,
    title={Rlbench: The robot learning benchmark \& learning environment},
    author={James, Stephen and Ma, Zicong and Arrojo, David Rovick and Davison, Andrew J},
    journal={IEEE Robotics and Automation Letters},
    volume={5},
    number={2},
    pages={3019--3026},
    year={2020},
    publisher={IEEE}
}
```

**VoxPoser**
```
@article{huang2023voxposer,
    title={VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models},
    author={Huang, Wenlong and Wang, Chen and Zhang, Ruohan and Li, Yunzhu and Wu, Jiajun and Fei-Fei, Li},
    journal={arXiv preprint arXiv:2307.05973},
    year={2023}
}
```