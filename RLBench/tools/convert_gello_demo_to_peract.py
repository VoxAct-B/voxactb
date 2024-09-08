from natsort import natsorted
import glob
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import tyro
from dataclasses import dataclass
from rlbench.backend.observation_two_robots import Observation2Robots
from rlbench.backend.utils import ClipFloatValues, float_array_to_rgb_image
from rlbench.demo import Demo
from rlbench.backend.const import *
from rlbench.backend.camera_const import *
from scipy.spatial.transform import Rotation as R

@dataclass
class Args:
    source_dir: str = "/home/arthur/Documents/gello_software/demos/gello/0521_172932"
    dest_dir: str = "/home/arthur/Documents/bimani/peract/data/train/debug_open_drawer_2024_05_21_v1/open_drawer/all_variations/episodes"
    ep_num: int = 0
    skip_frame_after_t_step: int = 1000000
    task_description: str = "hold the drawer with right hand and open the top drawer with left hand"

def center_crop(image):
    # Width to be cropped to
    new_width = image.shape[0]

    # Original dimensions
    original_height, original_width = image.shape[:2]

    # Calculating the starting and ending indices
    start_width = (original_width - new_width) // 2
    end_width = start_width + new_width

    # Cropping the image
    center_cropped_image = image[:, start_width:end_width]

    return center_cropped_image

def pad_image(image, mode='constant'):
    # Use width as the desired output shape
    output_shape = (image.shape[1], image.shape[1], image.shape[2])

    # Calculate the padding sizes for the first two dimensions
    padding_top = (output_shape[0] - image.shape[0]) // 2
    padding_bottom = output_shape[0] - image.shape[0] - padding_top
    padding_left = (output_shape[1] - image.shape[1]) // 2
    padding_right = output_shape[1] - image.shape[1] - padding_left

    # Apply the padding
    if mode == 'constant':
        padded_image = np.pad(image, 
                                ((padding_top, padding_bottom), 
                                (padding_left, padding_right), 
                                (0, 0)), 
                                mode=mode, constant_values=0)
    else:
        padded_image = np.pad(image, 
                            ((padding_top, padding_bottom), 
                            (padding_left, padding_right), 
                            (0, 0)), 
                            mode=mode)
    return padded_image

def convert_cam_coordinates_into_robot_frame_coordinates(arr):
    camera_frame_point = [arr[0], arr[1], arr[2], 1]
    robot_frame_point = np.matmul(LEFT_ARM_EXTRINSICS, camera_frame_point)
    return robot_frame_point[:3]

if __name__ == "__main__":
    args = tyro.cli(Args)
    print('Args: ', args)
    source_dir = args.source_dir
    pkls = natsorted(
        glob.glob(os.path.join(source_dir, "**/*.pkl"), recursive=True), reverse=False
    )

    # remove the first few frames because they are not useful.
    pkls = pkls[2:]

    # create folders for demos
    dest_dir = args.dest_dir
    save_path = (
        Path(dest_dir).expanduser()
    )
    save_path.mkdir(parents=True, exist_ok=True)
    ep_num = args.ep_num

    # make folders for current episode
    save_path_ep = save_path / f'episode{ep_num}'
    save_path_ep.mkdir(parents=True, exist_ok=True)
    save_path_ep_rgb = save_path_ep / 'front_rgb'
    save_path_ep_rgb.mkdir(parents=True, exist_ok=True)
    save_path_ep_depth = save_path_ep / 'front_depth'
    save_path_ep_depth.mkdir(parents=True, exist_ok=True)

    t_step = 0
    low_dim_obs = []

    # load misc.pkl
    assert 'misc.pkl' in pkls[-1]
    with open(pkls[-1], "rb") as f:
        misc = pickle.load(f)
        misc['front_camera_intrinsics']['fx']
        cam_intrinsics = np.array([
            [misc['front_camera_intrinsics']['fx'], 0, misc['front_camera_intrinsics']['cx']],
            [0, misc['front_camera_intrinsics']['fy'], misc['front_camera_intrinsics']['cy']],
            [0, 0, 1],
        ])

    for pkl in pkls:
        if 'misc.pkl' in pkl:
            # skip misc.pkl
            continue
        if t_step > args.skip_frame_after_t_step:
            # skip all pkls after t_step > skip_frame_after_t_step
            continue

        try:
            with open(pkl, "rb") as f:
                demo = pickle.load(f)
        except:
            print(f"Skipping {pkl} because it is corrupted.")
            continue

        # process and save images
        front_rgb = pad_image(demo['front_rgb'], mode='constant')
        front_rgb = Image.fromarray(front_rgb)
        front_rgb.save(os.path.join(save_path_ep_rgb, f'{t_step}.png'))

        front_depth = pad_image(demo['front_depth'], mode='edge')[:, :, 0]
        front_depth = front_depth / 1000.0 # convert to meters
        front_depth = float_array_to_rgb_image(front_depth, scale_factor=DEPTH_SCALE)
        front_depth.save(os.path.join(save_path_ep_depth, f'{t_step}.png'))
        t_step += 1

        misc_obj = {
            'front_camera_extrinsics': LEFT_ARM_EXTRINSICS, # not used; we're actually not going to use this in our method because we have two transformation matrices (left and right)
            'front_camera_intrinsics': cam_intrinsics,
            'front_camera_near': -1, # not used
            'front_camera_far': -1, # not used
            'left_arm_extrinsics': LEFT_ARM_EXTRINSICS,
            'right_arm_extrinsics': RIGHT_ARM_EXTRINSICS,
        }

        # gripper_position's z coordinate needs be offset to account for the 2F-85 gripper in visualization (not sure why)
        # z_coordinate_offset_for_gripper = -0.174 # meters based on the manuall: https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf
        # demo['ee_pos_quat'][2] += z_coordinate_offset_for_gripper
        # demo['ee_pos_quat'][9] += z_coordinate_offset_for_gripper

        low_dim_ob = Observation2Robots(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            demo['joint_velocities'][7:], # right joint velocities
            demo['joint_positions'][7:],  # right joint positions
            None,
            demo['joint_velocities'][:7], # left joint velocities
            demo['joint_positions'][:7],  # left joint positions
            None,
            np.array([demo['gripper_position'][1]]),
            demo['ee_pos_quat'][7:], # gripper right pose
            None,
            np.array([demo['gripper_position'][1], demo['gripper_position'][1]]),
            None,
            np.array([demo['gripper_position'][0]]),
            demo['ee_pos_quat'][:7], # gripper left pose
            None,
            np.array([demo['gripper_position'][0], demo['gripper_position'][0]]),
            None,
            None,
            1.0,
            misc_obj,
            misc['target_object_pos_cam_coordinates'], # save target_object_pos_cam_coordinates in camera frame
            None,
        )
        low_dim_obs.append(low_dim_ob)

    # save low_dim_obs
    low_dim_obs_pickle = Demo(low_dim_obs)
    with open(save_path_ep / 'low_dim_obs.pkl', 'wb') as f:
        # Serialize and save the object to the file
        pickle.dump(low_dim_obs_pickle, f)

    # save variation_number.pkl
    var_num = 0
    with open(save_path_ep / 'variation_number.pkl', 'wb') as f:
        # Serialize and save the object to the file
        pickle.dump(var_num, f)

    # save variation_descriptions.pkl
    variation_descriptions = [args.task_description]
    with open(save_path_ep / 'variation_descriptions.pkl', 'wb') as f:
        # Serialize and save the object to the file
        pickle.dump(variation_descriptions, f)

    print('Done!')