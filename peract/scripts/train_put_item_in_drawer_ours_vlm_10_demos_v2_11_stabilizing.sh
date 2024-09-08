seed=11
master_port=28501

python ../train.py \
    method=PERACT_BC \
    rlbench.tasks=[put_item_in_drawer] \
    rlbench.task_name=put_item_in_drawer_10_demos_ours_vlm_v2_${seed}_stabilizing \
    rlbench.cameras=[front,wrist,wrist2] \
    rlbench.demos=10 \
    rlbench.demo_path=$PERACT_ROOT/data/train/put_item_in_drawer_10_demos_corl_v3 \
    rlbench.scene_bounds=[-0.8,-1.0,0.8,1.2,1.0,2.8] \
    replay.batch_size=1 \
    replay.path=/tmp/replay_${master_port} \
    replay.max_parallel_processes=16 \
    method.voxel_sizes=[50] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=2048 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
    method.pos_encoding_with_lang=True \
    method.which_arm=assistive \
    method.crop_target_obj_voxel=True \
    method.crop_radius=0.4 \
    method.arm_pred_loss=True \
    framework.training_iterations=1000000 \
    framework.num_weights_to_keep=100 \
    framework.start_seed=${seed} \
    framework.log_freq=1000 \
    framework.save_freq=10000 \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=False \
    framework.wandb_logging=True \
    framework.load_existing_weights=True \
    ddp.num_devices=1 \
    ddp.master_port=${master_port}