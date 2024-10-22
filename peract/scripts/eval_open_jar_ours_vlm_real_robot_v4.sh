CUDA_VISIBLE_DEVICES=0 python ../eval_real.py \
    rlbench.tasks=[open_jar] \
    rlbench.task_name='open_jar_ours_vlm_real_robot_v4_acting' \
    rlbench.cameras=[front] \
    rlbench.demo_path=$PERACT_ROOT/data/train/open_jar_2024_06_14_v1_acting_stabilizing_trimmed \
    rlbench.scene_bounds=[-1.0,-1.0,-0.5,1.0,1.0,1.5] \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.left_arm_ckpt=$PERACT_ROOT/logs/open_jar_ours_vlm_real_robot_v4_stabilizing/PERACT_BC/seed0/weights/240000/QAttentionAgent_layer0.pt \
    framework.left_arm_train_cfg=$PERACT_ROOT/logs/open_jar_ours_vlm_real_robot_v4_stabilizing/PERACT_BC/seed0/config.yaml \
    framework.start_seed=0 \
    framework.eval_envs=1 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=10 \
    framework.csv_logging=True \
    framework.tensorboard_logging=False \
    framework.eval_type=390000 \
    framework.gpu=cpu \
    method.which_arm=dominant_assistive \
    method.no_voxposer=True \
    rlbench.headless=True