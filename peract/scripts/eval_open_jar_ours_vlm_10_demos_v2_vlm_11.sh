# NOTE: 'framework.eval_type' denotes the acting policy's checkpoint, and 'framework.left_arm_ckpt' denotes the stabilizing policy's checkpoint 
# validation step 1: evalaute all acting checkpoints in the weights folder using the latest stabilizing checkpoint
seed=11
# CUDA_VISIBLE_DEVICES=0 python ../eval.py \
#     rlbench.tasks=[open_jar] \
#     rlbench.task_name=open_jar_10_demos_ours_vlm_v2_${seed}_acting \
#     rlbench.cameras=[front,wrist,wrist2] \
#     rlbench.demo_path=$PERACT_ROOT/data/val/open_jar_25_demos_ours_corl_v1 \
#     rlbench.scene_bounds=[-0.8,-1.0,0.1,1.2,1.0,2.1] \
#     framework.gpu=0 \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.left_arm_ckpt=$PERACT_ROOT/logs/open_jar_10_demos_ours_vlm_v2_${seed}_stabilizing/PERACT_BC/seed${seed}/weights/990000/QAttentionAgent_layer0.pt \
#     framework.left_arm_train_cfg=$PERACT_ROOT/logs/open_jar_10_demos_ours_vlm_v2_${seed}_stabilizing/PERACT_BC/seed${seed}/config.yaml \
#     framework.start_seed=${seed} \
#     framework.eval_envs=1 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=25 \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=False \
#     framework.eval_type=missing \
#     method.which_arm=dominant_assistive \
#     method.no_voxposer=True \
#     rlbench.headless=True

# validation step 2: evalaute all stabilizing checkpoints in the weights folder using the best performing acting checkpoint
# CUDA_VISIBLE_DEVICES=0 python ../eval.py \
#     rlbench.tasks=[open_jar] \
#     rlbench.task_name=open_jar_10_demos_ours_vlm_v2_${seed}_acting \
#     rlbench.cameras=[front,wrist,wrist2] \
#     rlbench.demo_path=$PERACT_ROOT/data/val/open_jar_25_demos_ours_corl_v1 \
#     rlbench.scene_bounds=[-0.8,-1.0,0.1,1.2,1.0,2.1] \
#     framework.gpu=0 \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.left_arm_ckpt=$PERACT_ROOT/logs/open_jar_10_demos_ours_vlm_v2_${seed}_stabilizing/PERACT_BC/seed${seed}/weights \
#     framework.left_arm_train_cfg=$PERACT_ROOT/logs/open_jar_10_demos_ours_vlm_v2_${seed}_stabilizing/PERACT_BC/seed${seed}/config.yaml \
#     framework.start_seed=${seed} \
#     framework.eval_envs=1 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=25 \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=False \
#     framework.eval_type=490000 \
#     method.which_arm=dominant_assistive \
#     method.no_voxposer=True \
#     rlbench.headless=True

# test eval: use the best acting and stabilizing checkpoints
CUDA_VISIBLE_DEVICES=0 python ../eval.py \
    rlbench.tasks=[open_jar] \
    rlbench.task_name=open_jar_10_demos_ours_vlm_v2_${seed}_acting \
    rlbench.cameras=[front,wrist,wrist2] \
    rlbench.demo_path=$PERACT_ROOT/data/test/open_jar_25_demos_ours_corl_v1 \
    rlbench.scene_bounds=[-0.8,-1.0,0.1,1.2,1.0,2.1] \
    framework.gpu=0 \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.left_arm_ckpt=$PERACT_ROOT/logs/open_jar_10_demos_ours_vlm_v2_${seed}_stabilizing/PERACT_BC/seed${seed}/weights/560000/QAttentionAgent_layer0.pt \
    framework.left_arm_train_cfg=$PERACT_ROOT/logs/open_jar_10_demos_ours_vlm_v2_${seed}_stabilizing/PERACT_BC/seed${seed}/config.yaml \
    framework.start_seed=${seed} \
    framework.eval_envs=1 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=25 \
    framework.csv_logging=True \
    framework.tensorboard_logging=False \
    framework.eval_type=490000 \
    method.which_arm=dominant_assistive \
    method.no_voxposer=True \
    rlbench.headless=True \
    cinematic_recorder.enabled=True