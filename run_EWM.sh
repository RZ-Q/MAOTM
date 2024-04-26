# map_name="5m_vs_6m"
# algo="EWM"
# rollout_steps=3
# agent_rollout_steps=2
# for s in {0,2}
# do
#     CUDA_VISIBLE_DEVICES=$s nohup python3 RODE/src/main.py \
#         --config=$algo --env-config=sc2 \
#         with env_args.map_name=$map_name t_max=5050000 \
#         use_wandb=True rollout_steps=$rollout_steps agent_rollout_steps=$agent_rollout_steps \
#         >${map_name}${s}${rollout_steps}${agent_rollout_steps}".log" 2>&1 &
# done
map_name="3s_vs_5z"
algo="rode"
for s in {1,3}
do
    CUDA_VISIBLE_DEVICES=$s nohup python3 RODE/src/main.py \
        --config=$algo --env-config=sc2 \
        with env_args.map_name=$map_name t_max=2050000 \
        use_tensorboard=True \
        >/dev/null 2>&1 &
done