map_name="3s_vs_5z"
algo="EWM"
rollout_steps=3
agent_rollout_steps=1
for s in {0,1}
do
    CUDA_VISIBLE_DEVICES=$s nohup python3 RODE/src/main.py \
        --config=$algo --env-config=sc2 \
        with env_args.map_name=$map_name \
        use_wandb=True rollout_steps=$rollout_steps agent_rollout_steps=$agent_rollout_steps \
        >${map_name}${s}${rollout_steps}${agent_rollout_steps}".log" 2>&1 &
done