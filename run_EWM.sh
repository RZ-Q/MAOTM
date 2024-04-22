map_name="MMM2"
algo="EWM"
rollout_steps=1
agent_rollout_steps=1
for s in {0,1}
do
    CUDA_VISIBLE_DEVICES=$s nohup python3 RODE/src/main.py \
        --config=$algo --env-config=sc2 \
        with env_args.map_name=$map_name \
        use_wandb=True rollout_steps=1 agent_rollout_steps=1 \
        >${map_name}${s}${rollout_steps}${rollout_steps}".log" 2>&1 &
done