map_name="5m_vs_6m"
algo="EWM"
rollout_steps=3
agent_rollout_steps=2
t_max=2050000
q_lambda=1.0
world_model_utd=3
world_model_buffer_size=100
n_layer=1
for s in {0,6}
do
    CUDA_VISIBLE_DEVICES=$s nohup python3 RODE/src/main.py \
         --config=$algo --env-config=sc2 \
         with env_args.map_name=$map_name t_max=$t_max \
         use_wandb=True rollout_steps=$rollout_steps agent_rollout_steps=$agent_rollout_steps \
         q_lambda=$q_lambda world_model_utd=$world_model_utd \
         world_model_buffer_size=$world_model_buffer_size n_layer=$n_layer \
         epsilon_anneal_time=50000 epsilon_anneal_time_exp=50000 \
         >/dev/null 2>&1 &
done