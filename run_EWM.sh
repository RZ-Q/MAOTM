map_name="5m_vs_6m"
algo="EWM"
for s in {0,1,2,3,4}
do
    CUDA_VISIBLE_DEVICES=$s nohup python3 RODE/src/main.py \
        --config=$algo --env-config=sc2 \
        with env_args.map_name=$map_name \
        use_wandb=True \
        >0.log 2>&1 &
done