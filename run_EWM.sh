map_name="2c_vs_64zg"
algo="EWM"
for s in {0,3}
do
    CUDA_VISIBLE_DEVICES=$s nohup python3 RODE/src/main.py \
        --config=$algo --env-config=sc2 \
        with env_args.map_name=$map_name \
        use_wandb=True \
        >/dev/null 2>&1 &
done