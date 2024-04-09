# ours
# ================================================================ for starcraft II  ================================================================
# map_name = [1c3s5z, 2s_vs_1sc, 2s3z, 3a5z 10m_vs_11m, 2c_vs_64zg, 3a_vs_5z, 5m_vs_6m, MMM2, 6h_vs_8z, corridor, 3s5z_vs_3s6z]
# env_diff = ["easy","easy","easy","easy","easy","hard","hard","hard", "super hard", "super hard", "super hard", "super hard"]
# for super hard maps [6h_vs_8z, corridor, 3s5z_vs_3s6z] epsilon_anneal_time=1000000 t_max=5010000, for others epsilon_anneal_time=50000 t_max=2050000

# ================================================================ for predator-prey  ================================================================
# miscapture_punishment = [0, -0.5, -1.0, -2.0]
# t_max = 1000000

# ## mbom (Multi-Agent Opponent Trajectory Modeling, MA-OTM)    mixer=vdn/qmix
# python src/main.py --config=mbom --env-config=sc2 with env_args.map_name=2c_vs_64zg n_rollout_steps=3    # for starcraft II
# python src/main.py --config=mbom --env-config=stag_hunt with env_args.miscapture_punishment=-2.0 n_rollout_steps=1    # for predator-prey

# ## rbom (Role-based Decomposition for Opponent Modeling in Multi-Agent Reinforcement Learning, MA-RDOM)   mixer=vdn/qmix
# python src/main.py --config=rbom --env-config=sc2 with env_args.map_name=2c_vs_64zg n_role_clusters=3     # starcraft II


# baselines
# python src/main.py --config=dpiqn --env-config=sc2 with env_args.map_name=2c_vs_64zg
# python src/main.py --config=vdn --env-config=sc2 with env_args.map_name=2c_vs_64zg
# python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2c_vs_64zg
# python src/main.py --config=qplex --env-config=sc2 with env_args.map_name=2c_vs_64zg

# python src/main.py --config=dpiqn --env-config=stag_hunt with env_args.miscapture_punishment=-2.0



