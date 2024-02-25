# MAOTM
Multi-Agent Opponent Trajectory Modeling

1. this folder is used to train RL agent, including (dpiqn / mbom / rbom) with (qplex, vdn, qmix);
where dpiqn, mbom and rbom are represented as drpiqn, MA-OTM and MA-RDOM in the paper, respectively.

2. please install all packages and sc2 properly in the beginning (https://github.com/oxwhirl/pymarl/);

3. 'python ...' commands can be seen in run_exp.sh

4. uncomment 'sys.argv' in main.py to debug

- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
- The results in SMAC (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232 not SC2.4.10.
