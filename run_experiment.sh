python train.py --ename ./experiments/thompson_run_0 --lconfig ./configs/learning_config.json

python train.py --ename ./experiments/upper_confidence_bound_run_0 --lconfig ./configs/learning_config_ucb.json

python train.py --ename ./experiments/epsilon_greedy_run_0 --lconfig ./configs/learning_config_ep.json

python train_compose.py --ename ./experiments/thompson_run_compose_0 --lconfig ./configs/learning_config.json

python train_compose.py --ename ./experiments/upper_confidence_bound_compose_run_0 --lconfig ./configs/learning_config_ucb.json

python train_compose.py --ename ./experiments/epsilon_greedy_compose_run_0 --lconfig ./configs/learning_config_ep.json