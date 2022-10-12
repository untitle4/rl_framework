def get_algo_config():
    return {
        "states": {"s1": {"p1": 1, "p2": 0, "p3": 0},
                   "s2": {"p1": 0, "p2": 1, "p3": 0},
                   "s3": {"p1": 0, "p2": 0, "p3": 1}},
        "state_attr": ["p1", "p2", "p3"],
        "env_attr": ["p1", "p2", "p3"],
        "exploration": "epsilon_greedy",
        "actions": ["a1", "a2", "a3", "a4"],
        "transition": {"s1": {"a1": "s2", "a4": "s1"},
                       "s2": {"a2": "s3", "a4": "s2"},
                       "s3": {"a3": "s1", "a4": "s3"}},
        "task_name": "test_algo",
        "reward": "TestReward",
        "explore_config": {
            "distribution": "normal",
            "epsilon": 0.7,
            "learning_rate": 0.1,
            "discount": 0.9,
            "confidence_bound": 0.5,
            "initial_distribution": {'a': 0, 'b': 10}
        },
        "server_credential": "./credential.txt",
        "client_verbose": 0
    }
