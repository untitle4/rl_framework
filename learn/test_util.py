def get_config():
    return {
        "states": {"s1": {"p1": 1, "p2": 1, "p3": 1}, "s2": {"p1": 1, "p2": 1, "p3": 2}},
        "state_attr": ["p1", "p2", "p3"],
        "env_attr": ["e1", "e2", "e3"],
        "exploration": "epsilon_greedy",
        "actions": {"a": {"p3": 1}, "b": {"p3": -1}},
        "transition": {"s1": {"a": "s2"}, "s2": {"b": "s1"}},
        "task_name": "test",
        "reward": "SampleReward",
        "explore_config": {
            "distribution": "beta",
            "epsilon": 0.6,
            "learning_rate": 0.1,
            "discount": 0.9
        },
        "server_credential": "./credential.txt"
    }