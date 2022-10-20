import collections


def get_state_transition():
    light_time_range = [20, 60]
    interval = 5
    actions = ["K", "I", "D"]
    states = {}
    for i in range(light_time_range[1] - light_time_range[0]):
        states[str(i)] = {'g': light_time_range[0] + i * interval}
    transition = collections.defaultdict(dict)

    for state in states.keys():
        for action in actions:
            if action == 'K':
                transition[state][action] = state
            elif action == 'I':
                if states[state]['g'] == 40:
                    continue
                else:
                    transition[state][action] = str(int(state) + 1)
            else:
                if states[state]['g'] == 20:
                    continue
                else:
                    transition[state][action] = str(int(state) - 1)
    return transition, state





