import collections


def get_state_transition():
    light_time_range = [20, 60]
    interval = 5
    actions = ["K", "I", "D"]
    states = {}
    for i in range(int((light_time_range[1] - light_time_range[0]) / interval) + 1):
        states[str(i)] = {'g': light_time_range[0] + i * interval}
    transition = collections.defaultdict(dict)

    for state in states.keys():
        for action in actions:
            if action == 'K':
                transition[state][action] = state
            elif action == 'I':
                if states[state]['g'] == 60:
                    continue
                else:
                    transition[state][action] = str(int(state) + 1)
            else:
                if states[state]['g'] == 20:
                    continue
                else:
                    transition[state][action] = str(int(state) - 1)
    return transition, states


def get_compose_state_transition():
    length_range = [20, 60]
    actions = ["K", "I", "D"]
    interval = 5

    init = (20, 20, 20)

    state_map = {'1': init}
    state_inverse_map = {init: '1'}
    transition = collections.defaultdict(dict)

    queue = []
    visited = set()
    visited.add(init)
    queue.append(init)

    delta = {'I': interval, 'K': 0, 'D': -interval}
    count = 1

    while len(queue) > 0:
        curr = queue.pop(0)
        candidate = {}
        for a1 in actions:
            for a2 in actions:
                for a3 in actions:
                    s1, s2, s3 = curr
                    if (a1 == 'I' and s1 == length_range[1]) or (a1 == 'D' and s1 == length_range[0]):
                        continue
                    if (a2 == 'I' and s2 == length_range[1]) or (a2 == 'D' and s2 == length_range[0]):
                        continue
                    if (a3 == 'I' and s3 == length_range[1]) or (a3 == 'D' and s3 == length_range[0]):
                        continue
                    s1 = s1 + delta[a1]
                    s2 = s2 + delta[a2]
                    s3 = s3 + delta[a3]

                    new_state = (s1, s2, s3)
                    candidate[a1 + a2 + a3] = new_state

        for compose_action in candidate.keys():
            new_state = candidate[compose_action]
            if new_state not in visited:
                count += 1
                state_map[str(count)] = new_state
                state_inverse_map[new_state] = str(count)
                transition[state_inverse_map[curr]][compose_action] = state_inverse_map[new_state]
                queue.append(new_state)
                visited.add(new_state)
            else:
                transition[state_inverse_map[curr]][compose_action] = state_inverse_map[new_state]

    states = {s_id: {'g1': state_map[s_id][0], 'g2': state_map[s_id][1], 'g3': state_map[s_id][2]}
                  for s_id in state_map.keys()}

    return transition, states
