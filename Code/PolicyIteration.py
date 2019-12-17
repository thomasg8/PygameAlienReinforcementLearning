import BoulderMatrix, MovementRewards, numpy as np, pygame, random, json
from scipy.sparse import csr_matrix, lil_matrix
from progress.bar import Bar

def alien_movement(alien_p, n, d):
    """Moves alien from one state to another.
    Parameters:
        alien_p: alien position. Ex. 100 is an alien in the leftmost column of 3
        n: number of rows/columns
        d: direction of movement
    Returns: Alien state
    """
    empty_R=['0']*n; empty_L=['0']*n
    a_index=list(alien_p).index('1')
    if a_index == 0:
        empty_R[1]='1'; right=''.join(empty_R)
        left=alien_p
    elif a_index == n-1:
        right = alien_p
        empty_L[a_index-1]='1'; left = ''.join(empty_L)
    else:
        empty_R[a_index+1]='1'; right=''.join(empty_R)
        empty_L[a_index-1]='1'; left=''.join(empty_L)

    if d == 'R':
        return right
    elif d == 'L':
        return left
    else:
        return alien_p

def movement_tuples(state, action, n, state_down, boulder_positions, state_reward):
    """Generates movement details for a state.
    Parameters:
        state: full alien boulder states
        action: right, left, up in either integer or string
        n: number of rows/columns
        state_down: possible future states based on movement
        boulder_positions: possible boulder positions
        state_reward: 1 if alive, -5 if dead
    Returns: tuple of prob, next_state, reward
    """
    if type(action)==int:
        action_values = {0:'R', 1:'L', 2:'U'}
        action = action_values[action]
    # moves alien and determines possible new states
    a_s = alien_movement(state[:n], n, action); p_b_s = state_down[state[n:]]
    p_states = [a_s+b_s for b_s in p_b_s] # combines into full states
    prob = 1/(len(boulder_positions)-1) # prob of each state
    return [(prob, ns, state_reward[ns]) for ns in p_states]

def policy_eval(policy, nS, nA, P, full_state_to_index, gamma=1, theta=0.05):
    """Evaluate a policy.
    Parameters:
        policy: [S, A] matrix. Each row is a state and each col is an action
        P[s][a]: is list of transitional tuples (prob, next_state, reward)
        nS: number of states in environment
        nA: number of actions
        theta: stop evaluation once function change is less than or equal to this value
        gamma: discount factor

    returns: vector of length nS representing value function
    """
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            value = 0
            # for each action, iterate over possible next states
            for a, a_prob in enumerate(policy[s]):
                for prob, ns, reward in P[s][a]:
                    # calculate expected value
                    value += a_prob * prob * (reward+gamma*V[full_state_to_index[ns]])
            delta = max(delta, np.abs(value-V[s]))
            V[s] = value
        if delta <= theta: #end condition
            break
    return V

def value(s, V, full_state_to_index, nA, P, gamma = 1, theta=0.05):
    """Helper function that calculates value for all actions for a given state.
    Parameters:
        s: state index
        V: Value
        full_state_to_index: dictionary of state to index Values
        nA: number of actions (3)
        P: transitional tuples given state and action
        gamma: discount factor
        theta: stopping conditions
    Returns: A"""
    A = np.zeros(nA)
    for a in range(nA):
        for prob, ns, reward in P[s][a]:
            A[a] += prob * (reward + gamma * V[full_state_to_index[ns]])
    return A

def policy_improvement(nS, nA, P, full_state_to_index, g=.75,t=0.05):
    """Iteratively evaluates and improves a policy until an optimal policy is found
    or reaches threshold of iterations
    Parameters:
        nS: number of states
        nA: number of actions
        P: transitional tuples given state and action
        full_state_to_index: dictionary of state to index Values
        g: gamma which is discount factor
        t: theta or stopping condition
    Returns: tuple of policy and value of policy
    """
    policy = np.ones([nS, nA]) / nA # random policy (equal chance all actions)

    i=0
    while True:
        i+=1
        if i%100==0:
            print(i)
        V = policy_eval(policy, nS, nA, P, full_state_to_index,  gamma=g, theta=t) # eval current policy
        is_policy_stable = True # true is no changes false if we make changes

        for s in range(nS):
            chosen_a = np.random.choice(np.argwhere(policy[s] == np.amax(policy[s])).flatten().tolist())
            action_values = value(s, V, full_state_to_index, nA, P, gamma=g, theta=t)
            best_a = np.random.choice(np.argwhere(action_values == np.amax(action_values)).flatten().tolist())
            if chosen_a != best_a: # greedy update
                is_policy_stable = False
            policy[s] = np.eye(nA)[best_a]
        if is_policy_stable or i==10000:
            print(i, 'Iterations')
            return policy, V

def main(n, n_boulders, b_down, state_to_index, index_to_state, boulder_positions, state_reward, full_states):
    """Finds and saves optimal policy and values to json file.
    Parameters:
        n: number of rows/columns
        b_down: possible next boulder states
        state_to_index/index_to_state: dictionaries to convert states to index
        boulder_positions: possible boulder states
        state_reward: reward of moving to state
        full_states: all possible states given n and n_boulders
    Returns: None
    """
    actions = [0,1,2]; state_down = {}; full_state_to_index={}; nS = len(full_states); nA = len(actions)
    # gen conversion dictionaries
    for k,v in b_down.items():
        vals = [index_to_state[a] for a in v]
        state_down[index_to_state[k]] = vals
    for s in range(len(full_states)):
        full_state_to_index[full_states[s]] = s
    # create P
    P=[]
    for state in full_states:
        action_list=[]
        for a in actions:
            action_list.append(movement_tuples(state,a, n, state_down, boulder_positions, state_reward))
        P.append(action_list)

    policy, v = policy_improvement(nS, nA, P, full_state_to_index, g = .5)
    policy_actions = [int(np.argmax(p)) for p in policy]
    data = {'Policy':policy_actions, 'Values':v.tolist()}

    with open('PolicyIterationResults/data{}_{}.json'.format(n,n_boulders), 'w') as f:
        json.dump(data, f)

    print(policy_actions.count(0),policy_actions.count(1),policy_actions.count(2))
#main(n, n_boulders, b_down, state_to_index, index_to_state, boulder_positions, state_reward, full_states)
