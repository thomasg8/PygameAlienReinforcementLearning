import math

def collision(state, n):
    """returns true if boulder collides with alien"""
    for i in range(n):
        p=False
        if state[0:n][i]==state[-n:][i] and state[0:n][i]=='1':
            return True
    return False

def bottom_row_rewards(full_states, n):
    """returns the reward for only the bottom row
    1 for not collision, -inf for collisison"""
    next_row_only = list(set([a[:n]+a[-n-n:-n] for a in full_states]))
    collisions=[]
    for state in next_row_only:
        if collision(state, n):
            collisions.append(state)

    state_reward={}
    for state in next_row_only:
        if state in collisions:
            state_reward[state] = -1000
        else:
            state_reward[state] = 1
    return next_row_only, collisions, state_reward

def bottom_row_movement(next_row_only, n):
    """Defines movement logic and returns results of 3 actions RLU"""
    movement_states={}
    for state in next_row_only:
        empty_R=['0']*n; empty_L=['0']*n
        a_index=list(state[:n]).index('1')
        if a_index==0:
            empty_R[1]='1'; right=''.join(empty_R)
            left=state[:n]
        elif a_index==n-1:
            right=state[:n]
            empty_L[a_index-1]='1'; left = ''.join(empty_L)
        else:
            empty_R[a_index+1]='1'; right=''.join(empty_R)
            empty_L[a_index-1]='1'; left=''.join(empty_L)
        movement_states[state] = [right+state[n:], left+state[n:], state] #RLU
    return movement_states

def bottom_row_movement_rewards(movement_states, state_reward, n):
    """Assigns rewards for each action RLU"""
    movement_rewards_partial={}
    for k,v_list in movement_states.items():
        m_reward=[]
        for v in v_list:
            m_reward.append(state_reward[v])
        movement_rewards_partial[k]=m_reward
    return movement_rewards_partial

def gen_full_movement_rewards(full_states, movement_rewards_partial, n):
    """Expands to all states"""
    movement_rewards_full={}
    for state in full_states:
        partial_state=state[:n]+state[-n-n:-n]
        movement_rewards_full[state]=movement_rewards_partial[partial_state]
    return movement_rewards_full


def main(n, full_states):
    next_row_only, collisions, state_reward=bottom_row_rewards(full_states, n)
    movement_states = bottom_row_movement(next_row_only, n)
    movement_rewards_partial= bottom_row_movement_rewards(movement_states, state_reward, n)
    movement_rewards=gen_full_movement_rewards(full_states, movement_rewards_partial, n)
    return movement_rewards
