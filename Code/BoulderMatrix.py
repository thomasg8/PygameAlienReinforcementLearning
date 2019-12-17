from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import itertools
import random
import math

def gen_positions(n, n_boulders):
    """Generates state codes for boulders. Includes empty rows
    Parameters:
        n: number of rows/columns
        n_boulders: number of boulders per row
    return value:
        Possible boulder and alien states
    """
    boulder_positions=[]; b_p=[]

    alien_positions_with_0=["{}1{}".format('0'*(n-i-1),'0'*(i)) for i in range(n)]+['0'*n]
    if n_boulders==1:
        return alien_positions_with_0, alien_positions_with_0[0:n]
    else:
        positions=[]
        position_index=list(itertools.combinations(range(n), n_boulders))
        for tup in position_index:
            pos=''
            for i in range(n):
                if i in tup:
                    pos+='1'
                else:
                    pos+='0'
            positions.append(pos)
        if '0'*n not in boulder_positions:
            positions.append('0'*n)
        return positions, alien_positions_with_0[0:n]

def gen_state_positions(n, n_boulders, boulder_positions):
    """Generates all possible states of boulders and alien
    Parameters:
        n: number of rows/columns
        n_boulders: number of boulders per row
        boulder_positions: possible positions of boulders in row
            ex. '10010' would be a boulder in the 1st and 4th column.
    """
    boulder_states=[]; b_p=[]
    for item in itertools.product(boulder_positions, repeat=n):
        boulder_states.append(''.join(item))
    for b in boulder_states:
        if b.count('1')<=n*n_boulders:
            b_list=[int(str(int(b[i:i+n]))[0]) for i in range(0, len(b), n)]
            keep=True; first_i=b_list[0]
            for a in range(1,n):
                if first_i==0 and b_list[a]==1:
                    keep=False
                first_i=b_list[a]
            if keep:
                b_p.append(b)
        else:
            b_p.append(b)
    return b_p


def create_reference_dictionaries(state_space):
    """Creates reference dictionaries for future referall"""
    state_to_index={}
    index_to_state={}
    for i in range(len(state_space)):
        state_to_index[state_space[i]]=i
        index_to_state[i]=state_space[i]
    return state_to_index, index_to_state

def boulder_down(n, state_space, state_to_index, boulder_positions):
    """Moves boulder down one position. Creates dictionary of all possible locations of the new boulders"""
    b_down={}
    for state in state_space:
        b_list=[]
        b_pos=state[:-n] # removes the bottom row

        for opt in boulder_positions:
            if opt!='0'*n:
                new_state=opt+b_pos
                b_list.append(state_to_index[new_state])
        b_down[state_to_index[state]]=b_list
    return b_down

def create_B(n, n_boulders, b_down, state_space, boulder_positions):
    """Creates sparse matrix with movement probabilities from state space"""
    #state_matrix=np.zeros([len(state_space),len(state_space)])
    print('Size:',(len(state_space), len(state_space)))
    state_matrix=lil_matrix((len(state_space), len(state_space)))
    prob=1/(len(boulder_positions)-1)
    for k,v in b_down.items():
        state_matrix[k,v]=prob
    return csr_matrix(state_matrix)

def endgame_states(n, alien_positions, state_space):
    full_states=[]
    for a_state in alien_positions:
        for b_state in state_space:
            full_states.append(a_state+b_state)
    collisions=[]
    for state in full_states:
        for i in range(n):
            p=False
            if state[0:n][i]==state[-n:][i] and state[0:n][i]=='1':
                p=True
            if p:
                collisions.append(state)
    collisions=list(set(collisions))
    state_reward={}
    for state in full_states:
        if state in collisions:
            state_reward[state] = -5
        else:
            state_reward[state] = 1

    return collisions, state_reward, full_states

def main(n, n_boulders):
    """Takes n and n boulders and creates B matrix"""
    if n>=5 and n_boulders>=2:
        print('Warning: Large environment. Expect delays')
    boulder_positions, alien_positions=gen_positions(n,n_boulders)
    state_space=gen_state_positions(n,n_boulders,boulder_positions)
    state_to_index, index_to_state=create_reference_dictionaries(state_space)
    b_down=boulder_down(n, state_space, state_to_index, boulder_positions)
    collisions, state_reward, full_states=endgame_states(n, alien_positions, state_space)
    if n>=5 and n_boulders>=2:
        print('Creating boulder matrix...')
    B=create_B(n, n_boulders, b_down, state_space, boulder_positions)
    return B, b_down, state_to_index, index_to_state, state_space, boulder_positions, alien_positions, collisions, state_reward, full_states
