import pygame, random
import numpy as np
import BoulderMatrix

class Policies(object):
    def random_movement(boulder_state, alien_state, movement_rewards, RL_Policies, full_states):
        """Baseline random movement policy"""
        events_options=[pygame.K_RIGHT, pygame.K_LEFT, pygame.K_UP]
        ri=random.randint(0,2)

        return events_options[ri]
    def greedy(boulder_state, alien_state, movement_rewards, RL_Policies, full_states):
        """Greedy means that it will always take the step to maximize the next moves reward"""
        events_options=[pygame.K_RIGHT, pygame.K_LEFT, pygame.K_UP]
        state=alien_state+boulder_state
        if random.random() > 0:
            rewards=np.array(movement_rewards[state])
            try:
                argmax_indexes=np.where(rewards == 1)[0]
                movement_index=np.random.choice(np.array(argmax_indexes)) # randomly selects from argmax
                return events_options[movement_index]
            except Exception as e:
                return np.random.choice(events_options)
        else:
            return np.random.choice(events_options)


    def reinforcement_learning(boulder_state, alien_state, movement_rewards, RL_Policies, full_states):
        """RL using value iteration"""
        alien_state_index, alien_index_state=BoulderMatrix.create_reference_dictionaries(full_states)
        events_options=[pygame.K_RIGHT, pygame.K_LEFT, pygame.K_UP]
        state = alien_state + boulder_state
        RL_i = alien_state_index[state]
        movement = int(RL_Policies[RL_i])
        return events_options[movement]
