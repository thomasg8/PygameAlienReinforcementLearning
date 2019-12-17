from policies import Policies
from scipy.sparse import lil_matrix
from progress.bar import Bar
import BoulderMatrix, MovementRewards, json, pygame, sys, numpy as np, random, time, datetime, os

# Globals for easy testing

n=4 # number of rows/columns
n_boulders=2 # number of boulders/meteorites per row
n_iterations= 10000 # number of iterations in the simulation
delay=0 # adds delay between turns.... artifact of visual simulation

# Calculate state matrices

B, b_down, state_to_index, index_to_state, state_space, boulder_positions, alien_positions, collisions, state_reward, full_states = BoulderMatrix.main(n,n_boulders)
movement_rewards = MovementRewards.main(n, full_states)

try:
    fh = open('PolicyIterationResults/data{}_{}.json'.format(n,n_boulders), 'r') # tries to read in policy

except FileNotFoundError: # if fails, optimizes policy for n, n_boulders combinations
    import PolicyIteration
    print("Optimizing Policy")
    PolicyIteration.main(n, n_boulders, b_down, state_to_index, index_to_state, boulder_positions, state_reward, full_states)

# initialize pygame with a screen size of 600 by 600
#pygame.init()
#screen=pygame.display.set_mode((600, 600))

def calculate_indices(n):
    """Calculates the indices for presentation assuming screen size of 600x600.
    This determines where the boulders and alien will appear on screen.
    Parameters:
        n: number of rows/columns. Range(3,7)
    Returns:
        dictionary of the screen coordinates
    """
    x_y_index=[[int((600/n-50)/2+600/n*i) for i in range(n)] for n in range(3,7)] # calculates indices range [3,6]
    p_values={}
    for i in range(n):
        p_values[i]=x_y_index[n-3][i]
    return p_values

class Background(object):
    def __init__(self):
        """Displays background image on screen of size 600, 600"""
        self.image = pygame.transform.scale(pygame.image.load('figs/icon.png'),(600,600))
        screen.blit(self.image, (0, 0))

class Boulder(object):
    def __init__(self, tup):
        """Given cordinates, display image of meteorite/boulder
        Parameters:
            tup: tuple of x,y coordinates to display boulder/meteorite image
        Returns: None
        """
        self.x=tup[0]; self.y=tup[1]
        #self.image = pygame.transform.scale(pygame.image.load('figs/meteorite.png'),(50,50))
        #screen.blit(self.image, (self.x, self.y))

def T_Alien(n, B):
    """Creates alien transition matrix for n rows and columns
    Parameters:
        n: number of rows/columns
        B: Boulder transition matrix
    Returns: Alien transition matrix
    """
    empty=lil_matrix((len(state_space), len(state_space))) # sparse matrix for 0s
    # matrix of boulder transition matrix with 0s for non-alien locations
    t=[[[empty for row in range(n)] for column in range(n)] for action in range(3)]
    for i in range(1,n-1):
        t[0][i][i+1]=B #right
        t[1][i][i-1]=B #left
        t[2][i][i]=B #up
    # End cases
    t[0][0][1]=B; t[0][n-1][n-1]=B # right
    t[1][0][0]=B; t[1][n-1][n-2]=B # left
    t[2][0][0]=B; t[2][n-1][n-1]=B # up
    return t

class Alien(object):
    """This class handles the movement and creation of the alien agent.
    Attributes:
        draw: adds alien image to screen
        move: moves the alien from state to state
    """
    def __init__(self, p_values):
        """Creates alien agent
        """
        self.y = p_values[n-1]
        position=random.randint(0,n-1); self.position=position
        self.x=p_values[position]
        #self.image = pygame.transform.scale(pygame.image.load('figs/alien.png'),(50,50))
        #screen.blit(self.image, (self.x, self.y))
        state_l=['0']*n; state_l[position]='1' #quick alien state generation
        self.state=''.join(state_l)

    def draw(self):
        """Displays alien image"""
        screen.blit(self.image, (self.x, self.y))

    def move(self, event, p_values, T):
        """Moves the alien right, left or right given an action
        Parameters:
            event: pygame right, left, or up action
            p_values: dictionary of display coordinates
            T: Alien transition matrix
        """
        p_index = {v: k for k, v in p_values.items()} # invert dictionary
        movement={pygame.K_RIGHT:0, pygame.K_LEFT:1, pygame.K_UP:2} # action options
        d=movement[event] # converts action to index
        current_location=p_index[self.x]
        # Determine next position
        empty_check=[T[d][current_location][i].count_nonzero() for i in range(n)]
        new_position=int(np.nonzero(empty_check)[0])
        # update alien agent
        self.x=p_values[new_position]; self.position=new_position
        state_l=['0']*n; state_l[self.position]='1'
        self.state=''.join(state_l)
        # display alien agent
        #screen.blit(self.image, (self.x, self.y))

def update_boulders(state, p_matrix):
    """Adds all boulders in a given state to the proper positions on the screen
    Parameters:
        State: Current boulder states
        p_matrix: position matrix
    Returns: matrix of boulders
    """
    b_matrix=np.matrix([[int(a) for a in list(b)] for b in [state[i:i+n] for i in range(0,n*n,n)]])
    for x in range(0,n):
        for y in range(0,n):
            if b_matrix[x,y]==1:
                loc=p_matrix[x,y]
                Boulder(loc)
    return b_matrix


def GameLoop(policy):
    """Plays alien-meteorite game where the agent tries to stay alive.
    Parameters:
        policy: Random, Greedy, or RL which determines what policy for the agent to use.
            Random: Randomly chooses a movement direction
            Greedy: Chooses optimal direction only given next row (ties randomly resolved)
            RL: Reinforcement Learning policy which chooses optimal direction based on entire state
    Returns: Score and state of death/collision
    """
    state_list=[]
    with open('PolicyIterationResults/data{}_{}.json'.format(n,n_boulders)) as f: # RL generated policy
        data = json.load(f)
        RL_Policies = data['Policy']
    p_values=calculate_indices(n)
    p_matrix=np.matrix([[(x,y) for x in list(p_values.values())] for y in list(p_values.values())], dtype=np.dtype('int,int'))
    boulder_state=state_space[-1] # will be '0'*n so no boulders to start
    #Background() # updates screen with space background
    alien=Alien(p_values) # initializes alien agent
    T=T_Alien(n, B)

    if policy=="Random":
        policy_func = Policies.random_movement
    elif policy=="Greedy":
        policy_func = Policies.greedy
    else:
        policy_func = Policies.reinforcement_learning

    running=True; score=0
    while running: # while not dead
        # check for escape or exit action to end game
        #for event in pygame.event.get():
            #if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                #running=False
                #pygame.quit()
                #sys.exit()

        #Background()
        # Determine action based on policy
        event = policy_func(boulder_state, alien.state, movement_rewards, RL_Policies, full_states)
        # update alien position and boudler state with random top row
        alien.move(event, p_values, T)
        b_s=random.randint(0,len(b_down[0])-1)
        boulder_state=index_to_state[b_down[state_to_index[boulder_state]][b_s]]
        b_matrix=update_boulders(boulder_state, p_matrix)
        # test for collision
        full_state=alien.state+boulder_state; state_list.append(full_state)
        if full_state in collisions:
            running=False
        else:
            score+=1
        #pygame.display.update()
        #time.sleep(delay)

    return score, state_list

def Simulations():
    """Runs simulations of random, greedy, and reinforcement_learning policies.
    Parameters: None
    Returns: dictionary of results and exports results.
    """
    results = {}
    for policy in ["Random", "Greedy", "RL"]:
        results[policy] = {}
        policy_score = []; policy_states = []
        # Create progress bar to measure progress
        bar = Bar(policy, max=n_iterations, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        for i in range(n_iterations):
            score, states = GameLoop(policy)
            policy_score.append(score); policy_states.append(states)
            bar.next()
        bar.finish()
        results[policy]['Scores'] = policy_score
        results[policy]['States'] = policy_states

        print("{} averaged a score of {} over {} turns.".format(policy, sum(policy_score)/n_iterations,n_iterations))
    pygame.quit()
    # Save results in SimulationResults
    now = datetime.datetime.now(); time = now.strftime("%m_%d_%H_%M")
    results['Iterations']=n_iterations
    with open('SimulationResults/data{}{}_{}_{}.json'.format(n,n_boulders, time, n_iterations), 'w') as f:
        json.dump(results, f)
    return results

def main():
    results = Simulations()
    try:
        t=agfg
        import winsound
        winsound.Beep(1500, 1000)
    except:
        pass
main()
