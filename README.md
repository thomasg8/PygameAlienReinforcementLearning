# PygameReinforcementLearning
This project uses reinforcement learning to train an agent to play a custom game built in pygame. The game is a turn based game with an alient agent that has to dodge falling meteorites. The alien can only move left, none/up, or right, while the meteorites move down one row every turn. An example turn of the game is shown in the figure below.

<a href="url"><img src="https://github.com/thomasg8/PygameAlienReinforcementLearning/blob/master/Code/figs/ExState-1.png" align="left" height="256" width="256" ></a>

The agent uses 3 policies to play this game:
1. **Random:** Randomly selects action from the action set.
2. **Greedy:** The agent can 'see' one row ahead. Takes action to maximize the next state reward. Ties are broken randomly.
3. **Reinforcement Learning:** The agent can 'see' the entire environment. Takes action determined by policy iteration to maximize future reward.

After 10,000 simulations per method, the Random, Greedy, and Reinforcement Learning policies averaged 3.98, 16.55, and 45.86 turns, respectively. For these simulations, the visuals were removed to decrease run time. To see the trained agent's performance on 10 simulations for yourself, run SimulationWithVisual.py. 


This project was created under the advisement of Dr. Bonifonte during the Spring semester of 2019 at Denison University. For more details, please read the corresponding paper titled ReinforcementLearninginPygame.pdf.



