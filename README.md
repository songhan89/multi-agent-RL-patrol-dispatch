# Schedule 

24 Jul - finish implementation of environment and actions

What is the task required ? 

- Read in data points from scenario 

How should we create the environment ?

* A hidden initial schedule
* A hidden current schedule that is not part of the observation state 
* Observation space of N sectors + 1 , + 1 is a dummy node if the patrol agent is travelling
* At each state, one agent occupies one of the array 
* After each step, fill in the current schedule
* Done when `T = 72`

### Things to explore
#### Reward function 
* should i set zero , and calculate the reward when reach terminal state ?
* Or should i give incremental reward each step along the way

#### Action Space
* When N sector is small, it makes more sense to let action = |N| sectors, this helps to avoid any constraint imposed 
by dispatch rules
* We may need to explore dispatch rules benefit vs `free learning` approach


30 Jul - run a simple single action 