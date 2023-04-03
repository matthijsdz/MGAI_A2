# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import numpy as np
import multiprocessing as mp
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.num_agents = len(gameState.data.agentStates)
    self.depth = 7
    self.M = 20
    self.epsilon=1.0
    self.decay_factor=0.995
    self.min_epsilon=0.1
    #self.distancer.getDistance(p1, p2)
    '''
    Your initialization code goes here, if you need any.
    '''

  def simulate(self, node, depth):
      if depth == 0 or node.state.isOver():
          return self.getScore(node.state), node.getHeuristic();
      actions = node.state.getLegalActions(node.agent_index)
      total_reward = 0
      total_inter_reward=0
      num_sims = min(len(actions), self.M)
      for _ in range(num_sims):
          next_node= self.egreedy_action(actions,node)
          reward,inter_reward = self.simulate(next_node, depth - 1)
          total_reward += reward
          total_inter_reward+=inter_reward
      return total_reward / num_sims, total_inter_reward/num_sims
    
  def egreedy_action(self,actions,node):
    self.epsilon=max(self.decay_factor*self.epsilon, self.min_epsilon)
    max_value=-np.inf
    if(random.uniform(0,1)<self.epsilon):
      action = random.choice(actions)
      next_state =node.state.generateSuccessor(node.agent_index, action)
      next_node = Node(next_state, node.agent_index, node, action)
    else:
      for action in actions:
        temp_state = node.state.generateSuccessor(node.agent_index, action)
        temp_node = Node(temp_state, node.agent_index, node, action)
        next_value = temp_node.getHeuristic()  
        if(next_value >= max_value):
          next_node = temp_node
          max_value = next_value
    return next_node
  
  #Multiprocessing
  def treeExpansion(self,root, gameState):
        node, depth= root.select(self.depth)
        node=node.expand(node.state, node.state.getLegalActions(node.agent_index), self.num_agents)
        full_result,inter_result = self.simulate(node, depth)
        node.backpropagate(full_result,inter_result)
        i=0
  def chooseAction(self, gameState):
    start = time.time()

    root = Node(gameState, self.index)
    
    #with mp.Pool(processes=mp.cpu_count()) as pool:
    for i in range(self.M):
        self.treeExpansion(root,gameState)
        #args=[(root,gameState)] #* mp.cpu_count()
        #pool.apply_async(self.treeExpansion, args=(root, gameState), callback=lambda node: root.children.append(node))
    #threshold = 2
    #root.prune(threshold)
    
    #pool.apply_async(self.treeExpansion, args=(root, gameState), callback=lambda node: root.children.append(node))

    #pool.close()
    #pool.join()
    actions = gameState.getLegalActions(self.index)

    print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    #[print(child.ucb()) for child in root.children]
    #print(actions)
    return max(root.children, key=lambda child: child.inter_w).action
      
class Node:
    def __init__(self, state, agent_index, parent=None, action=None):
        self.state = state
        self.agent_index = agent_index
        self.parent = parent
        self.children = []
        self.unvisited_children=[]
        self.w = 0
        self.n = 1
        self.action = action
        self.inter_w=-np.inf
        self.pos=state.getAgentPosition(agent_index);

    def ucb(self):
        c = np.sqrt(2)
        ucb = self.w / self.n +  c * np.sqrt(np.log(self.parent.n) / self.n)

        return ucb

    def getHeuristic(self):
      #Team Red
      if(self.agent_index%2==0):
          food_matrix=self.state.getBlueFood()
          caps_matrix=self.state.getBlueCapsules()
          opp_ind=self.state.getBlueTeamIndices()
          team_ind=self.state.getRedTeamIndices()
      #Team Blue
      else:
          food_matrix =self.state.getRedFood()
          caps_matrix=self.state.getRedCapsules()
          opp_ind=self.state.getRedTeamIndices()
          team_ind=self.state.getBlueTeamIndices()
      max_distance = food_matrix.width + food_matrix.height
       #Attacking agent
      if(self.agent_index==team_ind[0]):
        c=1.0
        alpha=0.5
        if(self.state.getAgentState(self.agent_index).numCarrying>0):
          beta=-0.4
        else:
          beta=0.4
        beta=0.7
        gamma=0.0
        delta=0.0
        epsilon=0.2
        zeta=0.2
        tau=0.0
      #Defending Agents
      else:
        c=1.0
        alpha=0.4
        if(self.state.getAgentState(self.agent_index).numCarrying>0):
          beta=-0.4
        else:
          beta=0.4
        beta=0.7
        gamma=0.2
        delta=0.2
        epsilon=0.0
        zeta=0.0
        tau=0.3
      food_distance=np.inf
      horizontal_dist=np.inf
      capsule_distance =0
      carrying_food = 0
      scared_ghost_distance = 0
      non_scared_ghost_distance=0
      pacman_distance=0
      myPos = self.state.getAgentPosition(self.agent_index)

      for x in range(food_matrix.width):
        for y in range(food_matrix.height):
          if(food_matrix[x][y]==True):
              food_distance=min((abs(myPos[0]-x)+abs(myPos[1]-y)),food_distance)
              horizontal_dist=min(abs(myPos[0]-x),horizontal_dist)
      for capsule in caps_matrix:
        if(capsule_distance==0):
          capsule_distance=np.inf
          
        capsule_distance= min(capsule_distance, (abs(myPos[0]-capsule[0])+abs(myPos[1]-capsule[1])))
              
      opponents = [self.state.getAgentState(i) for i in opp_ind]
      for opponent in opponents:
        if opponent.isPacman and opponent.numCarrying > 0:
            carrying_food += opponent.numCarrying

      for opponent in opponents:
        if(opponent.getPosition()==None):
          continue
        if opponent.isPacman:
            if(pacman_distance==0):
              pacman_distance=np.inf
            pacman_distance = min(pacman_distance, (abs(myPos[0] - opponent.getPosition()[0]) + abs(myPos[1] - opponent.getPosition()[1])))
        if opponent.scaredTimer > 0 and not(opponent.isPacman):
          if(scared_ghost_distance == 0):
            scared_ghost_distance = np.inf
          scared_ghost_distance = min(scared_ghost_distance, (abs(myPos[0]-opponent.getPosition()[0])+abs(myPos[1]-opponent.getPosition()[1])))
        if opponent.scaredTimer == 0 and not(opponent.isPacman):
          if(non_scared_ghost_distance == 0):
            non_scared_ghost_distance = np.inf
          non_scared_ghost_distance = min(non_scared_ghost_distance, (abs(myPos[0]-opponent.getPosition()[0])+abs(myPos[1]-opponent.getPosition()[1])))

      heuristic_value =(0.5+
        c * self.w
        - alpha * food_distance/max_distance
        - beta * horizontal_dist/food_matrix.width 
        - gamma * capsule_distance/max_distance
        - delta * carrying_food
        - epsilon * scared_ghost_distance/max_distance
        + zeta * non_scared_ghost_distance/max_distance
        - tau * pacman_distance)
      return  heuristic_value
      """
      print("W",c*self.w, "innit_w", - alpha * food_distance/max_distance
        - beta * horizontal_dist/food_matrix.width 
        - gamma * capsule_distance/max_distance
        - delta * carrying_food
        - epsilon * scared_ghost_distance/max_distance
        + zeta * non_scared_ghost_distance/max_distance
        - tau * pacman_distance)
      """

                             
    def select(self, depth):
        if depth == 0 or self.state.isOver():
            return self, depth
        if len(self.children) == 0:
            return self, depth
        #or based on ucb()
        if self.children:
          node=max(self.children, key=lambda child: child.ucb())
          #next_state = gameState.generateSuccessor(self.agent_index, next_node.action)
          return node.select(depth-1)

    def expand(self, gameState, actions, num_agents):
        if len(self.children) > 0:
            return

        next_agent = (self.agent_index) % num_agents
        
        for action in actions:
          next_state = gameState.generateSuccessor(self.agent_index, action)
          child = Node(next_state, next_agent, self, action)
          self.children.append(child)
        return random.choice(self.children)

    def backpropagate(self, reward,inter_reward):
        self.n += 1
        self.w += reward
        self.inter_w = max(inter_reward,self.inter_w)
        if(not(self.parent==None)):
          self.parent.backpropagate(reward,inter_reward);
        
    def prune(self, threshold):
        self.children = [child for child in self.children if child.n > threshold]
        for child in self.children:
            child.prune(threshold)        

