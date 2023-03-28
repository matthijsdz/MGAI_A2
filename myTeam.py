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
    self.depth = 10
    self.M = 20
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
    epsilon=1.0
    inter_w=-np.inf
    next_agent = (node.agent_index + 2) % self.num_agents
    if(random.uniform(0,1)<epsilon):
      action = random.choice(actions)
      next_state =node.state.generateSuccessor(node.agent_index, action)
      next_node = Node(next_state, next_agent, node, action)
    else:
      for action in actions:
        temp_state = node.state.generateSuccessor(node.agent_index, action)
        temp_node = Node(temp_state, next_agent, node, action)
        next_value = temp_node.inter_w
        if(next_value>inter_W):
          next_node=temp_node
          inter_w=next_value
    return next_node
  
  #Multiprocessing
  def treeExpansion(self,root, gameState):
        node, depth= root.select(gameState,self.depth)
        node.expand(node.state, node.state.getLegalActions(node.agent_index), self.num_agents)
        for child in node.children:
          full_result,inter_result = self.simulate(child, depth)
          node.backpropagate(full_result,inter_result)
    
  def chooseAction(self, gameState):
    start = time.time()

    root = Node(gameState, self.index)
    
    #with mp.Pool(processes=mp.cpu_count()) as pool:
    while time.time() - start < 1:
        self.treeExpansion(root,gameState)
        #args=[(root,gameState)] * mp.cpu_count()
        #pool.map(self.treeExpansion, args)
    #threshold = 2
    #root.prune(threshold)
    
    #pool.apply_async(self.treeExpansion, args=(root, gameState), callback=lambda node: root.children.append(node))

    #pool.close()
    #pool.join()
    actions = gameState.getLegalActions(self.index)

    print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    print(max(root.children, key=lambda child: child.inter_w).inter_w)
    [print(child.inter_w) for child in root.children]
    [print(child.pos) for child in root.children]
    print("pap", root.pos)
    print(actions)
    print(max(root.children, key=lambda child: child.n).n, min(root.children, key=lambda child: child.n).n)
    return max(root.children, key=lambda child: child.inter_w).action
      
class Node:
    def __init__(self, state, agent_index, parent=None, action=None):
        self.state = state
        self.agent_index = agent_index
        self.parent = parent
        self.children = []
        self.w = 0
        self.n = 1
        self.action = action
        self.inter_w=0
        self.pos=state.getAgentPosition(agent_index);

    def ucb(self):
        c = np.sqrt(2)
        return self.w / self.n + c * np.sqrt(np.log(self.parent.n) / self.n) 

    def getHeuristic(self):
      alpha= 0.01
      food_distance=0
      
      beta=0.1
      horizontal_dist=0
      dist_caps=0
      
      #gamma=
      #capsule_distance=
      #delta=
      #carrying_food=
      #epsilon=
      #scared_ghost_distance=
      #myState = self.state.getAgentState(self.agent_index)
      #print(type(myState))
      #print(type(self.state))
      myPos = self.state.getAgentPosition(self.agent_index)
      #Team Red
      if(self.agent_index%2==0):
          food_matrix=self.state.getBlueFood()
          caps_matrix=self.state.getBlueCapsules()
          opp_ind=self.state.getRedTeamIndices()
      #Team Blue
      else:
          food_matrix =self.state.getRedFood()
          caps_matrix=self.state.getRedCapsules()
          opp_ind=self.state.getBlueTeamIndices()
          
      for x in range(food_matrix.width):
        for y in range(food_matrix.height):
          if(food_matrix[x][y]==True):
              food_distance+=abs(myPos[0]-x)+abs(myPos[1]-y)
              horizontal_dist+=abs(myPos[0]-x)
      return - alpha * food_distance - beta * horizontal_dist#ghost_distance - gamma * capsule_distance + delta * carrying_food - epsilon * scared_ghost_distance
                             
    def select(self,gameState, depth):
        if depth == 0 or self.state.isOver():
            return self, depth
        if len(self.children) == 0:
            return self, depth
        next_node = max(self.children, key=lambda child: child.ucb())
        next_state = gameState.generateSuccessor(self.agent_index, next_node.action)
        return next_node.select(next_state,depth - 1)

    def expand(self, gameState, actions, num_agents):
        if len(self.children) > 0:
            return

        next_agent = (self.agent_index) % num_agents
        for action in actions:

            next_state = gameState.generateSuccessor(self.agent_index, action)
            child=Node(next_state, next_agent, self, action)
            self.children.append(child)

    def backpropagate(self, reward,inter_reward):
        if(not(self.parent==None)):
          self.n += 1
          self.w += reward
          self.inter_w += inter_reward
          self.parent.backpropagate(reward,inter_reward);
        
    def prune(self, threshold):
        self.children = [child for child in self.children if child.n > threshold]
        for child in self.children:
            child.prune(threshold)        

