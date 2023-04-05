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
from game import Directions
import random
import random, time, util, sys
import tensorflow as tf
import numpy as np
from util import nearestPoint
import os
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

# Constants and hyperparameters
#STATE_SIZE = # Define the state size for your problem


class DummyAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState);
        self.actions= ["North","East", "South", "West"]
        self.state_size=10
        self.action_size = 4  # Number of possible actions (including STOP)
        self.state=gameState
        self.pos=gameState.getAgentPosition(self.index)
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.replay_buffer_size = 10000
        self.target_update_freq = 100
        self.model = self.create_dqn_model()
        self.load_weights("/Users/markrademaker/Documents/GitHub/MGAI_A2/weights.txt")
        self.target_model = self.create_dqn_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.epsilon = self.epsilon_start
        self.train_step = 0
        
    def create_dqn_model(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.Huber())
        return model

    def getPrediction(self, features, legal_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)
        arr = np.array(list(features.values()))
        tensor = tf.expand_dims(tf.convert_to_tensor(arr), axis=0)
        q_values = self.model.predict(tensor)
        actions_ids = [self.actions.index(i) for i in legal_actions]
        valid_q_values = [q_values[0][i] for i in actions_ids]
        return self.actions[actions_ids[np.argmax(valid_q_values)]]

    def update(self, features, action, reward, next_state, done):
        self.replay_buffer.add(features, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return
        # Initialize state and target arrays
        experiences = self.replay_buffer.sample(self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))

        for i, (features, action, reward, next_features, done) in enumerate(experiences):
            arr = np.array(list(features.values()))
            next_arr = np.array(list(next_features.values()))
            states[i] = arr
            next_states[i] = next_arr

        # Batch predictions
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i, (features, action, reward, next_features, done) in enumerate(experiences):
            action_id = self.actions.index(action)
            target = current_q_values[i]
            if done:
                target[action_id] = reward
            else:
                target[action_id] = reward + self.gamma * np.max(next_q_values[i])

            targets[i] = target
        # Fit the entire batch
        self.model.fit(states, targets, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step += 1

        if self.train_step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        if(self.train_step%50==0):
          self.save_weights("/Users/markrademaker/Documents/GitHub/MGAI_A2/weights.txt");

          
    def getFeatures(self, state):
        features = util.Counter()
        pos = state.getAgentPosition(self.index)
        features["numCarrying"]=f_numCarrying = state.getAgentState(self.index).numCarrying
        features["food_distance"],  features["capsule_distance"], features["teammate_distance"] = self.getTeamFeatures( state,pos)
        features["pacman_distance"], features["scared_ghost_distance"], features["non_scared_ghost_distance"] = self.getOpponentFeatures(state,pos)
        features["remaining_food_count"] = self.getFood(state).count()
        features["remaining_capsules_count"] = len(self.getCapsules(state))
        features["defending_food_count"] = self.getFoodYouAreDefending(state).count()
        food_matrix=self.getFood(state)
        #Normalize
        features["numCarrying"]/=(food_matrix.width/2 + food_matrix.height)
        features["remaining_food_count"] /= (food_matrix.width/2 + food_matrix.height)
        features["food_distance"] /= (food_matrix.width + food_matrix.height)
        features["capsule_distance"] /= (food_matrix.width + food_matrix.height)
        features["pacman_distance"] /= (food_matrix.width + food_matrix.height)
        features["scared_ghost_distance"] /= (food_matrix.width + food_matrix.height)
        features["non_scared_ghost_distance"] /= (food_matrix.width + food_matrix.height)
        features["teammate_distance"] /= (food_matrix.width + food_matrix.height)
        features["defending_food_count"] /= (food_matrix.width/2 + food_matrix.height)
        if(features["non_scared_ghost_distance"]==np.inf):
          features["non_scared_ghost_distance"] = 1
        if(features["scared_ghost_distance"]==np.inf):
          features["scared_ghost_distance"] =1
        if(features["pacman_distance"]==np.inf):
          features["pacman_distance"]=1
        return features
      
    def chooseAction(self, gameState):
        index = self.index
        legal_actions=self.state.getLegalActions(index)
        legal_actions.remove("Stop")
        state = self.state
        features=self.getFeatures(state)
        action_idx = self.getPrediction(features, legal_actions)
        action = legal_actions[legal_actions.index(action_idx)]
        successor = gameState.generateSuccessor(index, action)
        done = gameState.isOver()
        next_features=self.getFeatures(successor)
        reward = self.getReward(successor)
        self.update(features, action,reward,next_features, done)
        self.state = successor
        done = successor.isOver()
        return action

    def getTeamFeatures(self,  state ,pos):
        f_food_distance=np.inf
        f_capsule_distance=np.inf
        f_teammate_distance = np.inf
        food_matrix = self.getFood(state)
        caps_matrix = self.getCapsules(state)
        teammates = [state.getAgentState(i) for i in self.getTeam(state) if i != self.index]
        for teammate in teammates:
          if(teammate.getPosition() is not None):
            f_teammate_distance = min(f_teammate_distance, (abs(pos[0] - teammate.getPosition()[0]) + abs(pos[1] - teammate.getPosition()[1])))
        for x in range(food_matrix.width):
          for y in range(food_matrix.height):
            if(food_matrix[x][y]==True):
              f_food_distance=min((abs(pos[0]-x)+abs(pos[1]-y)),f_food_distance)
        for capsule in caps_matrix:
          if(f_capsule_distance==0):
            f_capsule_distance=np.inf
          f_capsule_distance= min(f_capsule_distance, (abs(pos[0]-capsule[0])+abs(pos[1]-capsule[1])))

        return f_food_distance,f_capsule_distance, f_teammate_distance
      
    def getOpponentFeatures(self, state, pos):
      f_non_scared_ghost_distance = np.inf
      f_pacman_distance = np.inf
      f_scared_ghost_distance = np.inf
      opponents = [state.getAgentState(i) for i in self.getOpponents(state)]
      for opponent in opponents:
        if(opponent.getPosition()==None):
          continue
        if opponent.isPacman:
            if(f_pacman_distance==0):
              f_pacman_distance=np.inf
            f_pacman_distance = min(f_pacman_distance, (abs(pos[0] - opponent.getPosition()[0]) + abs(pos[1] - opponent.getPosition()[1])))
            
        if opponent.scaredTimer > 0 and not(opponent.isPacman):
          if(f_scared_ghost_distance == 0):
            f_scared_ghost_distance = np.inf
          f_scared_ghost_distance = min(f_scared_ghost_distance, (abs(pos[0]-opponent.getPosition()[0])+abs(pos[1]-opponent.getPosition()[1])))
          
        if opponent.scaredTimer == 0 and not(opponent.isPacman):
          if(f_non_scared_ghost_distance == 0):
            f_non_scared_ghost_distance = np.inf
          f_non_scared_ghost_distance = min(f_non_scared_ghost_distance, (abs(pos[0]-opponent.getPosition()[0])+abs(pos[1]-opponent.getPosition()[1])))
          
      return f_pacman_distance, f_scared_ghost_distance, f_non_scared_ghost_distance
    
    def getReward(self, state):
        if self.getPreviousObservation() is None:
            return 0

        reward = 0
        previousState = self.state



        # Find out if we got a pac dot. If we did, add 10 points.
        previousFood = self.getFood(previousState)
        previousOppFood = self.getFoodYouAreDefending(previousState)
        myPosition = state.getAgentPosition(self.index)
        currentFood = self.getFood(state).asList()
        currentOppFood =self.getFoodYouAreDefending(state)

        if myPosition in previousFood and myPosition not in currentFood:
            reward += 10
        if currentOppFood.count()>previousOppFood.count():
            reward+=10
        reward+=self.getScore(state)
        return reward         
    def load_weights(self, filename):
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist. Using default weights.")
            return
        with open(filename, 'r') as f:
            lines = f.readlines()

        layer_weights = []
        layer_biases = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "Layer weights:":
                i += 1
                weights = []
                while i < len(lines) and lines[i].strip()!= "Layer biases:":
                    weight_row = [float(w) for w in lines[i].strip().split()]
                    weights.append(weight_row)
                    i += 1
                layer_weights.append(np.array(weights))
            if lines[i].strip() == "Layer biases:":
                i += 1
                biases = [float(b) for b in lines[i].strip().split()]
                layer_biases.append(np.array(biases))
            i += 1
        for layer, weights, biases in zip(self.model.layers, layer_weights, layer_biases):
            layer.set_weights([weights, biases])

    def save_weights(self, filename):
        print("saving....")
        with open(filename, 'w') as f:
            for layer in self.model.layers:
                layer_weights = layer.get_weights()[0].tolist()  # Weights
                layer_biases = layer.get_weights()[1].tolist()  # Biases
                f.write("Layer weights:\n")
                for weights in layer_weights:
                    f.write(" ".join([str(w) for w in weights]) + "\n")
                f.write("Layer biases:\n")
                f.write(" ".join([str(b) for b in layer_biases]) + "\n")
                f.write("\n")
  
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
