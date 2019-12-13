# qlearningAgents.py
# ------------------
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


# Modified by Quan Chen


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):

    def __init__(self, **args):

        ReinforcementAgent.__init__(self, **args)

        self.values = util.Counter()

    def getQValue(self, state, action):

        return self.values[(state, action)]


    def computeValueFromQValues(self, state):

        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        maxValue = float("-inf")
        for action in actions:
            if maxValue <= self.getQValue(state, action) or maxValue == float("-inf"):
                maxValue = self.getQValue(state, action)

        return maxValue

    def computeActionFromQValues(self, state):

        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        maxValue = float("-inf")
        stateAction = ""
        for action in actions:
            if maxValue <= self.getQValue(state, action) or maxValue == float("-inf"):
                maxValue = self.getQValue(state, action)
                stateAction = action

        return stateAction

    def getAction(self, state):

        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return None

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):

        self.values[(state, action)] = ((1 - self.alpha) * self.getQValue(state, action)) + (self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState)))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):

        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):

        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):

        features = self.featExtractor.getFeatures(state, action)
        qvalue = 0
        for feature in features:
            qvalue += features[feature] * self.weights[feature]

        return qvalue

    def update(self, state, action, nextState, reward):

        difference = (reward + (self.discount * self.getValue(nextState))) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.weights[feature] = self.weights[feature] + (self.alpha * features[feature] * difference)

    def final(self, state):
        PacmanQAgent.final(self, state)

        if self.episodesSoFar == self.numTraining:

            pass
