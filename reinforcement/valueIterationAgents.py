# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp
import util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(0, self.iterations):
            # copy previous values
            prev = self.values.copy()
            for state in self.mdp.getStates():

                # check if terminal then get best action
                if not self.mdp.isTerminal(state):
                    action = self.getAction(state)
                    qvalue = self.getQValue(state, action)
                    prev[state] = qvalue
                else:
                    prev[state] = 0
            self.values = prev

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        transitionProb = self.mdp.getTransitionStatesAndProbs(state, action)
        Qval = 0

        for futureState, prob in transitionProb:
            Qval += prob * (self.mdp.getReward(state, action, futureState) +
                            self.discount * self.getValue(futureState))
        return Qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actionValues = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            actionValues[action] = self.computeQValueFromValues(state, action)
        return actionValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(0, self.iterations):
            stateIdx = i % len(states)
            state = states[stateIdx]
            if not self.mdp.isTerminal(state):
                prev = self.values.copy()
                action = self.getAction(state)
                qvalue = self.getQValue(state, action)
                prev[state] = qvalue
                self.values = prev


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        queue = util.PriorityQueue()
        predVals = util.Counter()
        states = self.mdp.getStates()
        for state in states:
            if not self.mdp.isTerminal(state):
                predVals[state] = set()

        for state in states:
            if not self.mdp.isTerminal(state):
                # compute pred
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitionProb = self.mdp.getTransitionStatesAndProbs(
                        state, action)
                    for futureState, prob in transitionProb:
                        if prob != 0 and not self.mdp.isTerminal(futureState):
                            predVals[futureState].add(state)

                action = self.computeActionFromValues(state)
                maxQ = self.computeQValueFromValues(state, action)
                diff = abs(self.values[state] - maxQ)
                queue.push(state, -diff)

        for i in range(0, self.iterations):
            if (queue.isEmpty()):
                return
            s = queue.pop()

            action = self.computeActionFromValues(s)
            maxQ = self.computeQValueFromValues(s, action)
            self.values[s] = maxQ
            for pred in predVals[s]:
                pAction = self.computeActionFromValues(pred)
                pmaxQ = self.computeQValueFromValues(pred, pAction)
                diff = abs(self.values[pred] - pmaxQ)
                if diff > self.theta:
                    queue.update(pred, -diff)
