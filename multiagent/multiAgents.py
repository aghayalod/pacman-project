# multiAgents.py
# --------------
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


from os import terminal_size
from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # find min dist to food
        food_list = successorGameState.getFood().asList()
        min_dist_food = float("inf")
        for food in food_list:
            min_dist_food = min(min_dist_food, manhattanDistance(newPos, food))

        ghosts = successorGameState.getGhostPositions()
        for ghost in ghosts:
            if (manhattanDistance(newPos, ghost) <= 1): return float('-inf')

        # less minimum distance = greater reciprocal
        # biases successors that are closest to a food
        return successorGameState.getScore() + 1.0/min_dist_food


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def max_node(self, gameState: GameState, curr_depth):
        actions = gameState.getLegalActions(0)
        if (gameState.isWin() or gameState.isLose() or len(actions) == 0 or curr_depth == self.depth):
            return self.evaluationFunction(gameState)
        max_val = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            max_val = max(max_val, self.min_node(successor, curr_depth, 1))
        return max_val

    def min_node(self, gameState: GameState, curr_depth, agent_idx):

        if (gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent_idx)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        min_val = float('inf')

        for action in actions:
            successor = gameState.generateSuccessor(agent_idx, action)
            # if current agent is the last ghost, call max_node
            if (agent_idx == gameState.getNumAgents()-1):
                min_val = min(min_val, self.max_node(successor, curr_depth+1))
            else:
                min_val = min(min_val, self.min_node(
                    successor, curr_depth, agent_idx+1))
        return min_val

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        best_action = Directions.STOP
        if gameState.isWin() or gameState.isLose():
            return best_action
        actions = gameState.getLegalActions(0)
        max_val = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            val = self.min_node(successor, 0, 1)
            if (val > max_val):
                max_val = val
                best_action = action
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_node(self, gameState: GameState, curr_depth, alpha, beta):
        actions = gameState.getLegalActions(0)
        if (curr_depth == self.depth or len(actions) == 0):
            return self.evaluationFunction(gameState)
        max_val = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            max_val = max(max_val, self.min_node(successor, curr_depth, 1, alpha, beta))
            if (max_val > beta): return max_val
            alpha = max(alpha, max_val)
        return max_val

    def min_node(self, gameState: GameState, curr_depth, agent_idx, alpha, beta):
        actions = gameState.getLegalActions(agent_idx)
        if(len(actions) == 0):
            return self.evaluationFunction(gameState)
        min_val = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(agent_idx, action)
            #if current agent is the last ghost, call max_node
            if(agent_idx == gameState.getNumAgents()-1):
                min_val = min(min_val, self.max_node(successor, curr_depth+1, alpha, beta))
            else:
                min_val = min(min_val, self.min_node(successor, curr_depth, agent_idx+1, alpha, beta))  
            if(min_val < alpha): return min_val
            beta = min(beta,min_val)
        return min_val

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        actions = gameState.getLegalActions(0)
        alpha = float('-inf')
        beta = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            val = self.min_node(successor, 0, 1, alpha , beta)
            if(val > alpha):
                alpha = val
                best_action = action
        return best_action



        

 

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def max_node(self, gameState: GameState, curr_depth):
        actions = gameState.getLegalActions(0)
        if (gameState.isWin() or gameState.isLose() or len(actions) == 0 or curr_depth == self.depth):
            return self.evaluationFunction(gameState)
        max_val = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            max_val = max(max_val, self.min_node(successor, curr_depth, 1))
        return max_val

    def min_node(self, gameState: GameState, curr_depth, agent_idx):

        if (gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent_idx)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        values = 0

        for action in actions:
            successor = gameState.generateSuccessor(agent_idx, action)
            # if current agent is the last ghost, call max_node
            if (agent_idx == gameState.getNumAgents()-1):
                values += self.max_node(successor, curr_depth+1)
            else:
                values += self.min_node(successor, curr_depth, agent_idx+1)
        return values/ len(actions)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        best_action = Directions.STOP
        if gameState.isWin() or gameState.isLose():
            return best_action
        actions = gameState.getLegalActions(0)
        max_val = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            val = self.min_node(successor, 0, 1)
            if (val > max_val):
                max_val = val
                best_action = action
        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <bias closest food distance, closest ghost distance and #powerup pellets left. 
    each of the aforementioned values are multiplied by some constant to give each their own weighting so... 
    we care a lot about the closest food then the number of powerupss left then food left ... but we really 
    care about a terminal state and the closet ghost as those can cause us to lose the game >
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()


    min_food_dist = float('inf')
    for food in food_list:
        min_food_dist = min(min_food_dist, manhattanDistance(pos, food))

    ghosts = currentGameState.getGhostPositions()
    min_ghost_dist = float('-inf')
    for ghost in ghosts:
        ghost_dist = manhattanDistance(pos, ghost)
        if (ghost_dist < 2):
            return -float('inf')
        min_ghost_dist = min(min_ghost_dist, ghost_dist)

    food_left = currentGameState.getNumFood()
    powerups_left = len(currentGameState.getCapsules())

    food_left_mult = 20000
    powerups_left_mult = 5000
    food_dist_mult = 500

    terminal_state = 0
    if currentGameState.isLose():
        terminal_state -= 50000
    elif currentGameState.isWin():
        terminal_state += 50000

    return 1.0/(food_left + 1) * food_left_mult + 1.0/(min_food_dist + 1) * \
    food_dist_mult + 1.0/(powerups_left + 1) * powerups_left_mult + terminal_state + ghost_dist
# Abbreviation
better = betterEvaluationFunction
