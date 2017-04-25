# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, math
import numpy as np

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"
    #print("Action: {}".format(legalMoves[chosenIndex]))
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currGameState, pacManAction):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    nextGameState = currGameState.generatePacmanSuccessor(pacManAction) #"tabuleiro"
    #print(nextGameState)
    newPos = nextGameState.getPacmanPosition() #coordenadas
    #print("newPos",newPos)
    oldFood = currGameState.getFood()
    #print(oldFood)
    nearestFoodDistance = min([self.distance(newPos,(x,y)) for x,_ in enumerate(oldFood) for y,_ in enumerate(oldFood[x]) if oldFood[x][y]])
    #print(nearestFoodDistance)
    newGhostStates = nextGameState.getGhostStates()
    nearestGhostDistance = min([self.distance(newPos,g.getPosition()) for g in newGhostStates])
    #print(nearestGhostDistance) #lsita de objetos do fantasma. O .getPosition() retorna as coordenadas
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #timer do medo do fantasma
    #print("newScaredTimes",newScaredTimes)
    #print("------------------------------------------------")

    "*** YOUR CODE HERE ***"
    if pacManAction=="Stop":
      return -1

    w1 = 10.0
    if(nearestFoodDistance>0):
      #print("distancia comida: {}".format(nearestFoodDistance))
      w1 /= nearestFoodDistance

    w2 = 10.0
    if(nearestGhostDistance>0):
      #print("distancia fantasma: {}".format(nearestGhostDistance))
      w2 /= nearestGhostDistance
    #print(w1,w2,w1-w2)
    return w1-w2
    #return nextGameState.getScore()

  def distance(self,pos1,pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1]) #manhatam distance

class QAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.alpha=0.0004
    try:
      self.w = np.loadtxt('q-wheigts')
      print(self.w)
    except Exception:
      self.w = np.random.rand(3)
      self.agent = ReflexAgent()

  def saveWheigts(self):
    #print(self.w)
    np.savetxt('q-wheigts',self.w)

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    #chosenIndex = random.choice(range(len(scores)))# Pick randomly
    "Add more of your code here if you want to"
    #print("Action: {}".format(legalMoves[chosenIndex]))
    return legalMoves[chosenIndex]
    #return self.agent.getAction(gameState)

  def evaluationFunction(self, currGameState, pacManAction):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    nextGameState = currGameState.generatePacmanSuccessor(pacManAction)
    val,features = self.computeQ(currGameState,pacManAction)
    #print(currGameState.getPacmanPosition(),nextGameState.getPacmanPosition())
    self.w = [self.w[i]+self.alpha*self.difference(nextGameState,val)*features[i] for i,_ in enumerate(self.w)]
    #print(self.w)
    return val
    #return nextGameState.getScore()

  def difference(self,gameState,val):
    legalMoves = gameState.getLegalActions()
    scores = [self.computeQ(gameState, action)[0] for action in legalMoves]

    r = -1
    if self.countFood(gameState)==0:
      r = 500

    else:
      pos = gameState.getPacmanPosition()
      ghostStates = gameState.getGhostStates()
      nearestGhostDistance = min([self.distance(pos,g.getPosition()) for g in ghostStates])
      if(nearestGhostDistance==0):
        r = -500

    if len(scores)>0:
      return r - max(scores) - val
    else:
      return r - val

  def countFood(self,currGameState):
    pos = currGameState.getPacmanPosition()
    food = currGameState.getFood()
    return len([self.distance(pos,(x,y)) for x,_ in enumerate(food) for y,_ in enumerate(food[x]) if food[x][y]])

  def computeQ(self,currGameState,action):
    features = []
    # Useful information you can extract from a GameState (pacman.py)
    nextGameState = currGameState.generatePacmanSuccessor(action) #"tabuleiro"
    
    newPos = nextGameState.getPacmanPosition() #coordenadas
    
    food = currGameState.getFood()
    
    nearestFoodDistanceNew = min([self.distance(newPos,(x,y)) for x,_ in enumerate(food) for y,_ in enumerate(food[x]) if food[x][y]])
    
    newGhostStates = nextGameState.getGhostStates()
    nearestGhostDistanceNew = min([self.distance(newPos,g.getPosition()) for g in newGhostStates])
    
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #timer do medo do fantasma

    remainFood = self.countFood(currGameState)
    if nearestFoodDistanceNew>0:
      features.append(float(1/nearestFoodDistanceNew))
    else:
      features.append(1)
    if nearestGhostDistanceNew>0:
      features.append(float(1/nearestGhostDistanceNew))
    else:
      features.append(1)

    features.append(float(1/remainFood))
    return (np.dot(features,self.w),features)

  def distance(self,pos1,pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1]) #manhatam distance


def scoreEvaluationFunction(currentGameState):
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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, currGameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      currGameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      currGameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      currGameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, currGameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, currGameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def betterEvaluationFunction(currGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

