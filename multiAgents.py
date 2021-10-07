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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

from game import Actions

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
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newFoodList = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newCapsules = successorGameState.getCapsules()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        currPos = currentGameState.getPacmanPosition()
        currFood = currentGameState.getFood()
        currFoodList = currentGameState.getFood().asList()
        currGhostStates = currentGameState.getGhostStates()
        currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

        for ghostState in newGhostStates:
            if ghostState.getPosition() == newPos:
                return -1
        if action == 'Stop':
            return -1

        toReturn = 0
        if len(newFoodList) < len(currFoodList) or newPos in newCapsules:
            toReturn = 1

        for item in newScaredTimes:
            toReturn+=item

        foodDistance = 0
        shortestFood = 0
        shortestCap = 0

        if len(newFoodList) > 0:
            shortestFood = abs(newPos[0] - newFoodList[0][0]) + abs(newPos[1] - newFoodList[0][1])
            for item in newFoodList:
                foodDistance = abs(newPos[0] - item[0]) + abs(newPos[1] - item[1])
                if foodDistance < shortestFood:
                    shortestFood = foodDistance

        else:
            if (len(newCapsules) > 0):
                shortestCap = abs(newPos[0] - newCapsules[0][0]) + abs(newPos[1] - newCapsules[0][1])
                for item in newCapsules:
                    foodDistance = abs(newPos[0] - item[0]) + abs(newPos[1] - item[1])
                    if foodDistance < shortestCap:
                        shortestCap = foodDistance

        if shortestFood > 0:
            return 1.0/float(shortestFood) + toReturn
        else:
            return toReturn

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
        self.stateChoices = {}

    def updateDict(self, gameState, action):
        self.stateChoices[gameState] = action

    def getOptimalAction(self, gameState):
        return self.stateChoices[gameState]

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
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
        """
        legalActions = gameState.getLegalActions(self.index)
        successors = []
        successorsDict = {}

        for action in legalActions:
            added = gameState.generateSuccessor(self.index, action)
            successors.append(added)
            successorsDict[added] = action

        maxVal = self.minValue(successors[0], 1, gameState.getNumAgents(), self.depth)
        maxAction = successorsDict[successors[0]]

        for state in successors:
            if self.minValue(state, 1, gameState.getNumAgents(), self.depth) > maxVal:
                maxVal = self.minValue(state, 1, gameState.getNumAgents(), self.depth)
                maxAction = successorsDict[state]

        return maxAction


    def maxValue(self, gameState, index, numAgents, depth):
        if gameState.isWin() or gameState.isLose() or index==numAgents*depth:
            return self.evaluationFunction(gameState)

        div = (index + 1) / numAgents
        mod = index + 1 - div * numAgents

        legalActions = gameState.getLegalActions(0)
        successors = []

        for action in legalActions:
            successors.append(gameState.generateSuccessor(0, action))

        v = float('-inf')
        for successor in successors:
            v = max(v, self.minValue(successor, index+1, numAgents, depth))
        return v
        util.raiseNotDefined()

    def minValue(self, gameState, index, numAgents, depth):
        if gameState.isWin() or gameState.isLose() or index == numAgents*depth:
            return self.evaluationFunction(gameState)

        div2 = (index) / numAgents
        mod2 = index - div2 * numAgents

        legalActions = gameState.getLegalActions(mod2)
        successors = []

        for action in legalActions:
            successors.append(gameState.generateSuccessor(mod2, action))

        v = float('inf')
        for successor in successors:
            div = (index + 1) / numAgents
            mod = index + 1 - div * numAgents

            if (mod == 0):
                v = min(v, self.maxValue(successor, index+1, numAgents, depth))
            else:
                v = min(v, self.minValue(successor, index+1, numAgents, depth))
        return v
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        legalActions = gameState.getLegalActions(self.index)

        maxVal = self.maxValue(gameState, 0, gameState.getNumAgents(), self.depth, float('-inf'), float('inf'))
        return self.getOptimalAction(gameState)

        util.raiseNotDefined()

    def maxValue(self, gameState, index, numAgents, depth, alpha, beta):
        # terminal test
        if gameState.isWin() or gameState.isLose() or not gameState.getLegalActions() or index==depth*numAgents:
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(0)
        successors = []

        v = float('-inf')
        bestAction = legalActions[0]

        for action in legalActions:
            prev_v = v
            v = max(v, self.minValue(gameState.generateSuccessor(0, action), index+1, numAgents, depth, alpha, beta))
            if prev_v != v:
                bestAction = action
            if v > beta:
                #print "returning ", v, "because,", v, "is greater than", beta
                self.updateDict(gameState, bestAction)
                return v
            alpha = max(alpha, v)
        #print "returning ", v, "normally!"
        self.updateDict(gameState, bestAction)
        return v
        util.raiseNotDefined()

    def minValue(self, gameState, index, numAgents, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or not gameState.getLegalActions() or index == depth*numAgents:
            return self.evaluationFunction(gameState)

        div2 = (index) / numAgents
        mod2 = index - div2 * numAgents

        legalActions = gameState.getLegalActions(mod2)
        successors = []

        v = float('inf')
        bestAction = legalActions[0]

        for action in legalActions:

            div = (index + 1) / numAgents
            mod = (index + 1) - div * numAgents

            if (mod == 0):
                prev_v = v
                v = min(v, self.maxValue(gameState.generateSuccessor(mod2, action), index+1, numAgents, depth, alpha, beta))
                if prev_v != v:
                    bestAction = action
                if v < alpha:
                    self.updateDict(gameState, bestAction)
                    return v
                beta = min(beta, v)
            else:
                prev_v = v
                v = min(v, self.minValue(gameState.generateSuccessor(mod2, action), index+1, numAgents, depth, alpha, beta))
                if prev_v != v:
                    bestAction = action
                if v < alpha:
                    self.updateDict(gameState, bestAction)
                    return v
                beta = min(beta, v)
        self.updateDict(gameState, bestAction)
        return v
        util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        legalActions = gameState.getLegalActions(self.index)
        successors = []
        successorsDict = {}

        for action in legalActions:
            added = gameState.generateSuccessor(self.index, action)
            successors.append(added)
            successorsDict[added] = action

        maxVal = self.expectiMaxValue(successors[0], 1, gameState.getNumAgents(), self.depth)
        maxAction = successorsDict[successors[0]]

        for state in successors:
            if self.expectiMaxValue(state, 1, gameState.getNumAgents(), self.depth) > maxVal:
                maxVal = self.expectiMaxValue(state, 1, gameState.getNumAgents(), self.depth)
                maxAction = successorsDict[state]

        return maxAction
        util.raiseNotDefined()

    def maxValue(self, gameState, index, numAgents, depth):
        if gameState.isWin() or gameState.isLose() or index==numAgents*depth:
            return self.evaluationFunction(gameState)

        div = (index + 1) / numAgents
        mod = index + 1 - div * numAgents

        legalActions = gameState.getLegalActions(0)
        successors = []

        for action in legalActions:
            successors.append(gameState.generateSuccessor(0, action))

        v = float('-inf')
        for successor in successors:
            v = max(v, self.expectiMaxValue(successor, index+1, numAgents, depth))
        return v
        util.raiseNotDefined()

    def expectiMaxValue(self, gameState, index, numAgents, depth):
        if gameState.isWin() or gameState.isLose() or index == numAgents*depth:
            return self.evaluationFunction(gameState)

        div2 = (index) / numAgents
        mod2 = index - div2 * numAgents

        legalActions = gameState.getLegalActions(mod2)
        successors = []

        for action in legalActions:
            successors.append(gameState.generateSuccessor(mod2, action))

        v = 0
        divisor = len(successors)
        for successor in successors:
            div = (index + 1) / numAgents
            mod = index + 1 - div * numAgents

            # if pacman
            if mod == 0:
                tempv = self.maxValue(successor, index+1, numAgents, depth)
                tempProb = float(float(tempv)/float(divisor))
                v += tempProb
            # if not pacman
            else:
                tempv = self.expectiMaxValue(successor, index+1, numAgents, depth)
                tempProb = float(float(tempv)/float(divisor))
                v += tempProb
        return v

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Our linear combination included the reciprocal of the amount of food remaining, the reciprocal of
      the shortest maze distance to a food pellet, and the total scared times of the ghosts. We first calculated the shortest
      distance using Manhattan distance, stored the food pellet associated with that distance, and found the maze distance to that point
      and called it 'actualShortest.' We also calculated the total scared times by adding up the scared times for each ghost and called it 'totalScared.'
      If the current game state is the same as a ghost state, we return -100,000 and if all of the food has been found we return 100,000.
      If it is not a terminal state, we returned a linear combination. In our linear combination,
      we gave 1/(food remaining) a weight of 10,000, 1/(actualShortest) a weight of 10, and totalScared a weight of 1.
      Our evaluation function then returned the sum of 10,000/(food remaining), 10/actualShortest, and totalScared.
    """
    # Useful information you can extract from a GameState (pacman.py)
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currFoodList = currentGameState.getFood().asList()
    currGhostStates = currentGameState.getGhostStates()
    currCapsules = currentGameState.getCapsules()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

    # creating a linear combination of features
    foodDistance = 0
    shortestFood = 0
    actualShortest = 0
    shortestCap = 0

    if len(currFoodList) > 0:
        shortestList = []
        shortestFood = abs(currPos[0] - currGhostStates[0].getPosition()[0]) + abs(
            currPos[1] - currGhostStates[0].getPosition()[1])
        shortestFoodPos = (currGhostStates[0].getPosition()[0], currGhostStates[0].getPosition()[1])
        for item in currFoodList:
            foodDistance = abs(currPos[0] - item[0]) + abs(currPos[1] - item[1])
            if foodDistance < shortestFood:
                shortestFood = foodDistance
                shortestFoodPos = item
        actualShortest = mazeDistance((int(currPos[0]), int(currPos[1])), (int(shortestFoodPos[0]), int(shortestFoodPos[1])), currentGameState)


    if len(currGhostStates) > 0:
        shortestGhost = abs(currPos[0] - currGhostStates[0].getPosition()[0]) + abs(currPos[1] - currGhostStates[0].getPosition()[1])
        for item in currGhostStates:
            gDistance = abs(currPos[0] - item.getPosition()[0]) + abs(currPos[1] - item.getPosition()[1])
            if gDistance < shortestGhost:
                shortestGhost = gDistance

    if (len(currCapsules) > 0):
        shortestCap = abs(currPos[0] - currCapsules[0][0]) + abs(currPos[1] - currCapsules[0][1])
        for item in currCapsules:
            foodDistance = abs(currPos[0] - item[0]) + abs(currPos[1] - item[1])
            if foodDistance < shortestCap:
                shortestCap = foodDistance

    totalScared = 0
    if len(currScaredTimes) > 0:
        for item in currScaredTimes:
            totalScared+=item


    toReturn = float(0)
    if (shortestGhost == 0):
        return -100000
    if (shortestFood == 0):
        return 100000
    if (shortestFood > 0):
        return 10000*float(float(1.0)/float(len(currFoodList))) + 10*float(float(1.0)/float(actualShortest)) + totalScared
    util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # create the start node
    startNode = node(problem.getStartState(), None, 0, None, 0)
    # if the start node is the goal state, we are done (return startNode)
    if problem.isGoalState(problem.getStartState()):
        return []
    # create a queue d/s to hold nodes
    bfsQueue = util.Queue()
    # creates a list of states already in the queue
    inQueue = []
    # push start node onto queue
    bfsQueue.push(startNode)
    # add start node onto list of states already in the queue
    inQueue.append(startNode.getState())
    # create a list to hold the states of explored nodes
    exploredSet = []
    a = True
    # big while loop that runs until either pacman fails or a solution is found
    while(a):
        # if the stack is empty, then we have examined all nodes and not found a solution, so return failure
        if bfsQueue.isEmpty():
            a = False
        # pop the most recently pushed item from the stack, and call it the current node
        currentNode = bfsQueue.pop()
        # if the child node has a state that is the goal state
        if problem.isGoalState(currentNode.getState()):
            # create empty list of actions to return, and append direction from goal state node
            returnList = []
            returnList.append(currentNode.getDirection())
            # while each node still has a parent node
            while currentNode.getParent():
                # get the current node's parent
                tempNode = currentNode.getParent()
                # add the direction of the parent node to the return list of actions
                returnList.append(tempNode.getDirection())
                # change current node reference to parent node
                currentNode = currentNode.getParent()
            # remove direction from initial start node
            returnList.remove(None)
            # reverse directions in return list of actions
            returnList.reverse()
            # return list of actions
            return returnList
        # add the state of the current node into the explored set
        exploredSet.append(currentNode.getState())
        # find all the successors (child nodes) of the current node
        for i in problem.getSuccessors(currentNode.getState()):
            # construct an instance of each child node
            childNode = node(i[0],i[1],i[2],currentNode,0)
            # if the child node has a state that has not been explored yet
            if childNode.getState() not in exploredSet and childNode.getState()  not in inQueue:
                # if the child node has a state that is not the goal state, push onto the stack
                bfsQueue.push(childNode)
                inQueue.append(childNode.getState())
    util.raiseNotDefined()

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost



# node class
class node:
    # initialize node with state, direction, path cost, parent node pointer, heuristic function
    def __init__(self, state, direction, pathcost, parentnode, heuristic):
        self.state = state
        self.direction = direction
        self.pathcost = pathcost
        self.parentnode = parentnode
        self.heuristic = heuristic

    # print node
    def printNode(self):
        print self.state, self.direction, self.pathcost, self.parentnode, self.heuristic

    # get state
    def getState(self):
        return self.state

    # print state
    def printState(self):
        print self.state

    # get parent
    def getParent(self):
        return self.parentnode

    # get direction
    def getDirection(self):
        return self.direction

    # print direction
    def printDirection(self):
        print self.direction

    # set path cost
    def setPathCost(self, pathcost):
        self.pathcost = pathcost

    # get path cost
    def getPathCost(self):
        return self.pathcost

    # get heuristic
    def getHeuristic(self):
        return self.heuristic

    # get state as string
    def stateToString(self):
        return str(self.state)
# Abbreviation
better = betterEvaluationFunction

