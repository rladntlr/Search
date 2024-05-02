# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    #DFS를 사용하기 위한 스택
    DFS_Stack = util.Stack()
    #방문한 노드 넣기, set을 이용하여 연결리스트보다 시간절약
    visited_node = set()
    #경로는 순서가 중요하므로 리스트로
    to_direc_action = []
    #dfs스택에 시작점과 경로를 튜플 형태로 넣는다
    DFS_Stack.push((problem.getStartState(),to_direc_action))
    #스택에 아이템이 있을 동안 반복
    while DFS_Stack:
        #스택에서 마지막에 추가된 노드와 경로를 추출
        currentNode, currentActions = DFS_Stack.pop()
        #현재 노드가 방문한 적이 없으면
        if currentNode not in visited_node:
            #방문한 노드 집합에 현재 노드 추가
            visited_node.add(currentNode)
            #현재 노드가 goalstate인지 확인
            if problem.isGoalState(currentNode):
                #도달했다면 현재까지의 경로를 리턴
                return currentActions
            #현재 노드 다음 상황에 대해서 반복 successor : 다음상황, action : 다음 행동 stepCost : 행동에 필요한 비용=>dfs에서는 사용하지 않음
            for successor,action, stepCost in problem.getSuccessors(currentNode):
                #후속 상태와 새로운 경로를 스택에 추가, 새로운 경로는 현재 경로 + 다음 방향
                DFS_Stack.push((successor, currentActions + [action]))
    return 0

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #BFS를 사용하기 위한 큐
    BFS_QUEUE = util.Queue()
    visited_node = set()
    to_direc_action = []
    BFS_QUEUE.push((problem.getStartState(),to_direc_action))
    while BFS_QUEUE:
        currentNode, currentActions = BFS_QUEUE.pop()
        if currentNode not in visited_node:
            visited_node.add(currentNode)
            if problem.isGoalState(currentNode):
                return currentActions
            for successor,action, stepCost in problem.getSuccessors(currentNode):
                BFS_QUEUE.push((successor, currentActions + [action]))
    return 0
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    UCS_PQ = util.PriorityQueue()
    visited_node = set()
    to_direc_action = []
    #pq에서는 비용을 보고 탐색순서를 정하기 때문에 0으로 둔다
    UCS_PQ.push((problem.getStartState(),to_direc_action),0)
    while UCS_PQ:
        #pop 함수를 통해 비용이 적은 것을 뽑아낸다
        currentNode, currentActions = UCS_PQ.pop()
        if currentNode not in visited_node:
            visited_node.add(currentNode)
            if problem.isGoalState(currentNode):
                return currentActions
            for successor,action, stepCost in problem.getSuccessors(currentNode):
                #problem.getCostOfActions(currentActions + [action])부분에서 총 비용을 계산
                UCS_PQ.push((successor, currentActions + [action]), problem.getCostOfActions(currentActions + [action]))
    return 0
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    Astar_PQ = util.PriorityQueue()
    visited_node = set()
    to_direc_action = []
    #휴리스틱은 가장 가까운 목적지 비용 추정, 휴리스틱이 problem 객체에 접근 가능
    Astar_PQ.push((problem.getStartState(), to_direc_action), heuristic(problem.getStartState(), problem))
    while Astar_PQ:
        currentNode, currentActions = Astar_PQ.pop()
        if currentNode not in visited_node:
            visited_node.add(currentNode)
            if problem.isGoalState(currentNode):
                return currentActions
            for successor, action, stepCost in problem.getSuccessors(currentNode):
                #problem.getCostOfActions(currentActions + [action]) + heuristic(successor, problem) 실제 경로 비용 + 휴리스틱 비용
                Astar_PQ.push((successor, currentActions + [action]), problem.getCostOfActions(currentActions + [action]) + heuristic(successor, problem))
    return 0
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
