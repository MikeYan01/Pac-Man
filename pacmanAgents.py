# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import *
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts·
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)


class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        
        # store all visited states
        closed = []
        closed.append(state)
        
        # initialization
        queue = []
        depth = 0
        for current_action in state.getLegalPacmanActions():
            next_state = state.generatePacmanSuccessor(current_action)
            queue.append((current_action, next_state, depth))
        
        # forward model is limited to a certain amount of calls, so it is necessary to check whether limit is reached
        run_out_of_successors = False
        
        while len(queue):
            # next state will become None, break
            if run_out_of_successors:
                break
            
            # extract the first element
            path = queue.pop(0)
            current_action = path[0]
            current_state = path[1]
            depth = path[2]

            # whether to skip
            if (current_state.isLose()) or (current_state in closed):
                continue
            elif (next_state.isWin()):
                return current_action

            # current state is visited
            closed.append(current_state)

            # for current state and each current action, generate a new successor state; when it is not none, add to the queue
            for action in current_state.getLegalPacmanActions():
                next_state = current_state.generatePacmanSuccessor(action)
                if (next_state):
                    queue.append((current_action, next_state, depth + 1))
                else:
                    run_out_of_successors = True
                    break

        # evaluate the successor states using scoreEvaluation heuristic and randomly choose a best choice
        scored = [(admissibleHeuristic(state) + depth, action) for action, state, depth in queue]
        if scored == []:
            return Directions.STOP
        # get best choice
        bestScore = min(scored, key=lambda x: x[0])[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        
        # store all visited states
        closed = []
        closed.append(state)

        # initialization
        stack = []
        depth = 0
        for current_action in state.getLegalPacmanActions():
            next_state = state.generatePacmanSuccessor(current_action)
            stack.append((current_action, next_state, depth))
        
        # forward model is limited to a certain amount of calls, so it is necessary to check whether limit is reached
        run_out_of_successors = False
        
        while len(stack):
            # next state will become None, break
            if run_out_of_successors:
                break
            
            # extract the last element
            path = stack.pop()
            current_action = path[0]
            current_state = path[1]
            depth = path[2]

            # whether to skip
            if (current_state.isLose()) or (current_state in closed):
                continue
            elif (next_state.isWin()):
                return current_action

            # current state is visited
            closed.append(current_state)

            # for current state and each action, generate a new successor state; when it is not none, add to the stack
            for action in current_state.getLegalPacmanActions():
                next_state = current_state.generatePacmanSuccessor(action)
                if (next_state):
                    stack.append((current_action, next_state, depth + 1))
                    
                else:
                    run_out_of_successors = True
                    break

        # evaluate the successor states using scoreEvaluation heuristic and randomly choose a best choice
        scored = [(admissibleHeuristic(state) + depth, action) for action, state, depth in stack]
        if scored == []:
            return Directions.STOP
        # get best choice
        bestScore = min(scored, key=lambda x: x[0])[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # implement append function for priority queue: sort by fn on append
    def pq_append(self, priorityqueue, element):
        priorityqueue.append(element)
        priorityqueue.sort(key = lambda x : x[0])
            
    # GetAction Function: Called with every frame
    def getAction(self, state):
        
        # store all visited states
        closed = []
        closed.append(state)

        # initialization
        priorityqueue = []
        depth = 0
        for current_action in state.getLegalPacmanActions():
            next_state = state.generatePacmanSuccessor(current_action)
            fn = depth + admissibleHeuristic(next_state)
            self.pq_append(priorityqueue, (fn, current_action, next_state, depth))
        
        # forward model is limited to a certain amount of calls, so it is necessary to check whether limit is reached
        run_out_of_successors = False

        while len(priorityqueue):
            # next state will become None, break
            if run_out_of_successors:
                break
            
            # extract the first element
            path = priorityqueue.pop(0)
            current_action = path[1]
            current_state = path[2]
            depth = path[3]

            # whether to skip
            if (current_state.isLose()) or (current_state in closed):
                continue
            elif (next_state.isWin()):
                return current_action

            # current state is visited
            closed.append(current_state)

            # for current state and each current action, generate a new successor state; when it is not none, add to the stack
            for action in current_state.getLegalPacmanActions():
                next_state = current_state.generatePacmanSuccessor(action)
                if (next_state):
                    # update fn
                    update_fn = (depth + 1) + admissibleHeuristic(next_state)
                    self.pq_append(priorityqueue, (update_fn, current_action, next_state, depth))
                else:
                    run_out_of_successors = True
                    break

        if priorityqueue == []:
            return Directions.STOP
        # first element will be the best solution
        best = priorityqueue.pop(0)
        return best[1]

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP

        # use getAllPossibleActions() to assign the action sequence.
        possible_actions = state.getAllPossibleActions()

        # maintain an action sequence whose length is 5
        # choose a random action from all possible actions, and append it to action sequence
        action_sequence = []
        for i in range(5):
            action_sequence.append(random.choice(possible_actions))

        # set initial best score and best action sequence
        best_score = None
        best_sequence = action_sequence[:]

        while True:
            # set current and next state
            current_state = state
            next_state = None

            # each action has 50% chance to be changed into random action
            possible_actions = current_state.getAllPossibleActions()
            for action in action_sequence:
                if (random.random() <= 0.5):
                    action_sequence[i] = random.choice(possible_actions)

            for action in action_sequence:
                # generate next state
                next_state = current_state.generatePacmanSuccessor(action)
                
                # check if reach the terminal state
                if next_state == None or next_state.isWin() or next_state.isLose():
                    break
                else:
                    current_state = next_state
            
            # end the loop
            if next_state == None:
                break
            
            # compare new evaluation result with former highest score and copy
            new_score = gameEvaluation(state, current_state)
            if (new_score > best_score):
                best_score = new_score
                best_sequence = action_sequence[:]
        
        # return the first action from the sequence with the highest gameEvaluation
        return best_sequence[0]

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP

        # select chromosomes proportionally to their ranking, generate one '1', two '2', three '3', ... , eight '8'
        def random_pick():
            chromosome_choice = []
            for i in range(1, 9):
                temp = i
                while temp > 0:
                    chromosome_choice.append(i)
                    temp -= 1
            
            return random.choice(chromosome_choice)
        
        # use getAllPossibleActions() to assign the action sequence.
        possible_actions = state.getAllPossibleActions()

        # maintain a population sequence whose length is 8 and an action sequence whose length is 5
        population = []
        action_sequence = []
        for i in range(8):
            for j in range(5):
                action_sequence.append(random.choice(possible_actions))
            
            # store [action sequence, score, rank]
            population.append([action_sequence, None, None])
        
        # reproduce 20 times
        times = 20
        while times > 0:
            times -= 1

            # calculate fitness score for population
            for i in range(8):
                current_state = state
                next_state = None

                for action in population[i][0]:
                    # generate next state
                    next_state = current_state.generatePacmanSuccessor(action)
                    
                    # check next state's status
                    if next_state == None or next_state.isWin() or next_state.isLose():
                        break
                    else:
                        current_state = next_state
                population[i][1] = gameEvaluation(state, current_state)
            
            # give each chromosome a rank, from 1(worst) to length of chromosomes(best)
            population.sort(key = lambda x : x[1])
            for i in range(8):
                population[i][2] = i + 1

            # next population
            next_population = []
            while len(next_population) < 8:
                # pair two chromosomes according to their ranking; reduce by 1 when converting rank to index
                index_X = random_pick() - 1
                index_Y = random_pick() - 1
                while index_X == index_Y:
                    index_Y = random_pick() - 1

                # 70% to generate two children by crossing over
                if random.random() <= 0.7:
                    parent_X = population[index_X]
                    parent_Y = population[index_Y]

                    # two children
                    next_sequence1 = []
                    next_sequence2 = []

                    for i in range(5):
                        # for each gene (action) do a random test, 50% percent donated by X, otherwise by Y
                        if random.random() <= 0.5:
                            next_sequence1.append(parent_X[0][i])
                            next_sequence2.append(parent_Y[0][i])
                        else:
                            next_sequence1.append(parent_Y[0][i])
                            next_sequence2.append(parent_X[0][i])
                    
                    next_population.append([next_sequence1, None, None])
                    next_population.append([next_sequence2, None, None])
                
                # otherwise, keep the pair in next generation
                else:
                    next_population.append(population[index_X])
                    next_population.append(population[index_Y])

            # 10% to mutate
            for i in range(8):
                if random.random() <= 0.1:
                    # choose a random action in the chromosome sequence and replace it by another action
                    mutate_index = random.choice([0, 1, 2, 3, 4])
                    next_population[i][0][mutate_index] = random.choice(next_population[i][0])

            population = next_population

        # return the first action from the sequence with the highest gameEvaluation
        next_population.sort(key = lambda x : x[1])
        return next_population[-1][0][0]
    
class MCTSAgent(Agent):
        # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
    
        class Node():
            def __init__(self):
                self.action = None
                self.parent = None
                self.children = []
                self.score = 0.0
                self.visit_times = 1
            
        def tree_policy(node, state):
            while True:
                # judge fully expand: num of current node's children = num of legal actions in current state

                # not fully expand
                if (len(node.children) != len(state.getLegalPacmanActions())):
                    return expand(node, state)
                else:
                    # selection, use constant 1 between exploitation and exploration
                    return max(node.children, key = 
                        lambda x: x.score * 1.0 / x.visit_times + (math.sqrt(2 * math.log(x.visit_times * 1.0) / x.visit_times)))

        def expand(node, state):
            # get all already tried actions
            already_tried = []
            for each in node.children:
                already_tried.append(each.action)

            current_state = state
            next_state = None
            for action in current_state.getLegalPacmanActions():
                next_state = current_state.generatePacmanSuccessor(action)
                if next_state == None or next_state.isWin() or next_state.isLose():
                    return None
                else:
                    current_state = next_state

                # find an untried action
                if action not in already_tried:
                    # new node should be one child of current node, and current node is parent of new node.
                    new_node = Node()
                    new_node.action = action
                    node.children.append(new_node)
                    return new_node

        def default_policy(node, state):
            # the list of all available actions in the current state
            possible_actions = state.getLegalPacmanActions()
            if len(possible_actions) == 0:
                return None

            current_state = state
            next_state = None
            
            # fix the number of rollouts to 5
            for i in range(5):
                # randomly choose a legal action
                action = random.choice(possible_actions)
                next_state = current_state.generatePacmanSuccessor(action)
                if next_state == None or next_state.isWin() or next_state.isLose():
                    return None
                else:
                    current_state = next_state

            # return reward for current state
            return gameEvaluation(initial_state, current_state)

        def back_propagation(node, reward):
            while node != None:
                node.visit_times = node.visit_times + 1 # number of times current node is visited
                node.score = node.score + reward # accumulated reward
                node = node.parent # move to parent node

        # create root node with initial state
        root = Node()
        initial_state = state
        
        while True:
            vl = tree_policy(root, state)
            if vl == None:
                break
            
            reward = default_policy(vl, state)
            if reward == None:
                break
            
            back_propagation(vl, reward)
        
        # find the most visited node and return the action associated with it
        if (len(root.children) > 0):
            most_visited_times = max(root.children, key = lambda x: x.visit_times).visit_times
            for child in root.children:
                if child.visit_times == most_visited_times:
                    return child.action
        return Directions.STOP
