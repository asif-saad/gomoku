
#----------------------------------------------------------------------------
# Created By  : Asif Saad, Shihab Newaz
# Created Date: 26.05.2023
# ---------------------------------------------------------------------------
# This is a simple MakeC engine embedded in class that you can play with (and certainly can win).
# This AI simply predicts 3 next moves with some dummy evaluation function and selects the best from minimax with alpha/beta.
# Current recursion depth is 3 and I did not manage to increase it on pure python without major speed loss, still game AI is decent to play with.






import numpy as np
import random


class MakeCEngine:
    def __init__(self):
        self.board = np.full((15,15),-1, dtype=np.dtype(int)) #board to store moves. If user moves first: -1 - empty, 0 - user, 1 - AI
        self.positive_directions = [(0,1),(1,1),(1,0),(1,-1)] #direction vectors, that should be considered in evaluated multiplied by 1 and -1
        self.ai_depth = 2 #number of moves to forecast during AI move

    
    def getChildren(self, only_closest_moves = False): #returns coords on board that should be considered as possible moves. Dummy for now.
        if not only_closest_moves:
            positions = np.where(self.board < 0)
            positions = list(zip(positions[0],positions[1]))
        else:
            positions = np.where(self.board >= 0)
            positions = list(zip(positions[0],positions[1]))
            buf_positions = []
            for position in positions:
                for direction in self.positive_directions:
                    new_x = position[0] + direction[0]
                    new_y = position[1] + direction[1]
                    if new_x >= 0 and new_x <= 14 and new_y >= 0 and new_y <= 14 and self.board[new_x][new_y] == -1:
                        buf_positions.append((new_x, new_y))
                    new_x = position[0] - direction[0]
                    new_y = position[1] - direction[1]
                    if new_x >= 0 and new_x <= 14 and new_y >= 0 and new_y <= 14 and self.board[new_x][new_y] == -1:
                        buf_positions.append((new_x, new_y))
            positions = buf_positions
        return positions

    def evaluate(self, last_moves): #Very simple static evaluation, only last move is taken into consideration. Counts number of line segments that this move appears in and returns score for the move
        evaluation = 0
        init_move_switch = self.board[last_moves[0]]
        move_switch = init_move_switch
        for position in last_moves:
            self.board[position] = move_switch
            move_switch = (move_switch + 1) % 2
            turn = self.board[position]
            multiplier = 1
            if turn > 0:
                multiplier = -0.5
            for direction in self.positive_directions:
                line_count = 1
                can_move = True
                cur_position = list(position)
                is_blocked1 = True
                while can_move:
                    cur_position[0] += direction[0]
                    cur_position[1] += direction[1]
                    if not (cur_position[0] >= 0 and cur_position[0] <= 14 and cur_position[1] >= 0 and cur_position[1] <= 14):
                        # checking if within board
                        can_move = False
                    else:
                        # if current position within board
                        if self.board[tuple(cur_position)] == turn:
                            # when current position and actual position have same turn
                            line_count += 1
                            if line_count >= 5:
                                ## if all 5 points match?
                                for position in last_moves:
                                    self.board[position] = init_move_switch
                                    init_move_switch = (init_move_switch + 1) % 2
                                return multiplier * float('inf')
                        else: 
                            if self.board[tuple(cur_position)] == -1:
                                # it is set to initial, no one accessed this cell yet
                                is_blocked1 = False
                            can_move = False
                # reintroducing for again evaluation
                cur_position = list(position)
                can_move = True
                is_blocked2 = True
                while can_move:
                    # doing the reverse of previous can_move while loop like the convolution
                    cur_position[0] -= direction[0]
                    cur_position[1] -= direction[1]
                    if not (cur_position[0] >= 0 and cur_position[0] <= 14 and cur_position[1] >= 0 and cur_position[1] <= 14):
                        can_move = False
                    else:
                        if self.board[tuple(cur_position)] == turn:
                            line_count += 1
                            if line_count >= 5:    
                                for position in last_moves:
                                    self.board[position] = init_move_switch
                                    init_move_switch = (init_move_switch + 1) % 2
                                return multiplier * float('inf')
                        else:
                            if self.board[tuple(cur_position)] == -1:
                                is_blocked2 = False
                            can_move = False
                cur_position = list(position)
                mul_blocked = 1
                if line_count == 4:
                    mul_blocked = 4
                elif line_count == 3:
                    mul_blocked = 2
                if not is_blocked1:
                    mul_blocked *= line_count
                if not is_blocked2:
                    mul_blocked *= line_count
                if line_count > 1:
                    evaluation += multiplier * line_count * mul_blocked
            if evaluation == 0:
                evaluation = multiplier
        return evaluation


    def get_all_neigh_possible_moves(self, position): #Simple helper function to get possible neighbor-cells for moves
        result = []
        for direction in self.positive_directions:
                new_x = position[0] + direction[0]
                new_y = position[1] + direction[1]
                if new_x >= 0 and new_x <= 14 and new_y >= 0 and new_y <= 14 and self.board[new_x][new_y] == -1:
                    result.append((new_x, new_y))
                new_x = position[0] - direction[0]
                new_y = position[1] - direction[1]
                if new_x >= 0 and new_x <= 14 and new_y >= 0 and new_y <= 14 and self.board[new_x][new_y] == -1:
                    result.append((new_x, new_y))
        return result
    
    def minimax(self, last_moves, d, min_val, max_val, only_closest = True, children = None): #recursive minimax with alpha-beta that performs evaluation in every node, not only in leafs (thus evaluation should be very cheap)

        if d == 0:
            cur_evalutaion = self.evaluate(last_moves)
            return cur_evalutaion, d
        last_move = last_moves[-1]
        turn = (self.board[last_move] + 1) % 2
        if children == None:
            children = set(self.getChildren(only_closest))
        d1 = d
        if turn == 0: #we are in max node - user move
            v = min_val
            for coord in children:
                # iterating for all possible children
                if self.board[coord] != -1:
                    continue
                self.board[coord] = turn
                
                children = children.difference([coord])
                # subtracting current co-ordinate from children list
                buf = self.get_all_neigh_possible_moves(coord)
                # all possible neighbors of the child node
                children = children.union(buf)
                # merge possible neighbors with children list
                last_moves.append(coord)
                v1, d1 = self.minimax(last_moves, d-1, v, max_val, only_closest, children)
                last_moves.pop()
                children = children.difference(buf)
                children = children.union([coord])
                self.board[coord] = -1
                if v1 > v:
                    v = v1
                if v >= max_val:
                    return max_val, d1    
            return v, d1
        if turn == 1: #we are in min node - AI move
            v = max_val
            for coord in children:
                if self.board[coord] != -1:
                    continue
                self.board[coord] = turn
                children = children.difference([coord])
                buf = self.get_all_neigh_possible_moves(coord)
                children = children.union(buf)
                last_moves.append(coord)
                v1, d1 = self.minimax(last_moves, d-1, min_val, v, only_closest, children)
                last_moves.pop()
                children = children.difference(buf)
                children = children.union([coord])
                self.board[coord] = -1
                if v1 < v:
                    v = v1
                if v <= min_val:
                    return min_val, d1         
            return v, d1
    
    def find_closest_move(self, move_hint): # first move selection during AI move analysis
        min_dist = float('inf')
        min_move = (-1,-1)
        possible_moves = self.getChildren()
        for move in possible_moves:
            dist = np.abs(move_hint[0] - move[0]) + np.abs(move_hint[1] - move[1])
            if dist < min_dist:
                min_dist = dist
                min_move = move
        return min_move
    
    def ai_move(self, ai_depth, cur_turn, move_hint, only_closest = True, for_profiler = False): 
        #main function to choose AI move and update board evaluation
        best_move = self.find_closest_move(move_hint)
        best_evaluation = float('inf')
        best_d = 0
        possible_moves = set(self.getChildren(only_closest))
        d1 = 0
        for move in possible_moves:
            self.board[move] = cur_turn
            children = possible_moves.difference([move])
            evaluation, d1 = self.minimax([move], ai_depth, -1*float('inf'), float('inf'), only_closest, children)
            # sending every possible moves one by one to minimax function
            children = possible_moves.union([move])
            self.board[move] = -1
            if evaluation < best_evaluation or (evaluation == best_evaluation and d1 > best_d):
                best_evaluation = evaluation
                best_move = move
                best_d = d1
                if best_evaluation == -1 * float('inf') and best_d == ai_depth:
                    break
            elif (evaluation == best_evaluation and d1 == best_d):
                if random.randint(0,1) > 0:
                    best_move = move

        if not for_profiler:
            self.board[best_move] = cur_turn
        last_eval = self.evaluate([best_move])
        return last_eval, best_move
    
    def can_move(self, move):
        return self.board[move] == -1
    
    def user_move(self, cur_turn, cur_move):
        self.board[cur_move] = cur_turn
        last_eval = self.evaluate([cur_move])
        return last_eval
    
    def reset_board(self):
        self.board = np.full((15,15),-1, dtype=np.dtype(int))
        return