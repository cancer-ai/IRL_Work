import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OthelloEnv(gym.Env):
    """
    Othello is a strategy board game played on an 8x8 board.
    The goal is to have the majority of disks turned to display your color
    when the last playable empty square is filled.

    ## Action Space
    The action is a tuple (x, y) representing where to place a disk.

    ## Observation Space
    The observation is an 8x8 grid representing the board state.
    -1 represents black disks, 1 represents white disks, 0 represents empty.

    ## Episode Termination
    The episode ends when the board is full or no valid moves are available.
    """

    def __init__(self, render_mode=None):
        self.board_size = 8
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 1 for white, -1 for black
        self.render_mode = render_mode

        # Initialize the board with starting positions
        self._initialize_board()

    def step(self, player, action):
        x, y = divmod(action, self.board_size)
        if not self._is_valid_move((x, y)):
            return self._get_obs(), -1, True, {}

        self._make_move((x, y))
        self.current_player *= -1  # Switch player

        done = self._game_over()
        reward = self._calculate_reward() if done else 0
        return self._get_obs(), reward, done, {}

    def reset(self, player1_state=None, player2_state=None):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self._initialize_board()
        if player1_state is not None:
            self.board = player1_state
        if player2_state is not None:
            self.board = player2_state
        return self._get_obs()

    def render(self, mode='human'):
        if mode == 'human':
            print(self.board)

    def _get_obs(self):
        return self.board

    def _initialize_board(self):
        mid = self.board_size // 2
        self.board[mid-1][mid-1] = self.board[mid][mid] = -1
        self.board[mid-1][mid] = self.board[mid][mid-1] = 1

    def _is_valid_move(self, position):
        x, y = position

        # Check if the position is within the board and the cell is empty
        if not (0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == 0):
            return False

        # Check each direction for bracketing opponent pieces
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip checking the current position

                if self._can_flip(x, y, dx, dy):
                    return True  # Valid move found in this direction

        return False  # No valid moves found

    def _has_valid_moves(self, player):
        """
        Check if there are any valid moves available for a given player.
        """
        self.current_player = player
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self._is_valid_move((x, y)):
                    return True
        return False

    def _can_flip(self, x, y, dx, dy):
        """
        Check if placing a piece at (x, y) will flip any pieces in a given direction (dx, dy).
        """
        x, y = x + dx, y + dy
        found_opponent_piece = False

        while 0 <= x < self.board_size and 0 <= y < self.board_size:
            if self.board[x, y] == 0:
                return False  # Empty space, stop checking this direction
            if self.board[x, y] == self.current_player:
                return found_opponent_piece  # Valid move if at least one opponent piece is bracketed

            found_opponent_piece = True
            x, y = x + dx, y + dy

        return False  # Reached the edge of the board without finding a valid move
    def _make_move(self, position):
        x, y = position
        self.board[x, y] = self.current_player

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the case where both dx and dy are 0

                self._flip_pieces(x, y, dx, dy)

    def _flip_pieces(self, x, y, dx, dy):
        """
        Flip pieces in a given direction (dx, dy) starting from (x, y).
        """
        x, y = x + dx, y + dy
        pieces_to_flip = []

        while 0 <= x < self.board_size and 0 <= y < self.board_size:
            if self.board[x, y] == 0:
                break  # No more pieces to flip in this direction
            if self.board[x, y] == self.current_player:
                # Found a piece of the current player's color, flip the bracketed pieces
                for px, py in pieces_to_flip:
                    self.board[px, py] = self.current_player
                break

            pieces_to_flip.append((x, y))
            x, y = x + dx, y + dy

    def _game_over(self):
        # Check if the board is full
        if not np.any(self.board == 0):
            return True

        # Check for valid moves for the current player
        if self._has_valid_moves(self.current_player):
            return False

        # Check for valid moves for the opponent
        if self._has_valid_moves(-self.current_player):
            return False

        return True  # No valid moves for either player

    def _calculate_reward(self):
        # Count the pieces for each player
        white_count = np.sum(self.board == 1)
        black_count = np.sum(self.board == -1)

        # Determine the game outcome
        if white_count > black_count:
            return 1 if self.current_player == 1 else -1  # White wins
        elif black_count > white_count:
            return 1 if self.current_player == -1 else -1  # Black wins
        else:
            return 0  # Draw
