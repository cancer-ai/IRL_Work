class Othello_Board:
    def __init__(self, size=8):
        self.size = size
        # Put gaps (' ') in a list to denote an empty space.
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.initialize_board()

    def initialize_board(self):
        # Othello always starts with two black and two white in opposing diagonals
        # in the middle of the board.
        mid = self.size // 2
        self.board[mid - 1][mid - 1] = 'W'
        self.board[mid - 1][mid] = 'B'
        self.board[mid][mid - 1] = 'B'
        self.board[mid][mid] = 'W'

    def display_board(self):
        print('  ' + '  '.join(str(i) for i in range(self.size)))
        for i in range(self.size):
            print(f"{i} {'  '.join(str(self.board[i][j]) for j in range(self.size))}")

    def is_valid_move(self, row, col, player):
        # If the location on the grid is not empty then putting a piece in that spot will be
        # considered an invalid move.
        if self.board[row][col] != ' ':
            return False

        other_player = 'B' if player == 'W' else 'W'

        # Directions correspond to N, NE, E, SE, S, SW, W, NW
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, -1), (-1, 0), (-1, 1)]

        for d_row, d_col in directions:
            r, c = row + d_row, col + d_col
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == other_player:
                r += d_row
                c += d_col

            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                return True
        return False

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            print("Invalid move!")
            return

        self.board[row][col] = player
        other_player = 'B' if player == 'W' else 'W'

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == other_player:
                to_flip.append((r, c))
                r += dr
                c += dc

            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                for flip_row, flip_col in to_flip:
                    self.board[flip_row][flip_col] = player

        self.display_board()


# Example:
othello = Othello_Board()
othello.display_board()

othello.make_move(7, 3, 'B')
othello.make_move(2, 4, 'W')

othello.display_board()