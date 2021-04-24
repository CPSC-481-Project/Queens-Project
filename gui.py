import pygame 
import sys
import math

# Start pygame
pygame.init()

# creates screen
size = width, height = 512, 512

white = 255, 178, 102
black = 255, 128, 0
highlight = 192, 192, 192

width = 64 #width of each square
original_color = ' '

screen = pygame.display.set_mode(size)

rect_list = list()

# Title
pygame.display.set_caption("NxN Queens Project")

# Loop creating list of brown rectangles
for i in range(0, 8):
    for j in range(0, 8):
        if i % 2 == 0:
            if j % 2 != 0:
                rect_list.append(pygame.Rect(j * 64, i * 64, 64, 64))
        else:
            if j % 2 == 0:
                rect_list.append(pygame.Rect(j * 64, i * 64, 64, 64))

chess_board_surface = pygame.Surface(size)
chess_board_surface.fill(white)

# counts all the queens on the board
def queen_loc(screen):
    n = len(screen)
    loc = 0
    
    for i in range(n):
        if screen[i] != -1:
            loc += 1
        
    return loc

def validMove(self, pos, loc):
    conflict = 0  #conflict between queens if near the movememnt of one another
    board = self.screen
    col = 0

    for queen_loc in pos:
        
        for i in range(col - 1, -1, -1):
            if board[loc][i] == 1:
                conflict += 1 

        for i, j in zip(range(loc - 1, -1, -1), range(col - 1, -1, -1)):
            if board[i][j] == 1:
                conflict += 1
        for i, j, in zip(range(loc, - 1, self.screen, -1), range(col - 1, -1, -1)):
            if board[i][j] == 1:
                conflict += 1
        col += 1
    return conflict  #suppose to return 0 to represent that there are no hits within this algorithm

# Game Loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            x = math.floor(pos[0] / width)
            y = math.floor(pos[1] / width)
            original_color = chess_board_surface.get_at((x * width, y ( width)))
            pygame.draw.rect(chess_board_surface, highlight, pygame.Rect((x) * width, (y) * width, 64, 64))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            x = math.floor(pos[0] / width)
            y = math.floor(pos[1] / width)
            pygame.draw.rect(chess_board_surface, original_color, pygame.Rect((x) * width, (y) * width, 64, 64))

    screen.blit(chess_board_surface, (0, 0))
    pygame.display.update()
