import neat
import os
import pickle
from random import randint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

world_rows = 60
world_columns = 60
colors = ['white', 'yellow', 'red', 'blue', 'black']

#Represent movement vectors
class Dir():
  UP = [-1, 0]
  DOWN = [1, 0]
  LEFT = [0, -1]
  RIGHT = [0, 1]

  @classmethod
  def index(cls, key):
    if key == 0:
      return cls.UP
    elif key == 1:
      return cls.DOWN
    elif key == 2:
      return cls.LEFT
    elif key == 3:
      return cls.RIGHT

#Saves cell type and its one hot encoded array
class Cell:
  EMPTY = 0
  FOOD = 1
  ATTACK = 4
  BLOCK = 5

  def __init__(self, cell_type):
    self.cell_type = cell_type
    self.arr = [1 if i == cell_type else 0 for i in range(1, 6)]

  #Returns a 2D array of cells
  def getCellMap(map):
    return [[Cell(cell_type) for cell_type in row] for row in map]


class Organism:
  def __init__(self, 
               body=Cell.getCellMap([[0, 3, 0], [4, 2, 3], [0, 4, 0]]),
               pos_r=None,
               pos_c=None):
    
    #Position is random if not passed
    if pos_r==None:
      pos_r = randint(0, world_rows - len(body))
    if pos_c==None:
      pos_c = randint(0, world_columns - len(body[0]))

    #Position is the top left cell coordinate
    self.pos_r = pos_r
    self.pos_c = pos_c
    self.rows = len(body)
    self.columns = len(body[0])
    self.body = body
    self.atk_cell = [2, 1] #Position of attack cell relative to top left corner of body
    self.energy = 0
  
  #Returns a 5x5 2D array of cells surrounding organism
  def getState(self):
    state = []
    state_row = []
    for row in range(self.pos_r - 1, self.pos_r + self.rows + 1):
      for column in range(self.pos_c - 1, self.pos_c + self.columns +  1):
        if row < 0 or row >= world_rows or column < 0 or column >= world_columns:
          state_row.append(Cell(Cell.BLOCK))
        else:
          state_row.append(world.map[row][column])
      state.append(state_row)
      state_row = []
    return state

  #Moves organism taking a vector as input
  #Also updates energy
  def move(self, move):
    world.remove_organism(self)
    new_pos_r = self.pos_r + move[0]
    new_pos_c = self.pos_c + move[1]
    atk_cell_pos = [new_pos_r + self.atk_cell[0], new_pos_c + self.atk_cell[1]]

    #Check if move goes outside the world
    if new_pos_r < 0 or new_pos_r + self.rows > world_rows or new_pos_c < 0 or new_pos_c + self.columns > world_columns:
      world.place_organism(self)
      return False

    #Check if move causes collision
    for row in range(self.rows):
      for column in range(self.columns):
        if self.body[row][column].cell_type not in [Cell.EMPTY, Cell.ATTACK]:
          if world.map[new_pos_r + row][new_pos_c + column].cell_type != Cell.EMPTY:
            world.place_organism(self)
            return False
    
    #Check if new attack cell position overlaps with food
    if world.map[atk_cell_pos[0]][atk_cell_pos[1]].cell_type == Cell.FOOD:
      self.energy += 10
      world.remove_cell(atk_cell_pos[0], atk_cell_pos[1])

    self.pos_r = new_pos_r
    self.pos_c = new_pos_c
    world.place_organism(self)
    return True
  
  
class World:
  def __init__(self):
    self.map = [[Cell(Cell.EMPTY) for _ in range(world_columns)] for _ in range(world_rows)]

  #Displays current world map
  def show(self):
    cmap = ListedColormap(colors)
    cell_type_map = [[cell.cell_type + 1 for cell in row] for row in self.map]
    plt.imshow(cell_type_map, cmap=cmap, interpolation='nearest')
    plt.show()

  def is_empty(self, row, column):
    if self.map[row][column].cell_type == Cell.EMPTY:
      return True
    else:
      return False

  #Places food cell if cell type not passed
  def place_cell(self, 
               row = randint(0, world_rows-1), 
               column = randint(0, world_columns), 
               cell = Cell(Cell.FOOD)):
    self.map[row][column] = cell

  def remove_cell(self, row, column):
    self.map[row][column] = Cell(Cell.EMPTY)

  #Randomly places food cell or empty cell
  #Completely overwrites the world map
  def initialize_food(self):
    for row in range(world_rows):
      for column in range(world_columns):
        if randint(0, 100) < 10:
          self.place_cell(row, column)
        else:
          self.remove_cell(row, column)

  def place_organism(self, organism):
    for i in range(organism.rows):
      for j in range(organism.columns):
        if organism.body[i][j].cell_type != Cell.EMPTY:
          self.map[organism.pos_r + i][organism.pos_c + j] = organism.body[i][j]
  
  def remove_organism(self, organism):
    for i in range(organism.rows):
      for j in range(organism.columns):
        if organism.body[i][j].cell_type != Cell.EMPTY:
          self.remove_cell(organism.pos_r + i, organism.pos_c + j)

#Flattens 5x5 2D array of cells to 1D array
def flatten(state):
  input = []
  for row in state:
    for cell in row:
      input.extend(cell.arr)
  return input 

def get_fitness(genomes, config):
    for _, genome in genomes:
        
        #Initialize game
        genome.fitness = 0
        world.initialize_food() #Resets world map and places food for each new game
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        organism = Organism()
        world.place_organism(organism)

        for _ in range(100):
            state = organism.getState()
            output = net.activate(flatten(state))
            move = Dir.index(output.index(max(output)))
            if not organism.move(move):
                break
            
            #Other reward conditions
            # for row in range(5):
            #     for column in range(5):
            #         if state[row][column].cell_type == Cell.FOOD:
            #           if row in [0, 4] or column in [0, 4]:
            #             genome.fitness += 1
            #           if row in [1, 3] or column in [1, 3]:
            #             genome.fitness += 2
        genome.fitness += organism.energy

def train(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)    
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    return p.run(get_fitness, 5000)

def save(genome):
  with open("winner.pkl", "wb") as f:
    pickle.dump(genome, f)
    f.close()

def test(config_file):
  config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_file)  
  
  cmap = ListedColormap(colors)
  fig, ax = plt.subplots()
  artists = []

  with open("winner.pkl", "rb") as f:
    genome = pickle.load(f)
    genome.fitness = 0
    old_energy = 0
    world.initialize_food()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    organism = Organism()
    world.place_organism(organism)

    for i in range(100):
        print(genome.fitness)

        state = organism.getState()
        output = net.activate(flatten(state))
        move = Dir.index(output.index(max(output)))
        if not organism.move(move):
            break
        
        cell_type_map = [[cell.cell_type + 1 for cell in row] for row in world.map]
        img = ax.imshow(cell_type_map, cmap=cmap)
        title = ax.set_title(f'Frame {i}')
        artists.append([img, title])

        # for row in range(5):
        #     for column in range(5):
        #         if state[row][column].cell_type == Cell.FOOD:
        #           if row in [0, 4] or column in [0, 4]:
        #             genome.fitness += 1
        #           if row in [1, 3] or column in [1, 3]:
        #             genome.fitness += 2

        genome.fitness += (organism.energy - old_energy)
        old_energy = organism.energy

    ani = animation.ArtistAnimation(fig, artists, interval=200, blit=True, repeat=False)
    ani.save(filename="winner.mp4", writer="ffmpeg")
    plt.show()


world = World()

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')

train = 0 # 1 means train, 0 means simulate

if train == 1:
  winner = train(config_path)
  save(winner)
else:
  test(config_path)
