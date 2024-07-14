# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
main.py
~~~~~~~~~~

Main file for this project

Here I provide some examples for you to run easily,
you just need ton uncomment the part you want and comment what you don't want,
each part is independent of others
"""



"""
Watch games of snake played by my best neural nets !

Only 3 games are played here but you can load more networks from the saved folder if you wish
"""
from game import Game
from genetic_algorithm import GeneticAlgorithm
from neural_network import NeuralNetwork
import os


net = NeuralNetwork()
game = Game()

# Get the snake you want to watch from the args
npy = os.environ.get('SNAKE_NPY', '')
generate = os.environ.get('SNAKE_GENERATE', '')
if npy:
  net.load(filename_weights='saved/{0}_weights.npy'.format(npy), filename_biases='saved/{0}_biases.npy'.format(npy))
  game.start(display=True, neural_net=net)

elif generate:
  """
  Train your own snakes !

  Starts the genetic algorithm with parameters that I've already tested
  Best snake of each generation is saved in current folder
  The training speed depend a lot on your CPU and its cores number

  Contact me if you know how to make it run on GPU
  """
  gen = GeneticAlgorithm(population_size=1000, crossover_method='neuron', mutation_method='weight')
  gen.start()

else:
  """
  Play a game of snake !

  I do not recommend it as it is in first person and not that fun
  But if you want, you can
  """
  game = Game()
  game.start(playable=True, display=True, speed=10)








# Hey pssst, you, yes you.. Sometimes I boost training by making the snake already huge at the begining
# Also don't hesitate to put a iteration limit in the game loop (see game.py)