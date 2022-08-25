# Deep Learning Exercises

This is a set of exercises about Deep Learning. These exercises will help you better familiarize with PyTorch and building models of fully connected networks, CNN, graph neural netowrks, attention mechanism and reinforcement learning.

The level of difficulty goes from 1 to 7 (1 being the easiest, 7 the hardest):

<ul>
  <li>Homework 1: FC network (1)</li>
  <li>Homework 2: CNN for classification (2) and image reconstruction (4)</li>
  <li>Homework 3: Graph neural network for classification (5) and for path prediction (6)</li>
  <li>Homework 4: Graph neural network with slot attention mechanism (7)</li>
  <li>Homework 5: Reinforcement learning to play the game of Pong (3)</li>
</ul>

### Homework 1: galaxies classification with FC network

The goal of the homework is to get used to the work flow and training a simple neural network:

<ul>
  <li>building a data loader</li>
  <li>building the model</li>
  <li>training the model</li>
</ul>

In this exercise you will have a main Jupyter notebook and some .py files (where you will have to write your data loader and model).

### Homework 2

<b>Part 1</b>: We want to use CNN to classify galaxies. We are going to take one of the pertained models and we will use that to build a new model.

<b>Part 2</b>: We want to build a CNN in which both input and output are images. Not a classifier anymore. This to start opening up the idea that output doesn't have to be just an array. This homework is more related on building the model.

### Homework 3: Graph neural network

<b>Part 1</b>: Point cloud classification.

<b>Part 2</b>: Message passing network (build edge and node networks). You REALLY need to understand what is happening in the code, otherwise you are lost... read about message passing if you are stuck!

### Homework 4: Graph neural network and attention mechanism

You will be using a GNN to build boxes around different cluster of data. This exercise will allow you to get familiar with key, query and value.

### Homework 5: RL for atari game

You have to build a RL network to play the game of Pong.
