# Deep Learning Exercises

This is a set of exercises about Deep Learning. They are intented to people that have played with AI in the past and would like to improve their expertise and, maybe, apply neural netowrks in their research.

These exercises will help you better familiarize with PyTorch and building models of fully connected networks (FC), convolutional neural networks (CNN), graph neural networks (GNN), attention mechanism and reinforcement learning (RL).

A good introduction to these concepts can be found <a href="http://cs231n.stanford.edu/">here</a> and <a href="http://introtodeeplearning.com/">here</a>.

<!-- 
The level of difficulty goes from 1 to 7 (1 being the easiest, 7 the hardest):

<ul>
  <li>Exercise 1: FC network (1)</li>
  <li>Exercise 2: CNN for classification (2); image reconstruction (4)</li>
  <li>Exercise 3: Graph neural network for classification (5) and for path prediction (6)</li>
  <li>Exercise 4: Graph neural network with slot attention mechanism (7)</li>
  <li>Exercise 5: Reinforcement learning to play the game of Pong (3)</li>
</ul> -->

### Exercise 1: galaxies classification with FC network

The goal of the exercise is to get used to the work flow and training a simple neural network. Important building blocks will be:

<ul>
  <li>building a data loader</li>
  <li>building the model</li>
  <li>training the model</li>
</ul>

In this exercise you will have a main Jupyter notebook and some .py files (where you will have to write your data loader and model).

### Exercise 2: CNN for classification and image reconstruction

<b>Part 1</b>: We want to use CNN to classify galaxies. We are going to take one of the pretrained models and we will use that to build a new model.

<b>Part 2</b>: We want to build a CNN in which both input and output are images. Not a classifier anymore. This to start opening up the idea that output doesn't have to be just an array. This homework is more related on building the model.

### Exercise 3: Graph neural network

<b>Part 1</b>: Point cloud classification.

<b>Part 2</b>: Message passing network (build edge and node networks). You REALLY need to understand what is happening in the code, otherwise you are lost... read about message passing if you are stuck!

### Exercise 4: Graph neural network and attention mechanism

You will be using a GNN to build boxes around different clusters of data. This exercise will allow you to get familiar with the concepts of “key, query and value.”

### Exercise 5: RL for atari game

You have to build a RL network to play the game of Pong.
