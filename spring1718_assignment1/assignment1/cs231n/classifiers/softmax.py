import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  dScores = np.zeros_like(scores)
  # stability
  scores -= np.max(scores,axis = 1,keepdims = True)
  scores = np.exp(scores)
  
  for i in range(num_train):
      correct_scores = scores[i,y[i]]
      total_score = 0.0
      dScores[i,:] += scores[i,:]/np.sum(scores[i,:])
      dScores[i,y[i]] -= 1
      for j in range(num_classes):
          total_score += scores[i,j]
      loss -= np.log(correct_scores/total_score)

  
  loss = loss/num_train + reg*np.sum(W*W)
  dW = X.T.dot(dScores)
  dW = dW/num_train + 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  DScores = np.zeros_like(scores)
  scores = scores - np.max(scores,axis = 1,keepdims = True)
  correct_score = scores[np.arange(num_train),y] 
  
  loss = - np.sum(np.log(np.exp(correct_score)/np.sum(np.exp(scores),axis = 1)))
  loss = loss/num_train + reg*np.sum(W*W)
  
  DScores = np.exp(scores)/np.sum(np.exp(scores),axis = 1,keepdims = True)
  DScores[np.arange(num_train),y] -= 1
  DScores = DScores/num_train
  
  dW = X.T.dot(DScores)
  dW = dW + 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

