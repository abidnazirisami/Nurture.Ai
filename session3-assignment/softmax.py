import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  it = 0
  for x in X:
    f = np.dot(x, W)
    f -= np.max(f)
    
    p = np.exp(f)/np.sum(np.exp(f))
    local_loss = -1.0*np.log(p[y[it]])
    
    loss += local_loss
    
    for i in range(W.shape[1]):
      dW[:, i] += (p[i] - 1*(i==y[it]))*x
    it+=1
  loss = loss/X.shape[0] + 0.5*reg*(np.sum(np.square(W)))
  dW = dW/X.shape[0] + reg*(np.sum(W))
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = np.dot(X, W)
  f -= np.max(f, axis=1, keepdims=True)
  ind = [np.arange(y.shape[0]), y]
  p = np.exp(f)/np.sum(np.exp(f), axis=1, keepdims=True)
  loss = np.sum(-1.0*np.log(p[ind]))
  o = np.zeros_like(p)
  o[ind]=1
  dW = np.dot(X.T, (p - o))
  loss = loss/X.shape[0] + 0.5*reg*(np.sum(np.square(W)))
  dW = dW/X.shape[0] + reg*(np.sum(W))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

