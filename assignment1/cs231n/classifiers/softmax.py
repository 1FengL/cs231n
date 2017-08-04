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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
        y_pred = X[i,:].dot(W)
        y_pred -= np.max(y_pred) #avoid numerical instability:exp can explode
        correct_pred = y_pred[y[i]]
        
        exp_sum = np.sum(np.exp(y_pred))
        loss += np.log(exp_sum) - correct_pred
        
        dW[:,y[i]] -= X[i,:]
        for j in xrange(num_classes):
            dW[:,j] += np.exp(y_pred[j]) / exp_sum * X[i,:] 
        
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
        
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
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  XW = X.dot(W)
  XW -= np.max(XW, axis=1, keepdims=True)
  expXW = np.exp(XW)
  p = expXW / np.sum(expXW,axis=1, keepdims =True)
  y_trueclass = np.zeros_like(p)
  y_trueclass[xrange(num_train), y] = 1
  loss += np.sum(np.log(np.sum(expXW,axis=1)))
  loss -= np.sum(XW[xrange(num_train),y])
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
    
  dW = np.dot(X.T, p - y_trueclass)
  dW /= num_train
  dW += reg * W   

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

