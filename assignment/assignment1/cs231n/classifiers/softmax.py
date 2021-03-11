from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    A = X @ W
    for i in range(X.shape[0]):
        A[i] = A[i] - np.max(A[i])
        y_pred = np.exp(A[i])/np.sum(np.exp(A[i]))
        loss += -np.log(y_pred[y[i]])
        for j in range(W.shape[1]):
            dW[:,j] += X[i] * y_pred[j]
        dW[:,y[i]] -= X[i]
    loss /= X.shape[0]
    dW /= X.shape[0]
    loss += reg * np.sum(W * W) / 2
    dW += reg * W 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    A = X @ W
    A -= np.max(A, axis=1, keepdims=True)
    y_pred = np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)
    loss = np.sum(-np.log(y_pred[np.arange(num_train), y]))
    loss /= num_train
    y_pred[np.arange(num_train), y] -= 1
    dW = X.T @ y_pred
    dW /= num_train
    loss += reg * np.sum(W * W) / 2
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
