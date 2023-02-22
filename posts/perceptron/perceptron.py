import numpy as np

class Perceptron:
    def __init__(self, w = None, history = []):
        self.w = w
        self.history = history
    
    def fit(self, X, y, max_steps):
        # reset history to avoid accumulating history in multiple datasets
        self.history = []
        
        # unpack the tuple returned by X.shape by assigning the number of data points to n, and the number of features to p
        n, p = X.shape
        
        # initialize a random initial weight vector w_tilt(0)
        # not sure here, how -b plays a role
        self.w = np.random.uniform(-1,1, size=(p+1)) # this should not be an array, but a 1D thing
        # print(self.w.shape)
        # print(self.w)
        
        # modify X into X_ (which contains a column of 1s and corresponds to X_tilde)
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        # print(X_.shape)
        # print(X_)
        
        for t in range(max_steps):
            # pick a random number within n
            i = np.random.randint(n)
            
            # compute the prediction and update weight
            y_tilde_i = 2 * y[i] - 1  # convert y to {-1, 1}
            x_tilde_i = X_[i]
            y_tilde_pred = int(y_tilde_i * np.dot(x_tilde_i, self.w) < 0)
            
            # print(f"{y_tilde_pred=}")
            # print(f"{y_tilde_i=}")
            # print(f"{x_tilde_i=}")
            
            self.w += y_tilde_pred * y_tilde_i * x_tilde_i
            
            # compute the score and add to history
            score = self.score(X, y)
            self.history.append(score)
            
            # check if the accuracy is 1; if so, terminate the loop
            if score == 1:
                break          
       
    
    def predict(self, X):
        # compute X_tilde 
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        # compute predictions
        y_pred = np.sign(X_ @ self.w)

        # convert {-1, 1} back to {0, 1}
        y_pred = (y_pred + 1) / 2

        return y_pred
    
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        return accuracy 

            