'''
This is a small toy example for educational purposes. 
Here I'm trying to predict the middle part of Rule 30 of cellular automata using a simple neural network. 
To Learn more check out the associated YouTube Video: https://youtu.be/TT4I2VIPYrg
'''

import numpy as np
import matplotlib.pyplot as plt

# define the rule 
rule = {}
rule[(1,1,1)] = 0
rule[(1,1,0)] = 0
rule[(1,0,1)] = 0
rule[(1,0,0)] = 1
rule[(0,1,1)] = 1
rule[(0,1,0)] = 1
rule[(0,0,1)] = 1
rule[(0,0,0)] = 0

# define the grid
steps = 1000
size = steps*2
grid = np.zeros((steps, size))
mid = size // 2
grid[0, mid] = 1

# apply the rule to grid
for i in range(steps-1):
    for j in range(size-2):
        a = grid[i, j]
        b = grid[i, j+1]
        c = grid[i, j+2]       
        grid[i+1, j+1] = rule[(a,b,c)]
                 
# visualize
plt.figure(1)
plt.imshow(grid, cmap='binary')
plt.show()

#define the input and target
t = 2 #time set t 2 as we are trying to directly predict 2 steps ahead 
xs = grid[:-t, mid-2:mid+1]
ys = grid[t:, mid][:, None]

# define neural network structure
ins = xs.shape[1]
outs = 1
w = np.random.randn(ins, outs)*0.1
f = lambda x: 1 / (1 + np.exp(-x)) 
df = lambda x: x * (1 - x)

# train the network
for i in range(1000):  
    yh = f(xs @ w)
    e = (yh - ys) 
    dw = (xs.T @ (e * df(yh)))
    w -= dw * 0.01 #learning rate

#binarize output and check the accuracy
yh = (yh > 0.5).astype(int)
corrects = ys == yh
accuracy = np.sum(corrects)/np.size(corrects)
print("accuracy:", accuracy)
