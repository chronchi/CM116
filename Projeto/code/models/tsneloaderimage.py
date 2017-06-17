import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as skm
import numpy.random as npr

compo = 2
from next_batch$compo import *
path = '../dataset/zsample'
if compo == 2:
    vector = npr.choice(range(0,272), 272, replace=False)
else:
    vector = np.load('../dataset/vectors/vector5.npy')

#load data
xdata, ydata = next_batch(vector, path, 1, 32)

#convert the ydata one hot to a one column vector
ydata1d = np.zeros(ydata.shape[0], dtype=np.int)
for i in range(0,ydata.shape[0]):
    ydata1d[i] = np.argmax(ydata[i,:])

model = skm.TSNE(n_components=2, perplexity=50.0, n_iter=10000, learning_rate=100.0,
early_exaggeration=.01)
xdata2d = model.fit_transform(xdata)

plot = plt.scatter(xdata2d[:,1], xdata2d[:,1], c=ydata1d)

plt.show()
