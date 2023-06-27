# import matplotlib.pyplot as plt
# import pandas as pd
#
# epoch = [1,2,3,4,5]
# acc = [0.5,0.7,0.8,0.85,0.9]
# acc2 = [0.5,0.6,0.65,0.675,0.69]
# loss = [60,40,25,15,10]
# loss2 = [60,45,35,25,20]
#
# fig,ax=plt.subplots()
# ax.set_xlabel("Epoch")
#
# ax.set_ylabel("Loss", color="red", fontsize=14)
# line1, = ax.plot(epoch, loss, color="red", marker="o")
# line2, = ax.plot(epoch, loss2, color="orange", marker="o")
#
# ax2 = ax.twinx()
# ax2.set_ylabel("Acc", color="blue", fontsize=14)
# line3, = ax2.plot(epoch, acc, color="blue", marker="o")
# line4, = ax2.plot(epoch, acc2, color="cyan", marker="o")
#
# ax2.legend([line1, line2, line3, line4], ['Loss1', 'Loss2', 'Acc1', 'Acc2'], loc = 'upper center')
#
# fig.savefig('model_acc.jpg',format='jpeg',dpi=100,bbox_inches='tight')
# plt.show()


# import numpy
#
# a = [[j+i for j in range(5)] for i in range(3)]
# a = numpy.asarray(a)
# a1 = a[1:]
# a2 = a[:-1,]
# a2-a1

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torchvision
from torchvision import datasets,transforms
import torchvision.transforms as transforms

"""
import numpy
import pandas as pd
a = numpy.asarray([[1,2,3,4,5,6,7,8,9], [1.04,2.02,3.12,4.42,5.13,6.14,7.15,8.16,9.16], [1.04,2.02,3.12,4.42,5.13,6.14,7.15,8.16,9.16], [1.04,2.02,3.12,4.42,5.13,6.14,7.15,8.16,9.16]])
df = pd.DataFrame(a).transpose()
df.columns = ["e","d[i]","d[i+1]","d[2]"]
print(df)
df.to_csv("tests/foo.csv", index=False, sep=" ")
"""

"""
import numpy
import pandas as pd
res = {"a":[1,1,1,0,1],"b":[0,0,1,1,0]}
rest ={"a":[1],"b":[0]}

df = pd.DataFrame.from_dict(res).transpose()
df.columns = [f"E{int(name)+1}" for name in list(df.columns.values)]
df2 = pd.DataFrame.from_dict(rest).transpose()
df2.columns = ["true"]

df = df2.join(df)
print(df)
"""

d = [np.array([32, 36, 27, 28, 30, 31]), np.array([32, 34, 30, 33, 29, 36, 24]), np.array([39, 40, 42])]

n1,n2,n3 = len(d[0]),len(d[1]),len(d[2])
dp = np.hstack([d[0],d[1],d[2]])
m1,m2,m3,mp = d[0].mean(), d[1].mean(), d[2].mean(),dp.mean()
v1,v2,v3,vp = d[0].var(), d[1].var(), d[2].var(),dp.var()

print ("Means:",m1,m2,m3)
print ("Variances:",v1,v2,v3)
print ("Means p:",mp)
print ("Variances p:",vp)

ap = (n1*m1 + n2*m2 + n3*m3) / (n1+n2+n3) 
mean_of_var = (n1*v1 + n2*v2 + n3*v3) / (n1+n2+n3) 
var_of_means = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2) / (n1+n2+n3)
print ("Alt variances:",mean_of_var + var_of_means)

n_prev = len(d[0])
d_prev = d[0].mean()
v_prev = d[0].var()
for i in range(2):
    n1,n2 = n_prev, len(d[i+1])
    m1,m2 = d_prev, d[i+1].mean()
    v1,v2 = v_prev, d[i+1].var()

    print (f"{i} Means:",m1,m2)
    print (f"{i} Variances:",v1,v2)

    ap = (n1*m1 + n2*m2) / (n1+n2)
    mean_of_var = (n1*v1 + n2*v2) / (n1+n2)
    var_of_means = (n1*(m1-ap)**2 + n2*(m2-ap)**2) / (n1+n2)
    print (f"{i} Alt variances:",mean_of_var + var_of_means)

    n_prev = n_prev + n2
    d_prev = ap
    v_prev = mean_of_var + var_of_means
    print(f"{i} Alt means:",ap)