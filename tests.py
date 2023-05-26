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
df.columns = ["e","d1","d2","d3"]
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

input = torch.randn(5)
m = nn.BatchNorm1d(5)
r = nn.ReLU(inplace=False)
output = m(input)
output2 = r(output)
print(input)