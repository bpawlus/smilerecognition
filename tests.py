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