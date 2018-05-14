import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


fig = plt.figure(figsize = (8, 4))

filenames = [
    # 'csv/decoder-fc-only.csv',
    # 'csv/decoder-conv3.csv',
    # 'csv/decoder-up-conv1.csv',
    # 'csv/decoder-conv3-2-256.csv',
    # 'csv/decoder-conv3-1000.csv',
    'csv/decoder-conv3-1000-3lines.csv',
    # 'csv/decoder-conv3-0.0001.csv',
    # 'csv/decoder-conv3-0.001.csv',
    # 'csv/decoder-conv2-0.001.csv',
]

for filename in filenames:
    tmp = np.loadtxt(filename, delimiter=',', dtype=str)
    labels = tmp[0,:]
    data = tmp[1:,:].astype(np.float)
    # max_x = data[1:,0].max()

    for i in range(1, data.shape[1]):
        x, y = data[1:,0], data[1:,i]
        xvals = np.linspace(0, max(data[1:,0]), 50)
        yinterp = np.interp(xvals, x, y)
        fx = interpolate.interp1d(xvals, yinterp, kind='cubic')
        # plt.plot(x, y, '--', label=labels[i])
        plt.plot(x, fx(x), '-', label=filename + ' - ' + labels[i])
    
plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.)
# plt.xscale("log")
plt.grid(True, axis='y')
plt.title('Train Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('data/loss.png')
