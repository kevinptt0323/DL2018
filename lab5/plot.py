import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


fig = plt.figure(figsize = (8, 5))
ax = plt.subplot(111)

filenames = [
    'td.csv',
    'td-after.csv',
]

for filename in filenames:
    tmp = np.loadtxt(filename, delimiter=',', dtype=str)
    labels = tmp[0,:]
    data = tmp[1:,:].astype(np.float)
    c = {
        'lossD': 'C0',
        'lossG': 'C1'
    }
    # max_x = data[1:,0].max()

    for i in range(1, data.shape[1]):
        x, y = data[1:,0], data[1:,i]
        xvals = np.linspace(0, max(data[1:,0]), 50)
        yinterp = np.interp(xvals, x, y)
        fx = interpolate.interp1d(xvals, yinterp, kind='cubic')
        ax.plot(x, y, linestyle='-', linewidth=1, label=filename[:-4])
        # ax.plot(x, fx(x), linestyle='-', linewidth=1, label=labels[i] + ' (interp)')
    
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4)
# plt.xscale("log")
plt.grid(True, axis='y')
plt.xlabel('Iteration')
plt.ylabel('Mean Score')
plt.savefig('history.png')
