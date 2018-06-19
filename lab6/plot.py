import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

fig = plt.figure(figsize = (8, 5))
ax1 = plt.subplot()
ax2 = ax1.twinx()

filenames = [
    'result.csv',
]

lines = []

for filename in filenames:
    tmp = np.loadtxt(filename, delimiter=',', dtype=str)
    tmp[tmp == ''] = -1
    labels = tmp[0,:]
    loss_idx = np.where(labels == 'loss')[0].item()
    reward_idx = np.where(labels == 'reward')[0].item()
    data = tmp[1:,:].astype(np.float)
    c = {
        'loss': 'C0',
        'reward': 'C1'
    }

    x, y = data[1:,0], data[1:,loss_idx]
    x, y = x[y != -1], y[y != -1]
    xvals = np.linspace(0, max(x), 100)
    yinterp = np.interp(xvals, x, y)
    fx = interpolate.interp1d(xvals, yinterp, kind='cubic')
    l = ax1.plot(x, fx(x), linestyle='-', color=c['loss'], linewidth=1, label='loss')
    lines += l

    x, y = data[1:,0], data[1:,reward_idx]
    x, y = x[y != -1], y[y != -1]
    l = ax2.plot(x, y, linestyle='-', color=c['reward'], linewidth=1, label='reward')
    lines += l
    
labs = [l.get_label() for l in lines]
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax1.legend(lines, labs, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4)
plt.xlabel('Iteration')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Reward')
plt.savefig('loss.png')
