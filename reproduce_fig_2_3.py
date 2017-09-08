#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  figuers_2_3_complexity-and-pluralism.py
#  
#  Copyright 2017 Claudius Gräbner <claudius@claudius-graebner.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


import itertools
import numpy as np
import matplotlib.pyplot as plt

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

# Figure 2: illustration of the transition function

def prob_func(shares):
    if sum(shares) != 1.0:
        if 1.0 - sum(shares) < 0.000001:
            shares[0] = 1-sum(shares[1:])
        else:
            raise AssertionError("Shares do not sum to unity but to {}".format(sum(shares)))
    probs = []
    for x_i in shares:
        p_x_i = (12*x_i**2 - 5*x_i**3) / sum([12*x_j**2 - 5*x_j**3 for x_j in shares])
        probs.append(p_x_i)
    if sum(probs) != 1.0:
        if 1.0 - sum(probs) < 0.000001:
            probs[0] = 1-sum(probs[1:])
        else:
            raise AssertionError("Probabilities do not sum to unity but to {}".format(sum(probs)))
    return probs    
    
shares_x_i_print = np.linspace(0, 1, num=11)
shares_x_i = np.linspace(0, 1, num=110)

probs = []
probs_x_i_2 = [prob_func([x_i, 1-x_i])[0] for x_i in shares_x_i]
probs.append(probs_x_i_2)
probs_x_i_3 = [prob_func([x_i] + 2*[(1-x_i)/2])[0] for x_i in shares_x_i]
probs.append(probs_x_i_3)
probs_x_i_4 = [prob_func([x_i] + 3*[(1-x_i)/3])[0] for x_i in shares_x_i]
probs.append(probs_x_i_4)
probs_x_i_5 = [prob_func([x_i] + 4*[(1-x_i)/4])[0] for x_i in shares_x_i]
probs.append(probs_x_i_5)

locs = list(itertools.product(range(2), range(2)))
nbs_big = ["Two", "Three", "Four", "Five"]
nbs_small = ["two", "three", "four", "five"]

fixed_point = [int(x * 100) for x in [1/2, 1/3, 1/4, 1/5]]

plt.clf()
fig, ax = plt.subplots(2,2, figsize=(12, 9))
for i in range(len(locs)):
    probs[i] = np.asarray(probs[i])
    z = ax[locs[i]].plot(shares_x_i, probs[i], 
                         color="#0080FF", alpha=0.5)
    x = ax[locs[i]].plot(shares_x_i[1:fixed_point[i]+4], 
                         probs[i][1:fixed_point[i]+4], 
                         color="#0080FF", linestyle="None")[0]
    y = ax[locs[i]].plot(shares_x_i[fixed_point[i]: ][::2], 
                         probs[i][fixed_point[i]:][::2], 
                         color="#0080FF", linestyle="None")[0]
    
    ax[locs[i]].plot(fixed_point[i]/100, fixed_point[i]/100, 
                     marker='o', markersize=8, color="#0080FF")

    if i <= 1:
        for j in shares_x_i[2+1:fixed_point[i]][::2]:
            add_arrow(x, j,direction="left", size=20)
        for j in shares_x_i[8+fixed_point[i]:-2][::2]:
            add_arrow(y, j, direction="right", size=20)
    else:
        for j in shares_x_i[2+1:fixed_point[i]][::2]:
            add_arrow(x, j,direction="left", size=20)
        for j in shares_x_i[4+fixed_point[i]:-2][::2]:
            add_arrow(y, j, direction="right", size=20) 
    ax[locs[i]].set_title(nbs_big[i] + " research programs", fontsize=10)
    ax[locs[i]].plot(shares_x_i, shares_x_i, linestyle="--", color="silver")
    ax[locs[i]].set_xlabel("x_i")
    ax[locs[i]].set_ylabel("p(x_i)")
    ax[locs[i]].set_xticks(shares_x_i_print)
    ax[locs[i]].spines["top"].set_visible(False)  
    ax[locs[i]].spines["right"].set_visible(False)  
    ax[locs[i]].get_xaxis().tick_bottom()  
    ax[locs[i]].get_yaxis().tick_left() 

plt.tight_layout()
plt.savefig('output/fig02_transition-function.pdf', bbox_inches='tight')

# Figure 3: examples for the Polya process

def simulate_polya(adopters, timesteps):
    nb_paradigms = len(adopters)
    shares = []
    shares.append([x/sum(adopters) for x in adopters])
    for t in range(timesteps):
        new_adopted = np.random.choice(range(nb_paradigms), p=prob_func([x / sum(adopters) for x in adopters]))
        adopters[new_adopted] += 1
        shares.append([x/sum(adopters) for x in adopters])
    return(shares)
    
time_steps = 500
runs = 10
results = []
for i in range(runs):
    results.append(simulate_polya([1,1,1], time_steps))

results_total = []
results_2 = [simulate_polya([1,1], time_steps) for i in range(runs)]
results_total.append(results_2)
results_3 = [simulate_polya([1,1,1], time_steps) for i in range(runs)]
results_total.append(results_3)
results_4 = [simulate_polya([1,1,1,1], time_steps) for i in range(runs)]
results_total.append(results_4)
results_5 = [simulate_polya([1,1,1,1,1], time_steps) for i in range(runs)]
results_total.append(results_5)

plt.clf()
fig, ax = plt.subplots(2,2, figsize=(12, 9))
for i in range(len(locs)):
    for r in range(runs):
        ax[locs[i]].plot(range(time_steps + 1), [results_total[i][r][j][0] for j in range(len(results[i]))])
    ax[locs[i]].set_title("Shares for the first of " + nbs_small[i] + " research programs", fontsize=10)
    ax[locs[i]].set_xlabel("t")
    ax[locs[i]].set_ylabel("x_0")
    #ax[locs[i]].set_xticks(shares_x_i_print)
    ax[locs[i]].spines["top"].set_visible(False)  
    ax[locs[i]].spines["right"].set_visible(False)  
    ax[locs[i]].get_xaxis().tick_bottom()  
    ax[locs[i]].get_yaxis().tick_left()  

plt.tight_layout()
plt.savefig('output/fig03_polya-process.pdf', bbox_inches='tight')
