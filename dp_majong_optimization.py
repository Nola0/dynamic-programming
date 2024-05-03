# -*- coding: utf-8 -*-
"""
Created on Fri May  3 23:49:56 2024

@author: User
"""

import numpy as np
import pandas as pd
from itertools import combinations

## Basic DP Model

# Iterate all states

all_tiles = [1,2,3,4,5,6,7,8,9]*4
print('All tiles: ',all_tiles)

all_states = list(combinations(all_tiles,8))
all_states = [sorted(x) for x in all_states]
all_states = [''.join(map(str,x)) for x in all_states]
all_states = list(set(all_states))
all_states = [sorted([int(x) for x in state]) for state in all_states]
S = len(all_states)
n = 10
print('There are ',S, 'possible states in the state space')
print('Maximum', n, 'discards are allowed')

# Iterate winning states

sets = [3*[i] for i in range(1,10)]
print('All sets: ',sets)
straights = [[n]+[n+1]+[n+2] for n in range(1,8)]
print('All straights: ',straights)
pairs = [2*[i] for i in range(1,10)]
print('All pairs: ',pairs)
complete = sets+straights
print('All melds: ',complete)

# finding all combinaitons of completes
completes = list(combinations(complete, 2))
completes = [list(x) for x in completes]
completes = [x[0]+x[1] for x in completes]
#completes = [sorted(x) for x in completes]
winning = []

# combining pairs with completes to create winning hand
for x in completes:
    for pair in pairs:
        hand = sorted(x + pair)
        counts = [[i,hand.count(i)] for i in set(hand)]
        correct = True
        for item in counts:
            if item[1] > 4:
                correct = False
        
        if correct and hand not in winning:
            winning.append(hand)
            #print(winning)

# winning = set(winning)
print('There are', len(winning), 'winning hand combinations for one color')      

# Initialize value-to-go and decision matrices
# initialized the value to go and decision matrices
# columns correspond to the states, rows correspond to stages
J_matrix = pd.DataFrame(0, columns = range(S), index = reversed(range(0,n+1))) 
print(J_matrix,'\n')

u_matrix = pd.DataFrame('nil', columns = range(S), index = reversed(range(1,n+1))) 
print(u_matrix)

# insert Jk(xk)=1 for winning states
for x in winning:
    for k in J_matrix.index:
        J_matrix.loc[k ,all_states.index(x)] = 1
        
# Backward Induction

# bellman equation Jk(Xk) = max E(Jk-1(Xk\{uk}U wk)) 
for i in range(1,n + 1): # stages 1-10
    print('stage=',i)
    for j in range(S): # iterate all possible states
        if J_matrix.loc[i,j] == 1:
            continue
        else:
            xk = all_states[j] # xk contains 8 tiles, the state is after a random tile is drawn in round k
            xk_str = ''.join(map(str,xk))

            best_J = 0 # initialize value function at this xi,j to be 0
            best_u = 'nil' # initialize optimal decision at xi,j to be none

            for u in set(xk_str): # trying discarding each tile
                Jk = 0
                xk_7 = xk_str.replace(u,'',1)
                xk_7 = [int(item) for item in list(xk_7)] # convert string to list of int
                pool = all_tiles.copy()

                for item in xk_7:
                    pool.remove(item)  # prevent double counting

                counts = [[tile,pool.count(tile)] for tile in set(pool)]
                for item in counts:
                    next_x = sorted(xk_7 + [item[0]])
                    Jk += (item[1]/29)*J_matrix.loc[i-1,all_states.index(next_x)]

                if Jk > best_J:
                    best_J = Jk
                    best_u = int(u)

            J_matrix.loc[i,j] = best_J
            u_matrix.loc[i,j] = best_u

# print('Matrix of optimal value-to-go at each xi,j:')
# print(J_matrix,'\n')
# print('Matrix of optimal decision ui at each xi,j')
# print(u_matrix)

J_matrix.to_csv(r'C:\Users\Nola\OneDrive\ISM\J_matrix.csv')
print('Matrix of optimal value-to-go at each xi,j:')

u_matrix.to_csv(r'C:\Users\Nola\OneDrive\ISM\u_matrix.csv')
print('Matrix of optimal decision at each xi,j:')

# **Using the Optimzation Results**
'''
For example, I have a hand as given below and maximum 3 discards allowed. To find out the maximum probability of winning and the optimal decision to make at this point, we look up the J_matrix and u_matrix:
'''

x = [2,2,3,5,7,8,8,9]
k = 3

print('Given the hand ',x, 'and ',k, 'maximum discards allowed')
print('Maximum probability of winning is ', J_matrix.loc[k,all_states.index(x)])
print('The optimal tile to discard is ', u_matrix.loc[k,all_states.index(x)])

# Analyze Optimization Results**
import matplotlib.pyplot as plt
import seaborn as sns

# expected maximum probability of winning given optimal move
prob = []
for k in reversed(J_matrix.index):
    p = J_matrix.loc[k,:].mean()
    prob.append(p)

prob = [round(p, 5) for p in prob]
probs = pd.DataFrame(prob, index=reversed(J_matrix.index))
probs.columns = ['Expected maximum probability of winning at stage k']

print(probs)

# Visualize expected maximum probability of winning given optimal move
plt.figure(figsize=(8,6))
plt.title('Expected Probability of Winning in k Steps', size=16)

plt.scatter(list(reversed(J_matrix.index)), prob)
plt.ylim(0,1)

plt.xticks(size=16)
plt.yticks(size=16)

plt.xlabel('Number of steps', size=16)
plt.ylabel('Expected Probability', size=16)
# probability increases at a decreasing rate

# the number of states of sure win does not change no matter which stage it is in
J_matrix[J_matrix == 1].count(axis=1)

'''
**Visualize Distribution of Value to go**

Scale of vertical Axis is standardized; scale of horizontal axis is not standardized to ensure visibility
'''

f0 = sns.histplot(data=J_matrix.loc[0,:], kde=True, bins=10)
f0.set(title='Probability of winning in 0 step')
f0.set_xlabel('Probability', fontsize=12)
f0.set_ylabel('State Count', fontsize=12)
f0.set_ylim(0,S)
f0.set_xlim(min(J_matrix.loc[0,:]),1)
f0.figure.savefig("f0.png")
plt.show()

f1 = sns.histplot(data=J_matrix.loc[1,:], kde=True, bins=10)
f1.set(title='Probability of winning in 1 step')
f1.set_xlabel('Probability', fontsize=12)
f1.set_ylabel('State Count', fontsize=12)
f1.set_ylim(0,S)
f1.set_xlim(min(J_matrix.loc[1,:]),1)
f1.figure.savefig("f1.png")
plt.show()

f2 = sns.histplot(data=J_matrix.loc[2,:], kde=True, bins=10)
f2.set(title='Probability of winning in 2 steps')
f2.set_xlabel('Probability', fontsize=12)
f2.set_ylabel('State Count', fontsize=12)
f2.set_ylim(0,S)
f2.set_xlim(min(J_matrix.loc[2,:]),1)
f2.figure.savefig("f2.png")
plt.show()

f3 = sns.histplot(data=J_matrix.loc[3,:], kde=True, bins=10)
f3.set(title='Probability of winning in 3 steps')
f3.set_xlabel('Probability', fontsize=12)
f3.set_ylabel('State Count', fontsize=12)
f3.set_ylim(0,S)
f3.set_xlim(min(J_matrix.loc[3,:]),1)
f3.figure.savefig("f3.png",)
plt.show()

f4 = sns.histplot(data=J_matrix.loc[4,:], kde=True, bins=10)
f4.set(title='Probability of winning in 4 steps')
f4.set_xlabel('Probability', fontsize=12)
f4.set_ylabel('State Count', fontsize=12)
f4.set_ylim(0,S)
f4.set_xlim(min(J_matrix.loc[4,:]),1)
f4.figure.savefig("f4.png",)
plt.show()

f5 = sns.histplot(data=J_matrix.loc[5,:], kde=True, bins=10)
f5.set(title='Probability of winning in 5 steps')
f5.set_xlabel('Probability', fontsize=12)
f5.set_ylabel('State Count', fontsize=12)
f5.set_ylim(0,S)
f5.set_xlim(min(J_matrix.loc[5,:]),1)
f5.figure.savefig("f5.png",)
plt.show()

f6 = sns.histplot(data=J_matrix.loc[6,:], kde=True, bins=10)
f6.set(title='Probability of winning in 6 steps')
f6.set_xlabel('Probability', fontsize=12)
f6.set_ylabel('State Count', fontsize=12)
f6.set_ylim(0,S)
f6.set_xlim(min(J_matrix.loc[6,:]),1)
f6.figure.savefig("f6.png",)
plt.show()

f7 = sns.histplot(data=J_matrix.loc[7,:], kde=True, bins=10)
f7.set(title='Probability of winning in 7 steps')
f7.set_xlabel('Probability', fontsize=12)
f7.set_ylabel('State Count', fontsize=12)
f7.set_ylim(0,S)
f7.set_xlim(min(J_matrix.loc[7,:]),1)
f7.figure.savefig("f7.png",)
plt.show()

f8 = sns.histplot(data=J_matrix.loc[8,:], kde=True, bins=10)
f8.set(title='Probability of winning in 8 steps')
f8.set_xlabel('Probability', fontsize=12)
f8.set_ylabel('State Count', fontsize=12)
f8.set_ylim(0,S)
f8.set_xlim(min(J_matrix.loc[8,:]),1)
f8.figure.savefig("f8.png",)
plt.show()

f9 = sns.histplot(data=J_matrix.loc[9,:], kde=True, bins=10)
f9.set(title='Probability of winning in 9 steps')
f9.set_xlabel('Probability', fontsize=12)
f9.set_ylabel('State Count', fontsize=12)
f9.set_ylim(0,S)
f9.set_xlim(min(J_matrix.loc[9,:]),1)
f9.figure.savefig("f9.png",)
plt.show()

f10 = sns.histplot(data=J_matrix.loc[10,:], kde=True, bins=10)
f10.set(title='Probability of winning in 10 steps')
f10.set_xlabel('Probability', fontsize=12)
f10.set_ylabel('State Count', fontsize=12)
f10.set_ylim(0,S)
f10.set_xlim(min(J_matrix.loc[10,:]),1)
f10.figure.savefig("f10.png",)
plt.show()

## DP Model wtith Collapsed States
# **Convert original states to collapsed states**

# define a function to collapse original state into collapsed state
def collapse(state):
    counts = [[tile,state.count(tile)] for tile in sorted(set(state))]
        # print(counts)
    collapsed = []
    for item in counts:
        collapsed.append(item[1])

    for i in range(len(counts)-1): 
        d = counts[i+1][0] - counts[i][0]

        collapsed[i] = str(collapsed[i])+ str(d)

    collapsed[-1] = str(collapsed[-1])
    collapsed.insert(0, str(counts[0][0]-1))
    collapsed.append(str(9-counts[-1][0]))

    collapsed = ''.join(collapsed)

    return collapsed


c_states = []
for state in all_states:
    c_state = collapse(state)
    if (c_state not in c_states) and (c_state[::-1] not in c_states):
        c_states.append(c_state)
s = len(c_states)
print('There are', s, 'collapsed states, compared to', S, 'original states')
# a bit mroe than half of the number of original states, not excatly half cos some states are excatly in the middle eg. 23455678
# symmetric states become collapsed into 1 -> is there loss of information?

c_winning = []
for state in winning:
    c_state = collapse(state)
    if (c_state not in c_winning) and (c_state[::-1] not in c_winning):
        c_winning.append(c_state)

print('There are', len(c_winning), 'collapsed winning states, compared to', len(winning), 'original winning states')
# a bit mroe than half of the number of original states, not excatly half cos some states are excatly in the middle

# **Initialize Value to go and decision matrices**

# initialized the value to go and decision matrices
# columns correspond to the states, rows correspond to stages
J_matrix_c = pd.DataFrame(0, columns = range(s), index = reversed(range(0,n+1))) 
print(J_matrix_c,'\n')

u_matrix_c = pd.DataFrame('nil', columns = range(s), index = reversed(range(1,n+1))) 
print(u_matrix_c)

# insert Jk(xk) for winning states
for x in c_winning:
    for k in J_matrix_c.index:
        try:
            J_matrix_c.loc[k ,c_states.index(x)] = 1
        except:
            J_matrix_c.loc[k ,c_states.index(x[::-1])] = 1

# if x1 = reverse(x2), x1 = x2, the 2 collapsed states are used interchangeably


# **Backward Induction**
# bellman equation Jk(Xk) = max E(Jk-1(Xk\{uk}U wk)) 

for i in range(1,n + 1): # stages 1-10
    for j in range(s): # iterate all possible states
        if J_matrix_c.loc[i,j] == 1:
            continue
        else:
            xk = c_states[j]
            # print('xk=',xk)
            nunique = int((len(xk)-2)/2 + 0.5) # number of unique tiles
            # print('number of unique tiles=', nunique)

            best_J = 0 # initialize value function at this xi,j to be 0
            best_u = 'nil' # initialize optimal decision at xi,j to be none

            for u in range(1,nunique+1): # trying discarding each tile
                # print('uth tile is', u)
                Jk = 0
                xk = list(xk)
                xk_7 = xk.copy()
                if int(xk[2*u-1])==1: # if the one to be discarded is the only one of that pattern
                    xk_7[2*u-1] = str(int(xk[2*u-1])-1)
                    xk_7 = ''.join(xk_7)
                    # print('xk_7 is',xk_7)
                    loc = xk_7.index('0')
                    if loc == 0: # first 0 is distance between the smallest tile and 1
                        loc = xk_7.index('0',1) 
                    # print('loc of 0', loc)
                    neighbor = xk_7[loc-1] + '0' + xk_7[loc+1]
                    # print('neighbor of 0', neighbor)
                    d = str(int(xk_7[loc-1]) + int(xk_7[loc+1]))
                    # print('sum of distance',d)
                    xk_7 = xk_7.replace(neighbor,d,1)
                    # print('xk_7 after replace', xk_7)
                else:
                    xk_7[2*u-1] = str(int(xk[2*u-1])-1)

                # print('xk_7 is', xk_7) # xk_7 is a string
                head = int(xk_7[0])
                # print('head=',head)
                tail = int(xk_7[-1])
                # print('tail=',tail)
                nunique1 = int((len(xk_7)-2)/2 + 0.5)
                # print('number of unique existing tiles', nunique1)

                for w in range(1,nunique1+1): # existing tiles are drawn
                    # print('wth existing tile',w)
                    next_x = list(xk_7)
                    # print(next_x)
                    if int(xk_7[2*w-1]) <= 3:
                        next_x[2*w-1] = str(int(xk_7[2*w-1])+1)
                        next_x = ''.join(next_x)
                        # print(next_x)

                        try:
                            Jk += ((4-int(xk_7[2*w-1]))/29)*J_matrix_c.loc[i-1,c_states.index(next_x)]
                        except:
                            Jk += ((4-int(xk_7[2*w-1]))/29)*J_matrix_c.loc[i-1,c_states.index(next_x[::-1])]

                dunique = nunique1 -1
                # print('number of gaps', dunique)
                for d in range(1, dunique+1): # draw tiles in between existing tiles
                    # print(d,'th gap')
                    gap = int(xk_7[2*d])
                    # print('gap=',gap)
                    if gap > 1:
                        for g in range(1,gap):
                            next_x = list(xk_7)
                            # print('before replacement',next_x)
                            next_x[2*d] = str(g) + str(1) + str(gap-g) 
                            # print('new components',next_x[2*d])
                            next_x = ''.join(next_x)
                            try:
                                Jk += (4/29)*J_matrix_c.loc[i-1,c_states.index(next_x)]
                            except:
                                Jk += (4/29)*J_matrix_c.loc[i-1,c_states.index(next_x[::-1])]

                if head > 0: # tiles smaller than all existing tiles are drawn
    #                 print('head=',head)
    #                 print('before head removed', xk_7)
    #                 print('head removed', next_x)
                    for h in range(head):
                        next_x = list(xk_7)[1:]
                        next_x = ''.join(next_x)
                        # print('h',h)
                        add = str(h) + str(1) + str(head-h)
                        # print('add', add)
                        next_x = add + next_x
                        # print('after adding', next_x)
                        try:
                            Jk += (4/29)*J_matrix_c.loc[i-1,c_states.index(next_x)]
                        except:
                            Jk += (4/29)*J_matrix_c.loc[i-1,c_states.index(next_x[::-1])]

                if tail > 0: # tiles larger than existing tiles are drawn
                    for t in range(tail):
                        next_x = list(xk_7)[:-1]
                        next_x = ''.join(next_x)
                        add = str(tail-t) + str(1) + str(t)
                        next_x = next_x + add
                        try:
                            Jk += (4/29)*J_matrix_c.loc[i-1,c_states.index(next_x)]
                        except:
                            Jk += (4/29)*J_matrix_c.loc[i-1,c_states.index(next_x[::-1])]


                if Jk > best_J:
                    best_J = Jk
                    best_u = int(u)  # discardthe uth unique tile 

            J_matrix_c.loc[i,j] = best_J
            u_matrix_c.loc[i,j] = best_u

# print('Matrix of optimal value-to-go at each xi,j:')
# print(J_matrix,'\n')
# print('Matrix of optimal decision ui at each xi,j')
# print(u_matrix)

J_matrix_c.to_csv(r'C:\Users\Nola\OneDrive\ISM\J_matrix_c.csv')
print('Matrix of optimal value-to-go at each collapsed state xi,j:')


u_matrix.to_csv(r'C:\Users\Nola\OneDrive\ISM\u_matrix_c.csv')
print('Matrix of optimal decision at each collapsed state xi,j:')

# u means discard the uth unique tile in hand

# **Using Optimization Results of Model with Collapsed States**
# same example as above
x = [2,2,3,5,7,8,8,9]
k = 3

# convert the hand to its collapsed state
x_c = collapse(x)
print('Collapsed state x_c: ',x_c)

# the collapsed state may be either in normal order or reversed order
normal = True
try:
    max_p = J_matrix_c.loc[k,c_states.index(x_c)]
    print('Collapsed state in normal order')
except:
    max_p = J_matrix_c.loc[k,c_states.index(x_c[::-1])]
    print('Collapsed state in reverse order')
    normal = False
    
print('Given the hand ',x, 'and ',k, 'maximum discards allowed')
print('Maximum probability of winning is ', max_p)

if normal:
    decision = u_matrix_c.loc[k,c_states.index(x_c)]
    print('The', decision, 'th unique tile should be discarded')
else:
    decision = u_matrix_c.loc[k,c_states.index(x_c[::-1])]
    print('The', decision, 'th unique tile counted from the back should be discarded')
    
c_states.index(x_c)

# **Analyze and Compare 2 Models**

# probability of winning without any move
prob_c = []
for k in reversed(J_matrix.index):
    p = J_matrix_c.loc[k,:].mean()
    prob_c.append(p)
    
prob_c = [round(p, 5) for p in prob_c]

probs2 = pd.concat((probs, pd.Series(prob_c)), axis=1)
probs2.columns = ['Unollasped','Collapsed']
# mostly the same up to 3dp; deviations exist after 3dp

# plot probability density distribution from model with collapsed states
for k in reversed(J_matrix_c.index):
    print('Probability of winning in ',k,' steps')
    
    f = sns.histplot(data=J_matrix_c.loc[k,:], kde=True, bins=10)
    f.set_xlabel('Probability', fontsize=12)
    f.set_ylabel('State Count', fontsize=12)
    f.set_ylim(0,s)
    plt.show()
