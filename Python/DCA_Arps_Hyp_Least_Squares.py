import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math

data = pd.read_table('data_test.txt')

t = data.t
q = data.q

plt.semilogy(t,q,'ro')

# x is array of parameters to find: qi, Di, b
def fun(x, t, q):
    return  (x[0] / ((1.0 + x[2] * x[1] * t)**(1.0/x[2]))) - q



# Initial guess
x0 = np.array([np.max(q), 0.5, 2.0])

def Q(t,qi, Di, b):
    return (qi / ((1.0 + b * Di * t)**(1.0/b)))
    
    
# Least squares regression
res_lsq = least_squares(fun, x0, args=(t, q), bounds=([0,0,0],[np.max(q),1,2]))
#res_lsq = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t, q), bounds=([0,0,0],[np.max(q),1,3]))

# Assign parameters per the regression
qi = res_lsq.x[0]
Di = res_lsq.x[1]
b  = res_lsq.x[2]

print 'Least Squares'
print 'qi = ' + str(qi/30)
print 'Di = ' + str(Di*12)
print 'b  = ' + str(b)

for i in range(100):
    plt.semilogy(i,Q(i, qi, Di, b), 'bo',alpha=0.3)
#plt.show()

# Least squares regression
#res_lsq = least_squares(fun, x0, args=(t, q), bounds=([0,0,0],[np.max(q),1,2]))
res_lsq = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t, q), bounds=([0,0,0],[np.max(q),1,3]))

# Assign parameters per the regression
qi = res_lsq.x[0]
Di = res_lsq.x[1]
b  = res_lsq.x[2]

print ' '
print 'Robust Least Squares'
print 'qi = ' + str(qi/30)
print 'Di = ' + str(Di*12)
print 'b  = ' + str(b)

for i in range(100):
    plt.semilogy(i,Q(i, qi, Di, b),'go', alpha = 0.3)

tt = np.zeros(100)
qq = np.zeros(100)

for i in range(100):
    tt[i] = i
    qq[i] = Q(tt[i], 7098.5*30, 1.495/12.0, 3.0)
plt.semilogy(tt,qq, 'k-')

plt.grid(True, which='both')
plt.ylim(10000,1000000)
plt.show()
