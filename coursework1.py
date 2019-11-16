import matplotlib.pyplot as plt
import numpy as np
import time

A2 =np.array([[-1, 0,0], [-99, -100,0] , [-10098,9900,-10000]])



Interval = [0,0.1]
A1 = np.array([[-1000, 0], [1000, -1]])
Y0=np.array([[1],[0]])
Y00=np.array([[0],[1],[0]])

u=0.5*(1-1/3**0.5)

v=0.5*(3**0.5-1)

g1=3/(2*(3+3**0.5))

l=(3*(1+3**0.5))/(2*(3+3**0.5))


def bvector1(x):
    b_vector=np.array([[np.cos(10*x)-np.exp(-x)], [199*np.cos(10*x)-10*np.sin(10*x)] , [208*np.cos(10*x)+10000*np.sin(10*x)]])
    return b_vector
def bvector(x):
    return 0
def y1_func(x):
    return np.exp(-1000*x)
def y2_func(x):
    return (1000/999)*(np.exp(-x)-np.exp(-1000*x))

def error_rk3(y2_exact,y2_rk3,N,h):
    y2=np.delete(y2_rk3,0,0)
    y2_func =np.delete(y2_exact,0,0)
    rk3_er =((y2-y2_func/y2_func))
    rk3_error = np.linalg.norm(rk3_er, ord=1, axis=None, keepdims=False)
    rk3_error_sum = h*np.sum(rk3_error)
    return rk3_error_sum
    
def error_dirk3(y2_exact,y2_dirk3,N,h): 
    y2=np.delete(y2_dirk3,0,0)
    y2_func =np.delete(y2_exact,0,0)
    dirk3_er =((y2-y2_func)/y2_func)
    dirk3_error = np.linalg.norm(dirk3_er, ord=1, axis=None, keepdims=False)
    dirk3_error_sum = h*np.sum(dirk3_error)
    return dirk3_error_sum
    
def rk3(A, bvector, y0, interval, N):

    h = (interval[1]-interval[0])/N

    k=0
    x=interval[0]
     

    ysol1 = np.array([[],[]])
    ysol1 = np.append(ysol1, y0,axis=1)
    xsol1 =  np.array([])
    xsol1 = np.append(xsol1,x)
    y =np.empty(shape=(2,2))
    y[0,0]= y0[0,0]
    y[1,0]= y0[1,0]

    y1 =np.empty(shape=(2,1))
    y2 =np.empty(shape=(2,1))
    for i in range(N):
  
        b0 = bvector(x)
        b = bvector(x+h)

        y1 = (y[:,[k]]) +h*(np.dot(A,y[:,[k]])+b0) 
        y2=0.75*(y[:,[k]]) + 0.25*y1 + 0.25*h*(np.dot(A,y1) + b )     
        y[:,[k+1]]=((1/3)*(y[:,[k]])+(2/3)*y2+(2/3)*h*(np.dot(A,y2) + b)) 
        
        x=x+h
        
        ysol1 = np.append(ysol1, y[:,[k+1]],axis=1)
        xsol1 = np.append(xsol1,x)
        y[:,[k]]=y[:,[k+1]]
    return (xsol1, ysol1,h)

def dirk3(A, bvector, y0, interval, N):
    h = (interval[1]-interval[0])/N
    k=0
    x=interval[0]
    ysol2 = np.array([[],[]])
    ysol2 = np.append(ysol2, y0,axis=1)
    xsol2 =  np.array([])
    xsol2 = np.append(xsol2,x)
    y =np.empty(shape=(2,2))
    y[0,0]= y0[0,0]
    y[1,0]= y0[1,0]

    y1 =np.empty(shape=(2,1))
    y2 =np.empty(shape=(2,1))
    for i in range(N):
  
        b0 = bvector(x+h*u)
        b = bvector(x+h*v+2*h*u)
        a=(np.identity(2)-h*u*A)
        b=(y[:,[k]]) +h*u*(+b0)
        b1= y1+h*v*(np.dot(A,y1)+b0) +h*u*b
        y1 = np.linalg.solve(a, b)
        y2 = np.linalg.solve(a, b1)
        y[:,[k+1]]=(1-l)*(y[:,[k]])+l*y2+h*g1*(np.dot(A,y2)+b)
        x=x+h
        ysol2 = np.append(ysol2, y[:,[k+1]],axis=1)
        xsol2 = np.append(xsol2,x)
        y[:,[k]]=y[:,[k+1]]
    return (xsol2, ysol2 ,h)

error_rk3_array=np.array([])
h_array=np.array([])
for k in range(1,11):
    N=40*k
    x,y, h=rk3(A1,bvector,Y0,Interval,N)
    y2_exact = y2_func(x)
    error_value_rk3 = error_rk3(y2_exact,y[1],N,h)
#    print(N)
#    print(error_value_rk3)

    error_rk3_array=np.append(error_rk3_array,error_value_rk3)

    h_array=np.append(h_array,h)
print(error_rk3_array)
print(h_array)


error_dirk3_array=np.array([])

for k in range(1,11):
    N=40*k
    x,y, h=dirk3(A1,bvector,Y0,Interval,N)
    y2_exact = y2_func(x)
    error_value_dirk3 = error_dirk3(y2_exact,y[1],N,h)
    print(h)
    print(error_value_dirk3)

    error_dirk3_array=np.append(error_dirk3_array,error_value_dirk3)

#    N_array=np.append(N_array,N)

x1,y1,h= dirk3(A1,bvector,Y0,Interval,400)
x,y ,h =rk3(A1,bvector,Y0,Interval,400)


y2_exact = y2_func(x)
y1_exact = y1_func(x1)
print(error_dirk3_array)
y1_log=np.log(y[0])
y1_log_exact=np.log(y1_exact)


f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
ax1.plot(x,y1_log ,'b')
ax1.plot(x,y1_log_exact,'r')
ax1.set_title('RK3 y1 against x')
ax2.plot(x,y[1],'b')
ax2.plot(x,y2_exact, 'r')
ax2.set_title('RK3 y2 against x')
plt.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
ax1.plot(x1,y1[0],'b')
ax1.plot(x,y1_exact,'r')
ax1.set_title('DIRK3 y1 against x')
ax2.plot(x1,y1[1],'b')
ax2.plot(x,y2_exact, 'r')
ax2.set_title('DIRK3 y2 against x')
plt.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
ax1.plot(h_array,error_rk3_array)
ax1.set_title('rk3 error against N')
ax2.plot(h_array,error_dirk3_array)
ax2.set_title('dirk3 error against N')
plt.tight_layout()
plt.show()
