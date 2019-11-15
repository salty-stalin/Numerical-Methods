import matplotlib.pyplot as plt
import numpy as np
import time

Interval = [0,0.1]
A1 = np.array([[-1000, 0], [1000, -1]])
Y0=np.array([[1],[0]])

u=0.5*(1-1/3**0.5)
v=0.5*(3**0.5-1)
g=3/(2(3+3**0.5))
l=(3(1+3**0.5))/(2(3+3**0.5))

def bvector(x):
    return 0

    
def rk3(A, bvector, y0, interval, N):
    print(interval[1])
    print(interval[0])
    h = (interval[1]-interval[0])/N
    print(h)
    k=0
    x=interval[0]
     

    ysol = np.array([[],[]])
    ysol = np.append(ysol, y0,axis=1)
    xsol =  np.array([])
    xsol = np.append(xsol,x)
    y =np.empty(shape=(2,2))
    y[0,0]= y0[0,0]
    y[1,0]= y0[1,0]
    print(y)
    y1 =np.empty(shape=(2,1))
    y2 =np.empty(shape=(2,1))
    for i in range(N):
        b0=0
        b=0    
        #b0 = bvector(x)
        #b = bvector(x+h)
 #       print(y)
  #      print(x)  
   #     print(h*(np.dot(A,x)+b0))
        y1 = (y[:,[k]]) +h*(np.dot(A,y[:,[k]])+b0) 
    #    print(y1)
        y2=0.75*(y[:,[k]]) + 0.25*y1 + 0.25*h*(np.dot(A,y1) + b )
     #   print(y2)        
        y[:,[k+1]]=((1/3)*(y[:,[k]])+(2/3)*y2+(2/3)*h*(np.dot(A,y2) + b)) 
        print(y)
        x=x+h
        print(x)
        ysol = np.append(ysol, y[:,[k+1]],axis=1)
        xsol = np.append(xsol,x)
        y[:,[k]]=y[:,[k+1]]
    return [xsol, ysol]

def dirk3(A, bvector, y0, interval, N):
    h = (interval[1]-interval[0])/N
    k=0
    x=interval[0]
    ysol = np.array([[],[]])
    ysol = np.append(ysol, y0,axis=1)
    xsol =  np.array([])
    xsol = np.append(xsol,x)
    y =np.empty(shape=(2,2))
    y[0,0]= y0[0,0]
    y[1,0]= y0[1,0]
    print(y)
    y1 =np.empty(shape=(2,1))
    y2 =np.empty(shape=(2,1))
    for i in range(N):
        b0=0
        b=0    
        #b0 = bvector(x+h*u)
        #b = bvector(x+h*v+2*h*u)
        a=(np.identity(2)-h*u*A)
        b=(y[:,[k]]) +h*u*(+b0)
        b1= y1+h*v*(np.dot(A,y1)+b0) +h*u*b
        y1 = np.linalg.solve(a, b)
        y2 = np.linalg.solve(a, b1)
       
        y[:,[k+1]]=(1-l)*(y[:,[k]])+l*y2+h*g*(np.dot(A,y2)+b)
        print(y)
        x=x+h
        print(x)
        ysol = np.append(ysol, y[:,[k+1]],axis=1)
        xsol = np.append(xsol,x)
        y[:,[k]]=y[:,[k+1]]
    return [xsol, ysol] 

    
x1,y1= dirk3(A1,0,Y0,Interval,100)
x,y =rk3(A1,0,Y0,Interval,100)
print(x)
print(y)





y1=np.log(y[0])
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
ax1.plot(x,y1)
ax1.set_title('y1 against x')
ax2.plot(x,y[1])
ax2.set_title('y1 against x')
plt.show()
