import numpy as np
x = np.array([1,2])
y = x.T
A = np.array([[2,3],[6,5]])


#print(y)
print(A)
z= np.dot(A,y)
#print(z)

w, v = np.linalg.eig(A)




A_trans= A.T
print(A_trans)
w1, v1 =np.linalg.eig(A_trans)
print(w1)
print(v1)
x = v1[0]


print(x)