import numpy as np
import matplotlib.pyplot as plt

#単位球内に一様に分布する点の座標(x,y,z)を求める
xl=[]
yl=[]
zl=[]

for i in range(5000):
    theta = np.random.uniform(-1, 1)
    phi = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, 1)
    x=r**(1/3)*(1-theta**2)*np.cos(phi)
    y=r**(1/3)*(1-theta**2)*np.sin(phi)
    z=r**(1/3)*theta
    xl.append(x)
    yl.append(y)
    zl.append(z)
    
    
#3次元の散布図を描く
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.scatter(xl, yl, zl, s=1)
plt.show()

#球内に分布する点の距離のリストを取得する
distance=[]
for i in range(len(xl)):
    dis = np.sqrt(xl[i]**2 + yl[i]**2 + zl[i]**2)
    distance.append(dis) 

dist = 1000
distlist = list(map(lambda x: x *dist , distance))
