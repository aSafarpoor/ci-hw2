
import matplotlib.pyplot as plt
import numpy as np

#x, y = np.random.random((2, 10))
x=[]
y=[]

for i in range(40):
	for  j in range (40):
		x.append(i)
		y.append(j)

rgb = np.random.random((1600, 3))
# for i in  range(len(rgb)):
# 	rgb[i][2]=0
# 	rgb[i][1]=0

#print(rgb[0],"   ",rgb[0][2])

fig, ax = plt.subplots()
ax.scatter(x, y, s=60, marker='s',facecolors=rgb)
plt.show()

