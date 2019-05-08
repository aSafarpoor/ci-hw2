import numpy as np
import  math
import random
import matplotlib.pyplot as plt


iteration_counter=1
sigma_zero=1
width_of_map=40
t=1
r0=40
Lambda=1000
k0=1#learning rate
w_matrix = np.zeros((40,40,3))
for i in range(40):
 	for j in range(40):
 		for z in range(3):
 			w_matrix[i][j][z]=np.random.random(1)

# print (w_matrix[0])
# print (w_matrix[0][0])
# def compute_sigma(t):
# 	global width_of_map,iteration_counter,sigma_zero 
# 	Lambda=iteration_counter/width_of_map
# 	return (sigma_zero*(math.exp(-t/Lambda)))

def color_distance(x,w):
	sum=0
	for i in range(3):
		sum+=(x[i]-w[i])**2
	dis=sum**.5
	return dis

def n_distance(a1,a2,b1,b2):
	return ((a1-b1)**2+(a2-b2)**2)**2

def choose_min(x):
	global w_matrix
	min_dist=99999999999
	index_i=-1
	index_j=-1
	for i in range(40):
		for j in range(40):
			dis=color_distance(x,w_matrix[i][j])
			if(dis<min_dist):
				min_dist=dis
				index_j=j
				index_i=i
	rate = np.random.random((1))[0]*100
	if(rate<2):
		index_j=int(np.random.random((1))[0]*40)
		index_i=int(np.random.random((1))[0]*40)
	return [index_i,index_j]

def compute_r(t):#r(t)  means the  width  of  the  lattice   
	global r0,Lambda
	r=r0*math.exp(-t/Lambda)
	if(r<.001):
		r=.001
	return r
def compute_k(t):
	global k0,Lambda

	return k0*math.exp(-t/Lambda)

def compute_tetta(t,neorun_num,winning_num):
	global w_matrix
	a1=neorun_num[0]
	a2=neorun_num[1]
	b1=winning_num[0]
	b2=winning_num[1]
	d=n_distance(a1,a2,b1,b2)
	r=compute_r(t)
	return math.exp(-d**2/(2*r**2))

def update_weight(t,w,inp,tetta,k):
	return w+tetta*k*(inp-w)
	#w_t+1=w_t + tetta(t)k(t)(i(t)-w(t))

def learn_for_each_data(x):
	global t
	#each iteration with x is input_data:
	# x=[1,2,.1,.1,.1]
	# for iter in range(10):
	if(len(x)==5):x=x[2:0]
	index_of_winner=choose_min(x)
	#print(index_of_winner)
	#print(w_matrix[index_of_winner[0]][index_of_winner[1]])
	r=compute_r(t)
	k=compute_k(t)
	tetta=0
	for i in range(40):
		for j in range(40):
			tetta=compute_tetta(t,[i,j],index_of_winner)
			for z in range(3):
				w=w_matrix[i][j][z]
				w_matrix[i][j][z]=update_weight(t,w,x[z],tetta,k)
		

		#if(iter%100==99):
	#print (r,k,tetta)
	#print(w_matrix[index_of_winner[0]][index_of_winner[1]])







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
# ax.scatter(x, y, s=60, marker='s',facecolors=rgb)
# plt.show()

for z in range(40):
	for zz in range(40):
		ax.scatter(z,zz,s=60,marker='s',facecolors=w_matrix[z][zz])

number_of_iteration=50

for i in range(1,1+number_of_iteration):
	print (i)
	for j in range(1600):
		learn_for_each_data(rgb[j])
		
	t+=1

	if(i==10):
		for z in range(40):
			for zz in range(40):
				ax.scatter(z,zz+50,s=60,marker='s',facecolors=w_matrix[z][zz])

	if(i==20):
		for z in range(40):
			for zz in range(40):
				ax.scatter(z,zz+100,s=60,marker='s',facecolors=w_matrix[z][zz])

	if(i==30):	
		for z in range(40):
			for zz in range(40):
				ax.scatter(z+50,zz,s=60,marker='s',facecolors=w_matrix[z][zz])
	if(i==40):	
		for z in range(40):
			for zz in range(40):
				ax.scatter(z+50,zz+50,s=60,marker='s',facecolors=w_matrix[z][zz])
	if(i==50):	
		for z in range(40):
			for zz in range(40):
				ax.scatter(z+50,zz+100,s=60,marker='s',facecolors=w_matrix[z][zz])

plt.show()





