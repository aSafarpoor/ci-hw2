#*****************************************
import numpy as np
import matplotlib.pyplot as plt
from HodaDatasetReader.HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
plt.rcParams['figure.figsize'] = (7,9) # Make the figures a bit bigger
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, to_categorical
import keras

#load training data
nb_classes = 10
x_train, y_train = read_hoda_dataset(dataset_path='HodaDatasetReader/DigitDB/Train 60000.cdb',
                                images_height=20,
                                images_width=20,
                                one_hot=False,
                                reshape=True)
print(y_train[3])

x_test, y_test = read_hoda_dataset(dataset_path='HodaDatasetReader/DigitDB/Test 20000.cdb',
                              images_height=20,
                              images_width=20,
                              one_hot=True,
                              reshape=False)


x_train = x_train.reshape((60000, 20*20))
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((20000, 20 * 20))
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)

def main(epoch_num,batch_num):
	network = Sequential()
	network.add(Dense(100, activation='relu', input_shape=(400,)))
	keras.layers.Dropout(.05, noise_shape=None, seed=None)
	network.add(Dense(10, activation='softmax'))

	network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	network.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_num,validation_split=0.1)



	
	f = open("natayej.txt", "a")
	# print("x is : ",x_test[0]," *y is : ",y_test[0])
	s=str(batch_num)
	score = network.evaluate(x_test, y_test)
	ss=str(score[1])
	#print('Test score :', score[0])
	#print('Test accuracy :', score[1])
	score = network.evaluate(x_train, y_train)
	sss=str(score[1])
	#print('Train score:', score[0])
	#print('Train accuracy:', score[1])
	#print("\n\n")


	s=s+" "+ss+" "+sss+" \n"
	f.write(s)
	f.close()



	# # The predict_classes function outputs the highest probability class
	# # according to the trained classifier for each input example.
	predicted_classes = network.predict_classes(x_test)



	#it's for recall,f1,...
	tp =[0 for i in range(10)]
	tn =[0 for i in range(10)]
	fp =[0 for i in range(10)]
	fn =[0 for i in range(10)]

	for i in range(20000):
		z=0
		p=predicted_classes[i]
		for j in range(10):
			if(y_test[i][j]==1):
				z=j
		if(z==p):
			tp[z]+=1
			for j in range(10):
				if(z!=j):
					tn[j]+=1
		else:
			fp[p]+=1
			for j in range(10):
				if(z==j):
					fn[j]+=1
				elif(p==j):
					fp[j]+=1
				else:
					tn[j]+=1

	predicted_classes=network.predict_classes(x_train)

	for i in range(60000):
		z=0
		p=predicted_classes[i]
		for j in range(10):
			if(y_train[i][j]==1):
				z=j
		if(z==p):
			tp[z]+=1
			for j in range(10):
				if(z!=j):
					tn[j]+=1
		else:
			fp[p]+=1
			for j in range(10):
				if(z==j):
					fn[j]+=1
				elif(p==j):
					fp[j]+=1
				else:
					tn[j]+=1

	'''
	f1=2tp/(2tp+fp+fn)
	recall=tp/(tp+fn)
	precision=tp/(tp+fp)

	part_b = [,,,tn]
	'''
	avg_f1=0
	avg_re=0
	avg_pr=0

	for i in range(10):
		f1=2*tp[i]/(2*tp[i]+fp[i]+fn[i])
		recall=tp[i]/(tp[i]+fn[i])
		precision=tp[i]/(tp[i]+fp[i])
		avg_f1+=f1
		avg_pr+=precision
		avg_re+=recall
		print("class ",i,": f1=",f1," recall=",recall," precision=",precision)

	avg_f1/=10
	avg_re/=10
	avg_pr/=10
	print("total   ",": f1=",avg_f1," recall=",avg_re," precision=",avg_pr)


# for i in range (1,51,5):
#  	main(i)
#main(1)

main(4,1)
main(6,10)
main(6,50)
main(6,100)
main(6,500)
main(6,1000)
main(6,10000)
main(6,30000)
main(6,60000)
