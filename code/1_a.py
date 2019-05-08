

from matplotlib import pyplot as plt
from HodaDatasetReader.HodaDatasetReader import read_hoda_cdb, read_hoda_dataset

train_images, train_labels = read_hoda_cdb('HodaDatasetReader/DigitDB/Train 60000.cdb')

test_images, test_labels = read_hoda_cdb('HodaDatasetReader/DigitDB/Test 20000.cdb')

remaining_images, remaining_labels = read_hoda_cdb('HodaDatasetReader/DigitDB/RemainingSamples.cdb')


fig = plt.figure(figsize=(15, 4))

fig.add_subplot(1, 4, 1)
plt.title('train_labels[' + str(0) + '] = ' + str(train_labels[0]))
plt.imshow(train_images[0], cmap='gray')

fig.add_subplot(1, 4, 2)
plt.title('test_labels[' + str(0) + '] = ' + str(  test_labels[0]))
plt.imshow(test_images[0], cmap='gray')


plt.show()
