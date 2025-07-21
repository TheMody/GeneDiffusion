import matplotlib.pyplot as plt 
import numpy as np
#import csv reader
import csv

file_names = ['figures/wandbexports/CNN_loss.csv','figures/wandbexports/MLP_loss.csv','figures/wandbexports/MLP+CNN_loss.csv','figures/wandbexports/Transformer_loss.csv']
labels = ['CNN','MLP','MLP+CNN','Transformer']
for i,file_name in enumerate(file_names):
    #open the csv file
    with open(file_name) as csvfile:
        #read the csv file
        readCSV = csv.reader(csvfile, delimiter=',')
        #initialize the lists
        epochs = []
        loss = []
        #iterate over the rows
        for row in readCSV:
            #append the epoch and loss values to the lists
            epochs.append(row[0])
            loss.append(row[1])
    #remove the first element of the lists
    epochs.pop(0)
    loss.pop(0)

    #convert the strings to floats
    epochs = [float(i) for i in epochs]
    loss = [float(i) for i in loss]

    # print(epochs)
    # print(loss)

    plt.plot(epochs,loss, label=labels[i])
#plt.ylim(0, 1)
plt.xlim(0, 60000)
plt.rcParams.update({'font.size': 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Steps', fontsize=14)
plt.savefig('figures/loss.pdf')
plt.show()

# do the same for the rec error
file_names = ['figures/wandbexports/CNN_rec.csv','figures/wandbexports/MLP_rec.csv','figures/wandbexports/MLP+CNN_rec.csv','figures/wandbexports/Transformer_rec.csv']
for i,file_name in enumerate(file_names):
    #open the csv file
    with open(file_name) as csvfile:
        #read the csv file
        readCSV = csv.reader(csvfile, delimiter=',')
        #initialize the lists
        epochs = []
        loss = []
        #iterate over the rows
        for row in readCSV:
            #append the epoch and loss values to the lists
            epochs.append(row[0])
            loss.append(row[1])
    #remove the first element of the lists
    epochs.pop(0)
    loss.pop(0)

    #convert the strings to floats
    epochs = [float(i) for i in epochs]
    loss = [float(i) for i in loss]

    # print(epochs)
    # print(loss)

    plt.plot(epochs,loss, label=labels[i])

plt.ylim(0.2, 0.8)
plt.xlim(0, 60000)
plt.rcParams.update({'font.size': 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylabel('Reconstruction Error', fontsize=14)
plt.xlabel('Steps', fontsize=14)
plt.savefig('figures/recerror.pdf')
#increase font sizes

plt.show()