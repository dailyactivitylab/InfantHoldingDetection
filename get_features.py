import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import butter, lfilter
from scipy.signal import savgol_filter
from scipy.stats.stats import pearsonr   
from sklearn import svm
from scipy.signal import freqz
from scipy.fftpack import rfft, irfft, fftfreq
import scipy
import csv
from scipy.stats import entropy
from sklearn import svm
from scipy.signal import savgol_filter
from scipy.special import kl_div
from numpy import fft

all_data_folder = ['./P9/']

window_size = 7

def calculateEnergy(data, size):
    seconds = int(data[0][0])

    output = []

    temp_sum = 0
    previous_time = 0
    previous_acc = 0

    for i, d in enumerate(data):
        if i == 0:
            previous_time, previous_acc = d[0], d[1]
        else:
            if int(d[0]) == seconds:

                #temp_sum += previous_acc * previous_acc * (d[0] - previous_time)
                temp_sum += previous_acc * previous_acc
                previous_time, previous_acc = d[0], d[1]
            else:
                output.append([int(seconds), temp_sum])
                seconds = int(d[0])
                temp_sum =  previous_acc * previous_acc
                previous_time, previous_acc = d[0], d[1]
    output.append([seconds, temp_sum])

    output = np.array(output)
    final_output = []

    for aa in range(len(output)):
        begin = int(max(0, aa - (size - 1) / 2))
        end = int(min(len(output) - 1, aa + (size - 1) / 2 + 1))
        final_output.append([output[aa, 0], sum(output[begin : end, 1])])
  


    return final_output


def read_datafile(filename):
    file=open(filename,"r")
    reader = csv.reader(file)       

    data=[]
    
    for row in reader:
        data.append([int(float(row[0])), float(row[1])])

    file.close()
    return data

def read_rawdatafile(filename):
    file=open(filename,"r")
    reader = csv.reader(file)       

    data=[]
    
    for row in reader:
        data.append([float(row[0]), float(row[1])])
    file.close()
    return data

def read_labelfile(filename):
    file=open(filename,"r")
    reader = csv.reader(file)       

    data=[]
    this_detail=[]
    
    for row in reader:
    	data.append([int(row[0]), row[1]])
    file.close()

    return data


def diff(baby, mom):
    baby = np.array(baby)
    mom = np.array(mom)

    available_seconds = list(set(mom[:,0]) & set(baby[:,0]))

    mom = np.array([[x[0], x[1]] for x in mom if x[0] in available_seconds])

    baby = np.array([[x[0], x[1]] for x in baby if x[0] in available_seconds])

    differences = []

    for i in range(len(baby)):
        temp=[]
        temp.append(baby[i][0])
        momC = mom[i][1]
        babyC = baby[i][1]
        temp.append(abs(babyC-momC))
        differences.append(temp)
    return differences



def frange(start, end, step):
    tmp = [start]
    cur = start + step
    while(abs(cur - end) > 0.001):       
        tmp.append(cur)
        cur += step
    return tmp


def calculateNewCorrelation(mom, baby, size):

    lala = int((size - 1) / 2)

    seconds = mom[0][0]

    mom_data = {}
    temp = []

    for i in range(len(mom)):
        if  mom[i][0] == seconds:
            temp.append(mom[i][1])
        else:
            mom_data[seconds] = temp
            seconds = mom[i][0]
            temp = []

    seconds = baby[0][0]

    baby_data = {}
    temp = []

    for i in range(len(baby)):
        if  baby[i][0] == seconds:
            temp.append(baby[i][1])
        else:
            baby_data[seconds] = temp
            seconds = baby[i][0]
            temp = []

    key_to_be_deleted = []
    for key in mom_data:
        if key not in baby_data:
            key_to_be_deleted.append(key)
        else:
            len_mom = len(mom_data[key])
            len_baby = len(baby_data[key])
            if len_mom < len_baby:
                after_hz = frange(0, 1, 1.0 / len_baby)
                before_hz = frange(0, 1, 1.0 / len_mom)
                mom_data[key] = list(np.interp(after_hz, before_hz, mom_data[key]))
            elif len_mom > len_baby:
                after_hz = frange(0, 1, 1.0 / len_mom)
                before_hz = frange(0, 1, 1.0 / len_baby)

                baby_data[key] = list(np.interp(after_hz, before_hz, baby_data[key]))



    for k in key_to_be_deleted:
        del mom_data[k]

    new_data = []

    for key in mom_data:
        temp_mom = []
        temp_baby = []

        for i in range(lala, 0, -1):
            if key - i in mom_data:
                temp_mom.extend(mom_data[key - i])
                temp_baby.extend(baby_data[key - i])

        temp_mom.extend(mom_data[key])
        temp_baby.extend(baby_data[key])


        for i in range(1, lala + 1, 1):
            if key + i in mom_data:
                temp_mom.extend(mom_data[key + i])
                temp_baby.extend(baby_data[key + i])


        temp = np.corrcoef(temp_mom, temp_baby)[0][1]
        new_data.append([key, temp])


    return np.asarray(new_data)





def calculateVariance(data, size):
    lala = int((size - 1) / 2)

    seconds = data[0][0]

    second_data = {}
    temp = []

    for i in range(len(data)):
        if  data[i][0] == seconds:
            temp.append(data[i][1])
        else:
            second_data[seconds] = temp
            seconds = data[i][0]
            temp = []

    new_data_variance = []
    new_data_frequency = []

    for key in second_data:
        temp_data = []

        for i in range(lala, 0, -1):
            if key - i in second_data:
                temp_data.extend(second_data[key - i])

        temp_data.extend(second_data[key])


        for i in range(1, lala + 1, 1):
            if key + i in second_data:
                temp_data.extend(second_data[key + i])


        Hn = fft.fft(temp_data)
        freqs = fft.fftfreq(len(temp_data), 1/64.0)
        idxs = np.argsort(np.abs(Hn))
        temp_frequency = freqs[idxs[-2]]
        
        temp_variance = np.var(temp_data)

        new_data_variance.append([key, temp_variance])
        new_data_frequency.append([key, temp_frequency])


    return np.asarray(new_data_variance), np.asarray(new_data_frequency)
  

if __name__ == '__main__':


    recalls = []
    precisions = []
    acc = []
    for data_folder in all_data_folder:
        files = os.listdir(data_folder)

        print(data_folder)
        features = []
	    

        for f in files:
            if 'acc' in f:
                if 'mom' in f:
                    rawAccMom=read_rawdatafile(data_folder+str(f))
                    accMomData = read_datafile(data_folder + str(f))      
                else:   
                    accBabyData = read_datafile(data_folder + str(f))
                    rawAccBaby=read_rawdatafile(data_folder+str(f))
            elif 'label' in f:
        	    labels = read_labelfile(data_folder + str(f))


        accMomData = np.array(accMomData)
        accBabyData = np.array(accBabyData)
        accMomData[:, 1] = savgol_filter(accMomData[:, 1], 5, 2, mode = 'mirror')
        accBabyData[:, 1] = savgol_filter(accBabyData[:, 1], 5, 2, mode = 'mirror')

        print("smoothing done")


        accCor_9 = calculateNewCorrelation(accMomData, accBabyData, window_size)
        accEne = calculateEnergy(accCor_9, 3)

        print("Correlation features done")

        momVar_9, momFre_9 = calculateVariance(accMomData, window_size)
        babyVar_9, babyFre_9 = calculateVariance(accBabyData, window_size)

        print("variance features done")

        varDiff_9 = diff(momVar_9, babyVar_9)

        print("features done")
     
        features = [accCor_9, varDiff_9, momVar_9, babyVar_9, accEne]
        for it in range(len(features)):
            features[it] = np.array([x[1] for x in features[it]])


        result_file_name = "final.csv"
       
        result_file = open(data_folder + result_file_name,'w', newline='')
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(features)
        result_file.close()

