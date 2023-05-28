import numpy as np
import csv, os
from  sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, precision_recall_fscore_support, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as guss
from scipy.stats import pearsonr
from scipy.stats import rankdata
from collections import Counter
from datetime import datetime, timedelta
import xmltodict
import pickle


participant = 'P39'

def minute_rounder(t):
    # Rounds to nearest minute by adding a timedelta minute if second >= 30
    return (t.replace(second=0, microsecond=0, minute=t.minute)
               +timedelta(minutes=t.second//30))


def clean_test(data, size):
	lalala = int((size - 1) / 2)
	num = len(data)
	output = []
	
	for i in range(num):
		front = max(0, i - lalala)
		rear = min(num, i + 1 + lalala)
		temp = Counter(data[front:rear]).most_common()
		output.append(temp[0][0])

	##change the threshold values here
	output = whatIsAnEvent(combineIntoEvent(output, 30), 10)
	return output





def whatIsAnEvent(data, event_thre):
    previous = (-1, -1)
    start = (-1, -1)
    for i in range(len(data)):
        if data[i, 1] == 1 and previous[1] == -1:
            previous = (i, data[i, 0])
        elif data[i, 1] == 0 and previous[1] != -1 and data[i - 1, 1] == 1:
            start = (i, data[i, 0])
            if start[1] - previous[1] <= event_thre:
                data[previous[0] : start[0], 1] = 0
            previous = (-1, -1)
            start = (-1, -1)

    if previous[1] != -1 and data[-1, 0] - previous[1] + 1 <= event_thre:
        data[previous[0] :, 1] = 0
    return data[:, 1]


def combineIntoEvent(data, time_thre):

	data = np.asarray(data)
	time = np.asarray(range(len(data)))
	data = np.column_stack((time,data))
	previous = (-1, -1)
	for i in range(len(data)):
	    if data[i, 1] == 1:
	        start = (i, data[i, 0])
	        if previous[1] > 0 and start[1] - previous[1] <= time_thre:
	            data[previous[0] : start[0], 1] = 1
	        previous = start

	if previous[1] > 0 and data[i - 1, 0] - previous[1] <= time_thre:
	    data[previous[0] : i, 1] = 1

	return data


def read_finalfile(filename, use):
    file=open(filename,"r")
    reader = csv.reader(file)       

    data=[]
    
    len_list = []
    for row in reader:
    	len_list.append(len(row))
    	data.append(list(map(float, row)))

    if np.std(len_list) > 0:
    	min_len = np.min(len_list)
    	for i in range(len(len_list)):
    		data[i] = data[i][:min_len]

    data = np.asarray(data)
    file.close()
    output_data = np.transpose(data)

    return output_data

def read_unisens_time(file):
    with open(file) as f:
        doc = xmltodict.parse(f.read())
        time_string = doc['unisens']['@timestampStart'][0 : 23] + '000'
        start_time = datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")

    return start_time

def collapseHolding(zipped):
	output = []

	previous = (-1, -1)

	for ind, d in enumerate(zipped):
		if d[1] > 0: #holding
			if previous[0] == -1:
				previous = (d[0], d[1])
		else: #not holding
			if previous[0] != -1:
				output.append([minute_rounder(previous[0]), minute_rounder(zipped[ind - 1][0])])
				previous = (-1, -1)

	if previous[0] != -1:
		output.append([minute_rounder(previous[0]), minute_rounder(zipped[- 1][0])])


	return output


def process_holdingPrediction(data_folder):
	###this simply aims to get a start time such that the output time column can have a real world time
	mom_start = read_unisens_time(data_folder + 'mom/unisens.xml')

	output = []

	window_size  = 7
	total_holding_seconds = []
	total_holding_per = []
	total_seconds = []


	train_data = []
	train_label = []
	test_data = read_finalfile(data_folder + "final.csv", "test")

	loaded_model = pickle.load(open("./holding_model.pkl", 'rb') , encoding='latin1')
	pred_label = loaded_model.predict(test_data)  
	pred_label = clean_test(pred_label, window_size)

	times = int(len(pred_label) / 60)
	f = int(len(pred_label) % 60)
	for i in range(times):
		time = mom_start + timedelta(minutes = i)
		if sum(pred_label[i * 60: (i * 60 + 60)]) > 0:
			output.append([time, 1])
		else:
			output.append([time, 0])
	if f != 0:
		time = mom_start + timedelta(minutes = times)
		if sum(pred_label[i * 60: ]) > 0:
			output.append([time, 1])
		else:
			output.append([time, 0])

	return collapseHolding(output)


holding_output =  process_holdingPrediction('./' + participant + '/')
print(holding_output)
writer = csv.writer(open(participant + "_holding_predictions.txt", 'w') )
writer.writerows(holding_output)

