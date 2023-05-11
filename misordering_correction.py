import math
import tensorflow as tf
from tensorflow import keras
import pickle
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

nms_type = np.dtype([('value', 'float32'), ('x', int),  ('y', int)])

with (open("dataset_pickle/Predict_data/predict_2.pickle", "rb")) as openfile:
	while True:
		try:
			Y_pred = pickle.load(openfile)
		except EOFError:
			break
with (open("dataset_pickle/Test_data/test_Y.pickle", "rb")) as openfile:
	while True:
		try:
			Y_test = pickle.load(openfile)
		except EOFError:
			break
with (open("dataset_pickle/Test_data/test_Y_wrist.pickle", "rb")) as openfile:
	while True:
		try:
			Y_wrist = pickle.load(openfile)
		except EOFError:
			break
with (open("dataset_pickle/Test_data/test_X.pickle", "rb")) as openfile:
	while True:
		try:
			X_test = pickle.load(openfile)
		except EOFError:
			break

G1_FL = [0, 0, 0, 0, 0]
G2_FL = [0, 1, 0, 0, 0]
G3_FL = [1, 0, 0, 0, 0]
G4_FL = [0, 1, 1, 0, 0]
G5_FL = [1, 1, 1, 1, 1]
G6_FL = [1, 1, 0, 0, 0]
G7_FL = [0, 1, 0, 0, 1]
G8_FL = [1, 0, 0, 0, 1]
G9_FL = [0, 1, 1, 1, 1]
G0_FL = [1, 1, 0, 0, 1]
_gesture_finger_list = [G1_FL, G2_FL, G3_FL, G4_FL, G5_FL, G6_FL, G7_FL, G8_FL, G9_FL, G0_FL]
_gesture_finger_list = np.array(_gesture_finger_list, dtype=np.float32)


Y_pred_fm = Y_pred[2]     # load finger map part(fm + wrist)
Y_pred_cate = Y_pred[1]   # gesture part
Y_pred_LR = Y_pred[0]     # left(0) or right(1)

print("finger map and wrist: ", Y_pred_fm.shape)
print("gesture type        : ", Y_pred_cate.shape)
print("left or right       : ", Y_pred_LR.shape)


def nms_format(org_data):
	X = org_data.shape[0]
	Y = org_data.shape[1]
	res = []
	for j in range(Y):
		for i in range(X):
			tmp = (org_data[j][i], i, j)
			res.append(tmp)
	return res


def cal_dis(point1, point2):
	if len(point1) == 2:
		return math.sqrt(math.pow(point1[0]-point2[0], 2) + math.pow(point1[1]-point2[1], 2))
	elif len(point1) == 3:
		return math.sqrt(math.pow(point1[1]-point2[1], 2) + math.pow(point1[2]-point2[2], 2))
	else:
		print("cal_dis error!!!")
		return 0


def check_distance(p_list, cand):
	# compare distance between each element in p_list with cand
	for i in p_list:
		if cal_dis(i, cand) > 5:
			continue
		else:
			return False
	return True


def non_max_suspend(tmp):
	# Stop after finding 2 points
	nms = np.dtype([('value', 'float32'), ('x', int),  ('y', int)])
	nf = nms_format(tmp)
	tmp = np.array(nf, dtype=nms)
	sorted_list = np.sort(tmp, order='value')[::-1]
	point_1, point_2 = sorted_list[0], None

	for point_2 in sorted_list[1:]:
		if cal_dis((point_1[1], point_1[2]), (point_2[1], point_2[2])) > 10:
			break
	return point_1, point_2


def non_max_suspend_threshold(tmp, threshold=0.2):
	# Stop after finding 5 points or point value below threshold
	nf = nms_format(tmp)
	tmp = np.array(nf, dtype=nms_type)
	sorted_list = np.sort(tmp, order='value')[::-1]
	
	point_1 = sorted_list[0]
	point_2, point_3, point_4, point_5 = None, None, None, None
	point2_index = 1
	for point_2 in sorted_list[1:]:

		if point_2[0] < threshold:
			return np.array([point_1])
		if check_distance([(point_1[1], point_1[2])],  (point_2[1], point_2[2])):
			break
		point2_index += 1

	point3_index = point2_index+1
	for point_3 in sorted_list[point2_index:]:
		if point_3[0] < threshold:
			return np.array([point_1, point_2])
		if check_distance([(point_1[1], point_1[2]), (point_2[1], point_2[2])], (point_3[1], point_3[2])):
			break
		point3_index += 1

	point4_index = point3_index+1
	for point_4 in sorted_list[point3_index:]:
		if point_4[0] < threshold:
			return np.array([point_1, point_2, point_3])
		if check_distance([(point_1[1], point_1[2]), (point_2[1], point_2[2]), (point_3[1], point_3[2])], (point_4[1], point_4[2])):
			break
		point4_index += 1

	# point5_index = point4_index+1
	for point_5 in sorted_list[point4_index:]:
		if point_5[0] < threshold:
			return np.array([point_1, point_2, point_3, point_4])
		if check_distance([(point_1[1], point_1[2]), (point_2[1], point_2[2]), (point_3[1], point_3[2]), (point_4[1], point_4[2])], (point_5[1], point_5[2])):
			break
		# point4_index +=1
	return np.array([point_1, point_2, point_3, point_4, point_5])


def contrast_brightness_increase(alpha, beta, img):
	background = np.ones(img.shape, dtype=np.float32)
	return cv2.addWeighted(img, alpha, background, 1-alpha, beta)


def fm_candidate(fm, lr, cate):
	# 一次处理一个手势的5张fingermaps and 2 wrist points
	thumb = fm[:, :, 0]
	index = fm[:, :, 1]
	middle = fm[:, :, 2]
	ring = fm[:, :, 3]
	little = fm[:, :, 4]
	wrist = fm[:, :, 5]
	arm = fm[:, :, 6]
	# 通过手势找到对应的手指分布list
	cate_index = np.unravel_index(np.argmax(cate), cate.shape)
	gfl = _gesture_finger_list[cate_index]
	# 生成wrist line based on 2 wrist points
	# wrist_line = generate_wrist_line(wrist)
	augment_res = []
	for i in range(5):
		finger_i = fm[:, :, i]
		candidates = non_max_suspend_threshold(finger_i)
		augment_res.append(candidates)
		"""
		if gfl[i] == 1:			
			res.append(candidates)			
		else:
			candidates_non = np.zeros(1, dtype =nms_type )
			res.append(candidates_non)
		"""
	rres = candidate_filter(augment_res, gfl)
	t = order_fm(rres, wrist, arm, lr, gfl)
	return t


def check_duplication(cand, p, replace=False):
	# check if p is in cand list
	if not cand:
		return True
	for i in range(len(cand)):
		print(cand[i])
		if cal_dis(cand[i], p) > 5:
			return True
		else:
			if replace and cand[i][0] < p[0]:
				cand[i][0] = p[0]
	return False


def candidate_filter(cand, gfl):
	cand_list = []
	pred_list = []
	untouchable = []
	finish = []
	for i in range(5):
		if gfl[i] == 0:
			pred_list.append(None)
			finish.append(0)
			# cand_list.append(None)
		elif gfl[i] == 1:
			if len(cand[i]) == 1:
				pred_list.append(cand[i][0])		
				untouchable.append(cand[i][0])
				finish.append(0)
				# cand_list.append(None)
			else:
				pred_list.append(None)
				finish.append(1)
				for t in cand[i]:
					cand_list.append([i, t])
	
	if cand_list:
		# sort by value
		cand_list = sorted(cand_list, key=lambda pvalue: pvalue[1][0], reverse=True)

	for cl in cand_list:
		if check_duplication(untouchable, cl[1]) and finish[cl[0]]:
			pred_list[cl[0]] = cl[1]
			finish[cl[0]] = 0
			untouchable.append(cl[1])
	return pred_list
	

def find_base_wrist_point(wfm, afm, lr):
	# from 2 wrist point, find the one which colse to thumb as base
	a, b = non_max_suspend(wfm)  # a,b (value,x,y)
	arm_p = np.unravel_index(np.argmax(afm), afm.shape)
	x, y = arm_p[1], arm_p[0]
	base = []
	other = []
	if lr >= 0.5:
		# right
		if y > (a[2] + b[2])/2:
			print("right reverse")
			if a[1] > b[1]:
				base = b
				other = a
			else:
				base = a
				other = b
		else:
			if a[1] > b[1]:
				base = a
				other = b
			else:
				base = b
				other = a
	else:
		if y > (a[2] + b[2])/2:
			print("left reverse")
			if a[1] > b[1]:
				base = a
				other = b
			else:
				base = b
				other = a
		else:
			if a[1] > b[1]:
				base = b
				other = a
			else:
				base = a
				other = b

	return base, other


def cal_angle(point1, base, point2):
	p1 = np.array([point1[1], point1[2]])
	b = np.array([base[1], base[2]])
	p2 = np.array([point2[1], point2[2]])
	b1 = p1 - b
	b2 = p2 - b

	cosine_angle = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
	angle = np.arccos(cosine_angle)

	return np.degrees(angle)


def order_fm(fm, wfm, afm, lr, gesture_list):
	# sort fm position base one wrist point and lr
	base, p = find_base_wrist_point(wfm, afm, lr)
	fm_list = []
	sort_list = []
	for i in fm:
		if i:
			ang = cal_angle(p, base, i)
			sort_list.append(ang)
			fm_list.append(i)
	z = [x for _, x in sorted(zip(sort_list, fm_list), key=lambda y:y[0])][::-1]
	result = []
	z_size = len(z)
	z_index = 0
	
	for index in range(len(gesture_list)):
		if gesture_list[index] == 1:
			
			result.append(z[z_index])
			if z_index < z_size-1:
				z_index += 1
		else:
			result.append(None)
	return result


datasize = Y_pred_LR.shape[0]
_augment_res = []

for i_ds in range(datasize):
	fc_tmp = fm_candidate(Y_pred_fm[i_ds], Y_pred_LR[i_ds], Y_pred_cate[i_ds])
	_augment_res.append(fc_tmp)

_augment_res = np.array(_augment_res)
print("Augmentaion shape: ", _augment_res.shape)


pickle_out = open("augment_res_2.pickle", "wb")
pickle.dump(_augment_res, pickle_out)
pickle_out.close()
print("Dump filed successed")





