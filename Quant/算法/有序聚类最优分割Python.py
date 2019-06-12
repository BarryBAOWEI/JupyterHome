import pandas as pd
import numpy as np

'''
有序聚类，与一般聚类的输入相似，但会考虑到输入数据的顺序。
特别适合用于多元时间序列数据分段。
原理，需要分为N类，则寻找N-1个最优分割点，使得每个分割段内的所有点距离分割段中心的马氏距离总和最小。
欲找到第N-1个最优分割点，需要找到第N-1段已经被划分走后的第N-2个最优分割点，用到递归算法。
寻找的是全局最优解，数据长度大，分割点多是运算及其缓慢。
举例：用于股市风格分段，输入序列为（收盘价，交易量，换手率，近期波动，近期均价等）多元时间序列。
'''


def ocluster(dataframe, classnum):
	'''
	dataframe: pandas中的DataFrame对象,行为观测,列为变量,最好标准化
	classnum : 超参数,欲分类的数量，2-4类为宜
	'''
	sampleNum = len(dataframe)
	sampleIndex = dataframe.index
	sampleMatrix = np.array(dataframe)
	
	def diameterCompute(i ,j ):
		sampleMatrixItoJ = sampleMatrix[i:(j+1),:]
		muVec = np.mean(sampleMatrixItoJ,axis = 0)
		sampleMatrixItoJMinusMu = np.apply_along_axis((lambda x: x-muVec),1,sampleMatrixItoJ)
		distance = 0
		for cnt in range(j-i+1):
			vecForSample = sampleMatrixItoJMinusMu[cnt,]
			distance += np.dot(vecForSample,vecForSample)
		return distance
	
	diameterMatrix = np.zeros((sampleNum,sampleNum))
	for row in range(sampleNum-1):
		for col in range(row+1,sampleNum):
			diameterMatrix[row,col]=diameterCompute(row,col)
			diameterMatrix[col,row]=diameterCompute(row,col)
	
	leastLost = np.zeros((sampleNum-1,sampleNum-1))
	classId = np.diag(range(1,sampleNum))
	
	# 递归函数，各种输入都严格遵照公式，仅在矩阵取值时按照python特点“-1”；输出argmin时，直接返回的是list的索引，需要赋值到j的范围上！
	def eP2(n):
		num = n + 1
		series = [diameterMatrix[0,j-2]+diameterMatrix[j-1,n-1] for j in range(2,num)]
		return np.min(series),range(2,num)[np.argmin(series)]
	def eP(n,k):
		if k == 2:
			return eP2(n) 
		num = n + 1
		series = [eP(j-1,k-1)[0]+diameterMatrix[j-1,n-1] for j in range(k,num)]
		return np.min(series),range(k,num)[np.argmin(series)]

	restNum = sampleNum + 1
	catList = []
	for i in range(classNum-1):
		k = classNum - i
		# 总样本是 1 至 （上一步最优点序号-1）
		restNum = restNum-1
		_,restNum = eP(restNum,k)
		# 保存到catList中时-1变成python中的索引
		catList.append(restNum-1)
	catList = list(np.sort(catList))
	
	return list(sampleIndex[catList])