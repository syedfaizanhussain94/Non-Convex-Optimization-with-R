# Non-convex-optimization-with-R
-----------------------------------------------------------------------------------------------------------------------------------------------------

MATRIX REGRESSION

#Generate Data:
source('Data/generateData.R')
generateMatrixRegressionData(10,15,1000,2)		# Feature matrices of dimensions 10x15, 1000 samples, intristic low rank dimensionaliy of model = 2
# Data saved in the Data directory: data_matrixRegression.txt

#Train Model:
source('R/matrixRegression.R')
trainMatrixRegression('Data/data_matrixRegression.txt',2)		# Data file path, and desired low rank (k) = 2
# Trained model saved in models directory: model_matrixRegression

#Predict
source('R/matrixRegression.R')
predictMatrixRegression('Data/data_matrixRegression.txt','models/model_matrixRegression')		# Data file path, and trained model path
# Returns an array of predicted responses.
# Note: If the data file also has true responses, it prints the accuracy as well.

-----------------------------------------------------------------------------------------------------------------------------------------------------

MATRIX LOGISTIC REGRESSION

#Generate Data:
source('Data/generateData.R')
generateMatrixLogisticRegressionData(10,15,1000,2)	# Feature matrices of dimensions 10x15, 1000 samples, intristic low rank dimensionaliy of model = 2
# Data saved in the Data directory: data_matrixLogisticRegression.txt

#Train Model:
source('R/matrixLogisticRegression.R')
trainMatrixLogisticRegression('Data/data_matrixLogisticRegression.txt',2)		# Data file path, and desired low rank (k) = 2
# Trained model saved in models directory: model_matrixLogisticRegression

#Predict
source('R/matrixLogisticRegression.R')
predictMatrixLogisticRegression('Data/data_matrixLogisticRegression.txt','models/model_matrixLogisticRegression')  # Data file path, and trained model path
# Returns an array of predicted responses.
# Note: If the data file also has true responses, it prints the accuracy as well.

-----------------------------------------------------------------------------------------------------------------------------------------------------

ROBUST PCA

#Generate Data:
source('Data/generateData.R')
generateRobustPCAData(10,15,2, 1)		# Feature matrices of dimensions 10x15, intristic low rank dimensionaliy of model = 2, column sparsity of S <=1
# Data saved in the Data directory: data_robustPCA.txt

#Train Model:
source('R/robustPCA.R')
robustPCA('Data/data_robustPCA.txt',2)		# Data file path, and desired low rank (k) = 2
# Trained model saved in models directory: model_robustPCA
