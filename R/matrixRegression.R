f = file.path(getwd(), 'src/matrixRegression.so')
dyn.load(f)

trainMatrixRegression <- function(dataFile,k)
{
	load(dataFile)														# X and Y loaded from file
	X_vec = as.vector(do.call(cbind, X))
	N = length(X)
	m = dim(X[[1]])[1]
	n = dim(X[[1]])[2]
	out=.C("trainMatrixRegression", X = as.double(X_vec),Y = as.double(Y) , N= as.integer(N), m = as.integer(m), n = as.integer(n), k = as.integer(k), U_vec = as.double(matrix(0,m,k)), V_vec = as.double(matrix(0,n,k)))
	U <- matrix(out$U_vec, m, k)
	V <- matrix(out$V_vec, n, k)
	save(U, V, file = "models/model_matrixRegression")
}

predictMatrixRegression <- function(dataFile,model)
{
	load(dataFile)														# X and Y loaded (for X_test, Y_test)
	load(model)															# U and V loaded
	X_vec = as.vector(do.call(cbind, X))
	U_vec = as.double(U)
	V_vec = as.double(V)

	N = length(X)
	m = dim(U)[1]
	n = dim(V)[1]
	k = dim(V)[2]

	# If true responses are/aren't present
	Y_present = 1
	if(!exists('Y'))
	{
		Y_present = 0
		Y = matrix(0,N,1)
	}
	N = length(X)
	m = dim(X[[1]])[1]
	n = dim(X[[1]])[2]

	out=.C("predictMatrixRegression", X = as.double(X_vec),Y = as.double(Y) , N= as.integer(N), m = as.integer(m), n = as.integer(n), k = as.integer(k), U = as.double(U_vec), V = as.double(V_vec), Y_present = as.integer(Y_present), Y_predict_vec = as.double(matrix(0,N,1)), accuracy = as.double(0))
	Y_predict <- as.vector(matrix(out$Y_predict_vec, N, 1))

	if(Y_present == 1)
		cat("Accuracy = ",out$accuracy)
	return(Y_predict)
}
