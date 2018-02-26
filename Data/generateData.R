# Generates Matrix Regressio Data, where (m,n) is the matrix dimensionality, N is the number of samples, and k is the instrinsic low rank
generateMatrixRegressionData <- function(m, n, N, k)
{
	mn = m*n
	mk = m*k
	nk = n*k

	Y= matrix(0,N,1);
	X = list()
	U = matrix(rnorm(mk),m,k)
	V = matrix(rnorm(nk),n,k)
	UV = U %*% t(V)
	for(i in 1:N) 
	{
		x = matrix(rnorm(mn),m,n)
		Y[i,1] = sum(diag((UV %*% t(x))))
		X = c(X,list(x))
	}
	save(X, Y, file = "Data/data_matrixRegression.txt")
}

# Generates Matrix Logistic Regression Data, where (m,n) is the matrix dimensionality, N is the number of samples, and k is the instrinsic low rank
generateMatrixLogisticRegressionData <- function(m, n, N, k)
{
	mn = m*n
	mk = m*k
	nk = n*k

	Y= matrix(0,N,1);
	X = list()
	U = matrix(rnorm(mk),m,k)
	V = matrix(rnorm(nk),n,k)
	UV = U %*% t(V)
	for(i in 1:N) 
	{
		x = matrix(rnorm(mn),m,n)
		Y[i,1] = sign(1/(1 + exp(- sum(diag((UV %*% t(x)))))) - (1/2))
		X = c(X,list(x))
	}
	save(X, Y, file = "Data/data_matrixLogisticRegression.txt")
}

# Generates Robust PCA Data. Sparsity is (max) column sparsity
generateRobustPCAData <- function(m, n, k, sparsity)
{
	mn = m*n
	mk = m*k
	nk = n*k

	U = matrix(rnorm(mk),m,k)
	V = matrix(rnorm(nk),n,k)
	random_indices = ceiling(runif(n*sparsity,min=1,max=m))	# random row indices
	S = matrix(0,m,n)

	for(i in 1:n)
	{
		for(j in 1:sparsity)
			S[random_indices[sparsity*(i-1) + j],i] = rnorm(1)
	}
	Y = (U %*% t(V)) + S
	save(Y, file = "Data/data_robustPCA.txt")
}