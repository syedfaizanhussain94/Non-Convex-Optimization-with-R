f = file.path(getwd(), 'src/robustPCA.so')
dyn.load(f)

robustPCA <- function(dataFile,k)
{
	load(dataFile)														# Y matrix loaded from file
	Y_vec = as.double(Y)
	m = dim(Y)[1]
	n = dim(Y)[2]
	out=.C("robustPCA", Y = as.double(Y_vec), m = as.integer(m), n = as.integer(n), k = as.integer(k), U_vec = as.double(matrix(0,m,k)), V_vec = as.double(matrix(0,n,k)), S_vec = as.double(matrix(0,m,n)))
	U <- matrix(out$U_vec, m, k)
	V <- matrix(out$V_vec, n, k)
	S <- matrix(out$S_vec, m, n)
	save(U, V, S, file = "models/model_robustPCA")
}