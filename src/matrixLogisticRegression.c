# include <stdio.h>
# include "matrixAlgebra.h"

# define T_MAX 1000																// Maximum number of iterations of Alternating Minimization.
# define lambda 0.01															// Regularizer parameter.
# define alpha 0.01																// Step length
# define VAL_FREQ 10															// Validation frequency.
# define ERROR_THRESHOLD 0.0001													// Error Threshold to quit Alternating Minimization if error on validation set is below a minimal threshold.
# define VALIDATE 0
# define DEBUG 0

const int TVT_ratio = 622 ;												// Train-Validate-Test data ratio. eg: 622 => 60%, 20%, 20%.

// Signum function
int sign(double n)						
{
	if(n > 0)
		return 1;
	else
		return -1;
}

// Logistic Loss function
double logisticLoss(tensor *X, matrix *Y, matrix *U, matrix *V)
{
	int i,N,m,n,k;
	double Loss = 0;
	double loss, y;
	matrix *x;

	N = X->n_samples;
	m = X->n_rows;
	n = X->n_cols;
	k = U->n_cols;

	matrix *V_T = newMatrix(k,n);
	matrix *UV = newMatrix(m,n);
	matrix *x_T = newMatrix(n,m);
	matrix *UVX = newMatrix(m,m);

	transpose(V,V_T);
	matrixMultiply(U,V_T, UV);

	freeMatrix(V_T);

	for(i=0; i< N; i++)
	{
		x =getMatrix(X,i);
		y = getM(Y, i, 0);

		transpose(x,x_T);
		matrixMultiply(UV,x_T,UVX);
		loss = log(1 + exp(-y*trace(UVX)));
		Loss = Loss + loss;
	}

	freeMatrix(x_T);
	freeMatrix(UV);
	freeMatrix(UVX);

	Loss = Loss/(double)N;
	return Loss;
}

// Reads and copies X and Y from R arrays
void readRdata(double *p_X_train, double *p_Y_train, tensor *X_train, matrix *Y_train, int N, int m, int n)
{
	reshapeT(X_train, N, m, n);
	X_train->mat = p_X_train;

	reshapeM(Y_train, N,1);
	Y_train->mat = p_Y_train;
}

// Reads and copies U and V from R arrays
void readRmodel(double *p_U, double *p_V, matrix *U, matrix *V, int m, int n, int k)
{
	reshapeM(U, m, k);
	U->mat = p_U;

	reshapeM(V, n, k);
	V->mat = p_V;
}

// Train a Matrix Logistic Regrresion Model with data from R
void trainMatrixLogisticRegression(double *p_X_train, double *p_Y_train, int *p_N, int *p_m, int *p_n, int *p_k, double *p_U, double *p_V)
{
	// To generate random numbers.
	time_t tm;								
	srand((unsigned) time(&tm));											

	int i,j,t;

	t = 1;
	double obj = 0;

	int k = *p_k;																		// Desired low rank
 	int N = *p_N;																		// Number of Samples	
	int m = *p_m;																		// Number of rows
	int n = *p_n;																		// Number of columns

	tensor* X_train = emptyTensor();
	matrix* Y_train = emptyMatrix();

	readRdata(p_X_train, p_Y_train, X_train, Y_train, N, m, n);

	tensor* X_val = emptyTensor();
	matrix* Y_val = emptyMatrix();

	tensor* X_test = emptyTensor();
	matrix* Y_test = emptyMatrix();

	if(VALIDATE)
		constructValidationTest(X_train, Y_train, X_val, Y_val, X_test, Y_test, TVT_ratio);
														
	int N_train = X_train->n_samples;														// Number of samples in training set

	int mk = m*k;
	int nk = n*k;

	double y, alpha_opt;

	matrix *U = newMatrix(m, k);
	matrix *V = newMatrix(n, k);

	matrix *U_term_vec_V_V0 =  newMatrix(nk,1);
	matrix *U_term_2_add = newMatrix(mk,1);
	matrix *U_term_2 = newMatrix(mk,1);

	matrix *V_term_vec_U_U0 = newMatrix(mk, 1);
	matrix *V_term_2_add = newMatrix(nk,1);
	matrix *V_term_2 = newMatrix(nk,1);

	matrix *V_old =  newMatrix(n,k);
	matrix *vec_V_old = vec(V_old);

	randomizeMatrix(U);
	randomizeMatrix(V);

	if(DEBUG)
	{
		ones(m,k,U);
		ones(n,k,V);
	}

	matrix *vec_U = vec(U);
	matrix *vec_V = vec(V);

	matrix *U0 = newMatrix(m,k);
	matrix *V0 = newMatrix(n,k);
	internalCopy(U, U0);
	internalCopy(V, V0);

	matrix *vec_U0 = vec(U0);
	matrix *vec_V0 = vec(V0);

	matrix *x_T = newMatrix(n, m);
	matrix *XV = newMatrix(m, k);
	matrix *vec_XV_T = newMatrix(1, mk);
	matrix *XU = newMatrix(n, k);
	matrix *vec_XU_T = newMatrix(1, nk);

	// Matrices for Gradient and Hessian
	matrix *grad_U_term = newMatrix(mk,1);
	matrix *grad_V_term = newMatrix(nk,1);

	matrix *hess_UU_term = newMatrix(mk,mk);
	matrix *hess_VV_term = newMatrix(nk,nk);

	matrix *hess_VU_term = newMatrix(mk,nk);
	matrix *hess_VU_term1 = newMatrix(mk,nk);
	matrix *hess_VU_term2 = newMatrix(mk,nk);

	matrix *grad_U_sum_term = newMatrix(mk,1);
	matrix *grad_V_sum_term = newMatrix(nk,1);

	matrix *hess_UU_sum_term = newMatrix(mk,mk);
	matrix *hess_VU_sum_term = newMatrix(mk,nk);
	matrix *hess_VV_sum_term = newMatrix(nk,nk);

	matrix *exp_term_matrix = newMatrix(m,m);

	matrix *x, *vec_XV, *vec_XU;

	double exp_term, hess_exp_term, term_exp_power, exp_term_exp_power;

	matrix *V_T = newMatrix(k,n);
	matrix *UV = newMatrix(m,n);
	matrix *minus_eye_k = newMatrix(k,k);

	eye(k, minus_eye_k);
	scalarMultiplyInPlace(-1,minus_eye_k);

	// Gradient constant terms
	matrix *grad_U = newMatrix(mk,1);
	matrix *grad_V = newMatrix(nk,1);

	// Hessian constant terms
	matrix *hess_UU = newMatrix(mk, mk);
	matrix *hess_VV = newMatrix(nk, nk);
	matrix *hess_UV =  newMatrix(nk, mk);
	matrix *hess_VU =  newMatrix(mk, nk);

	// eye(mk, hess_UU);
	// eye(nk, hess_VV);

	// scalarMultiplyInPlace(lambda, hess_UU);
	// scalarMultiplyInPlace(lambda, hess_VV);

	double obj_i = logisticLoss(X_train, Y_train, U, V);
	obj_i = obj_i + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));
	printf("Training: Initial Objective = %f\n",obj_i);

	//Alternating Minimization starts
	while(t<=T_MAX)
	{
		printf("T = %d\n",t);

		// Calculating Grad and Hessian
		transpose(V0, V_T);
		matrixMultiply(U0,V_T, UV);

		scalarMultiply(lambda, vec_U0, grad_U);	
		scalarMultiply(lambda, vec_V0, grad_V);	

		zeros(grad_U_sum_term);
		zeros(grad_V_sum_term);
		zeros(hess_UU_sum_term);
		zeros(hess_VV_sum_term);
		zeros(hess_VU_sum_term);
		
		eye(mk, hess_UU);
		eye(nk, hess_VV);

		scalarMultiplyInPlace(lambda, hess_UU);
		scalarMultiplyInPlace(lambda, hess_VV);

		for(i=0; i<N_train; i++)
		{
			x = getMatrix(X_train, i);
			y = getM(Y_train, i, 0);

			transpose(x,x_T);

			matrixMultiply(x,V0,XV);
			matrixMultiply(x_T,U0,XU);

			vec_XV = vec(XV);
			vec_XU = vec(XU);

			transpose(vec_XV,vec_XV_T);
			transpose(vec_XU,vec_XU_T); 
			matrixMultiply(UV, x_T, exp_term_matrix);
			term_exp_power = - y * trace(exp_term_matrix);
			exp_term_exp_power = exp(term_exp_power);								
			exp_term = y * exp_term_exp_power/(1 + exp_term_exp_power);				
			hess_exp_term = y/(1 + exp_term_exp_power);								

			// Gradient U term
			scalarMultiply(exp_term, vec_XV, grad_U_term);
			matrixAddInPlace(grad_U_sum_term, grad_U_term);

			// Gradient V term
			scalarMultiply(exp_term, vec_XU, grad_V_term);
			matrixAddInPlace(grad_V_sum_term, grad_V_term);

			// Hessian UU term
			matrixMultiply(vec_XV, vec_XV_T,hess_UU_term);
			scalarMultiplyInPlace(hess_exp_term * exp_term, hess_UU_term);
			matrixAddInPlace(hess_UU_sum_term, hess_UU_term);	

			// Hessian VV term
			matrixMultiply(vec_XU, vec_XU_T, hess_VV_term);
			scalarMultiplyInPlace(hess_exp_term * exp_term, hess_VV_term);
			matrixAddInPlace(hess_VV_sum_term, hess_VV_term);

			// Hessian VU term
			// printf("hessVU\n");
			matrixMultiply(vec_XV, vec_XU_T, hess_VU_term1);
			scalarMultiplyInPlace(hess_exp_term, hess_VU_term1);
			kron(minus_eye_k, x, hess_VU_term2); 
			matrixAdd(hess_VU_term1, hess_VU_term2, hess_VU_term);
			scalarMultiplyInPlace(exp_term, hess_VU_term);
			matrixAddInPlace(hess_VU_sum_term, hess_VU_term);

		}

		scalarMultiplyInPlace(-1/(double)N_train, grad_U_sum_term);
		scalarMultiplyInPlace(-1/(double)N_train, grad_V_sum_term);

		matrixAddInPlace(grad_U, grad_U_sum_term);
		matrixAddInPlace(grad_V, grad_V_sum_term);

		scalarMultiplyInPlace(1/(double)N_train, hess_UU_sum_term);
		matrixAddInPlace(hess_UU, hess_UU_sum_term);

		scalarMultiplyInPlace(1/(double)N_train, hess_VV_sum_term);
		matrixAddInPlace(hess_VV, hess_VV_sum_term);

		scalarMultiply(1/(double)N_train, hess_VU_sum_term, hess_VU);
		transpose(hess_VU, hess_UV);

		if(DEBUG)
		{
			printf("U0 = %f\n",frobNorm(U0));
			printf("V0 = %f\n",frobNorm(V0));
			printf("GRAD_U = %f\n",frobNorm(grad_U));
			printf("GRAD_V = %f\n",frobNorm(grad_V));
			printf("HESS_UU = %f\n",frobNorm(hess_UU));
			printf("HESS_VV = %f\n",frobNorm(hess_VV));
			printf("HESS_UV = %f\n",frobNorm(hess_UV));
			printf("HESS_VU = %f\n",frobNorm(hess_VU));
		}
	
		// Compute U
		matrixSubtract(vec_V,vec_V0, U_term_vec_V_V0);							
		matrixMultiply(hess_VU,U_term_vec_V_V0, U_term_2_add);					
		matrixAddInPlace(U_term_2_add, grad_U);
		inverse(hess_UU);
		matrixMultiply(hess_UU, U_term_2_add, U_term_2);						

		alpha_opt = alpha;
		// alpha_opt = alpha/sqrt(t);
		scalarMultiplyInPlace(-alpha_opt, U_term_2);

		matrixAdd(vec_U0, U_term_2, vec_U);
		U = unvec(vec_U, m, k);

		// Compute V
		internalCopy(vec_V, vec_V_old);

		matrixSubtract(vec_U,vec_U0, V_term_vec_U_U0);							
		matrixMultiply(hess_UV,V_term_vec_U_U0, V_term_2_add);					
		matrixAddInPlace(V_term_2_add, grad_V);
		inverse(hess_VV);
		matrixMultiply(hess_VV, V_term_2_add, V_term_2);						

		alpha_opt = alpha;
		// alpha_opt = alpha/sqrt(t);
		scalarMultiplyInPlace(-alpha_opt, V_term_2);

		matrixAdd(vec_V0, V_term_2, vec_V);
		V = unvec(vec_V, n, k);

		if(DEBUG)
		{
			printf("U = %f\n",frobNorm(U));
			printf("V = %f\n",frobNorm(V));
		}

		internalCopy(vec_U,vec_U0);									
		internalCopy(vec_V_old, vec_V0);											// Store previous U and V's

		U0 =unvec(vec_U0,m,k);
		V0 =unvec(vec_V0,n,k);
	
		// Calculate Objective/Validate.
		if(t % VAL_FREQ == 0)
		{
			// Calculate Objective on Training set.
			obj = logisticLoss(X_train, Y_train, U, V);
			obj = obj + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));
			printf("Training: Objective = %f\n",obj);

			// Calculate Objective on Validation set.
			if(VALIDATE)
			{
				obj = logisticLoss(X_val, Y_val, U, V);
				obj = obj + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));
				printf("Validation: Objective = %f\n",obj);
				if(obj < ERROR_THRESHOLD)
					break;
			}
		}
		t++;
	}
	// Copy model arrays into R arrays
	for(i=0;i<m;i++)
	{
		for(j=0;j<k;j++)
			p_U[i + j*m] = getM(U,i,j);
	}

	for(i=0;i<m;i++)
	{
		for(j=0;j<k;j++)
			p_V[i + j*n] = getM(V,i,j);
	}

	// Free memory allocated to matrices
	freeMatrix(x_T);
	freeMatrix(XU);
	freeMatrix(XV);
	freeMatrix(vec_XU_T);
	freeMatrix(vec_XV_T);
	freeMatrix(exp_term_matrix);
	freeMatrix(grad_U_term);
	freeMatrix(grad_V_term);
	freeMatrix(hess_UU_term);
	freeMatrix(hess_VV_term);
	freeMatrix(hess_VU_term1);
	freeMatrix(hess_VU_term2);
	freeMatrix(hess_VU_term);

	freeMatrix(V_T);
	freeMatrix(UV);

	freeMatrix(grad_U_sum_term);
	freeMatrix(grad_V_sum_term);
	freeMatrix(hess_UU_sum_term);
	freeMatrix(hess_VV_sum_term);
	freeMatrix(hess_VU_sum_term);

	freeMatrix(U_term_vec_V_V0);
	freeMatrix(U_term_2_add);
	freeMatrix(U_term_2);	

	freeMatrix(V_term_vec_U_U0);
	freeMatrix(V_term_2_add);
	freeMatrix(V_term_2);

	freeMatrix(U0);
	freeMatrix(V0);
	freeMatrix(V_old);

	freeMatrix(grad_U);
	freeMatrix(grad_V);
	freeMatrix(hess_UU);
	freeMatrix(hess_VV);
	freeMatrix(hess_UV);
	freeMatrix(hess_VU);

	printf("Training: Initial Objective = %f\n",obj_i);

	if(VALIDATE)
	{
		// Calculate Objective on Test set.
		obj = logisticLoss(X_test, Y_test, U, V);
		obj = obj + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));
		printf("Testing: Objective = %f\n",obj);
	}

	freeMatrix(U);
	freeMatrix(V);
}

// Predict on Unknown data with trained model and data from R
void predictMatrixLogisticRegression(double *p_X_test, double *p_Y_test, int *p_N, int *p_m, int *p_n, int *p_k, double *p_U, double *p_V, int *p_Y_present, double *p_Y_predict, double *accuracy)
{
	int i;
	int k = *p_k;																		// Desired low rank
	int N = *p_N;																		// Number of Samples	
	int m = *p_m;																		// Number of rows
	int n = *p_n;																		// Number of columns
	int Y_present = *p_Y_present;

	tensor *X_test = emptyTensor();
	matrix *Y_test = emptyMatrix();
	matrix *U = emptyMatrix();
	matrix *V = emptyMatrix();

	matrix *V_T = newMatrix(k,n);
	matrix *UV = newMatrix(m,n);
	matrix *x_T = newMatrix(n,m);
	matrix *UVx = newMatrix(m,m);
	matrix *Y_predict = newMatrix(N,1);
	matrix *Y_minus_Y_predict = newMatrix(N,1);

	matrix *x;

	readRdata(p_X_test, p_Y_test, X_test, Y_test, N, m, n);
	readRmodel(p_U, p_V, U, V, m, n, k);

	transpose(V,V_T);
	matrixMultiply(U,V_T,UV);

	// Get Predicted Y
	for(i=0;i<N;i++)
	{
		x = getMatrix(X_test,i);
		transpose(x, x_T);
		matrixMultiply(UV, x_T, UVx);
		setM(Y_predict,i,0,sign(1/(1 + exp(-trace(UVx))) - 0.5));
	}
	// Copy into R array
	for(i=0;i<N;i++)
		p_Y_predict[i] = getM(Y_predict,i,0);

	// If true Y present, report Accuracy
	if (Y_present == 1)
	{
		matrixAdd(Y_test, Y_predict, Y_minus_Y_predict);
		*accuracy = norm(Y_minus_Y_predict,0)/(double)N;
	}

	freeMatrix(V_T);
	freeMatrix(UV);
	freeMatrix(x_T);
	freeMatrix(UVx);
	freeMatrix(Y_predict);
	freeMatrix(Y_minus_Y_predict);
}







