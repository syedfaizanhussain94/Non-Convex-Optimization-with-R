# include <stdio.h>
# include "matrixAlgebra.h"
// # include "mat.h"	

# define T_MAX 1000					// Maximum number of iterations of Alternating Minimization.
# define lambda 0.01				// Regularizer parameter.
# define VAL_FREQ 10				// Validation frequency.
# define ERROR_THRESHOLD 0.0001		// Error Threshold to quit Alternating Minimization if error on validation set is below a minimal threshold.
# define VALIDATE 0
# define DEBUG 0

const int TVT_ratio = 622 ;												// Train-Validate-Test data ratio. eg: 622 => 60%, 20%, 20%.

// Calculates the least squared loss
double leastSquaredLoss(matrix *U, matrix *V, tensor *X, matrix *Y)
{
	int i,N,m,n,k;
	N = X->n_samples;
	m = X->n_rows;
	n = X->n_cols;
	k = U->n_cols;

	matrix *x;
	double y, y_pred;
	double obj = 0;
	matrix *V_T = newMatrix(k,n);
	transpose(V, V_T);								
	matrix *UV = newMatrix(m, n);
	matrixMultiply(U, V_T, UV);					
	freeMatrix(V_T);

	matrix *x_T = newMatrix(n, m);
	matrix *UVX = newMatrix(m, m);

	for(i=0; i<N; i++)
	{
		x = getMatrix(X, i);
		y = getM(Y, i, 0);
		transpose(x, x_T);							
		matrixMultiply(UV,x_T, UVX);				
		y_pred = trace(UVX);
		obj = obj + pow((y - y_pred),2);
	}

	freeMatrix(UV);
	freeMatrix(x_T);
	freeMatrix(UVX);

	obj = obj/(2 * (double) N);
	return obj;
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

// Train a Matrix Regrresion Model with data from R
void trainMatrixRegression(double *p_X_train, double *p_Y_train, int *p_N, int *p_m, int *p_n, int *p_k, double *p_U, double *p_V)
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

	matrix *U = newMatrix(m, k);													
	matrix *V = newMatrix(n, k);
	matrix *vec_U = vec(U);
	matrix *vec_V = vec(V); 

	matrix *x;
	double y, obj_i;

	matrix *x_T = newMatrix(n, m);
	matrix *XV = newMatrix(m, k);
	matrix *XV_vec_T = newMatrix(1, mk);
	matrix *XU = newMatrix(n, k);
	matrix *XU_vec_T = newMatrix(1, nk);

	matrix *sum_term_U1 = newMatrix(mk, mk);
	matrix *sum_term_U2 = newMatrix(mk, 1);

 	matrix *sum_term_V1 = newMatrix(nk, nk);
    matrix *sum_term_V2 = newMatrix(nk, 1);

    matrix *temp_term_U = newMatrix(mk,mk);
 	matrix *temp_term_V = newMatrix(nk,nk);

	matrix *XV_vec, *XU_vec;																// Mostly temporary matrices used later.
				
	randomizeMatrix(U);																		// Elements of U sampled from standard normal.
	randomizeMatrix(V);																		// Elements of V sampled from standard normal.

	if(DEBUG)
	{
		ones(m,k,U);
		ones(n,k,V);
	}

	int N_train_lambda = N_train * lambda;
	matrix* const_term_trainU = newMatrix(mk, mk);
	matrix* const_term_trainV = newMatrix(nk, nk);

	eye(mk,const_term_trainU);
	eye(nk,const_term_trainV);

	scalarMultiplyInPlace(N_train_lambda, const_term_trainU);				
	scalarMultiplyInPlace(N_train_lambda, const_term_trainV);

	obj_i = leastSquaredLoss(U, V, X_train, Y_train);
	obj_i = obj_i + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));					// Calculated Iniitial Objective
	printf("Training: Initial Objective = %f\n",obj_i);

	// Alternating Minimization Starts
	while(t<=T_MAX)
	 {
		printf("T = %d\n",t);

		// Compute U
		 zeros(sum_term_U1);
		 zeros(sum_term_U2);

		for(i=0; i<N_train; i++)
		{
			x = getMatrix(X_train,i); 														
			y = getM(Y_train, i, 0);

			matrixMultiply(x,V, XV);														
			XV_vec = vec(XV);																
			transpose(XV_vec, XV_vec_T); 												
			matrixMultiply(XV_vec,XV_vec_T, temp_term_U);								
		
			scalarMultiplyInPlace(y,XV_vec);											

			matrixAddInPlace(sum_term_U1,temp_term_U);	
			matrixAddInPlace(sum_term_U2,XV_vec);	
       }

       matrixAddInPlace(sum_term_U1, const_term_trainU);		
       inverse(sum_term_U1);							
       matrixMultiply(sum_term_U1, sum_term_U2, vec_U);		
       U = unvec(vec_U, m, k);

   		// Compute V
       zeros(sum_term_V1);
       zeros(sum_term_V2);

       for(i=0; i<N_train; i++)
       {
			x = getMatrix(X_train,i); 							
			y = getM(Y_train, i, 0);

			transpose(x,x_T);									
			matrixMultiply(x_T,U, XU);							
			XU_vec = vec(XU);										
			transpose(XU_vec, XU_vec_T);					

			matrixMultiply(XU_vec,XU_vec_T, temp_term_V);		
			scalarMultiplyInPlace(y,XU_vec);				
			matrixAddInPlace(sum_term_V1, temp_term_V);					
			matrixAddInPlace(sum_term_V2, XU_vec);					
		}

		matrixAddInPlace(sum_term_V1, const_term_trainV);				
		inverse(sum_term_V1);									
		matrixMultiply(sum_term_V1, sum_term_V2, vec_V);			
		V = unvec(vec_V, n, k);

		if(DEBUG)
		{
			printf("U = %f\n",frobNorm(U)); 
			printf("V = %f\n",frobNorm(V)); 
		}
      	// Calculate objective/Validate.
		if(t % VAL_FREQ == 0)
		{
			// Calculate Objective on Training set.
			obj = leastSquaredLoss(U, V, X_train, Y_train);
			obj = obj + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));
			printf("Training: Objective = %f\n",obj);

			// Calculate Objective on Validation set.
			if(VALIDATE)
			{
				obj = leastSquaredLoss(U, V, X_val, Y_val);
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

	freeMatrix(const_term_trainU);
	freeMatrix(const_term_trainV);

	freeMatrix(x_T);
	freeMatrix(XV);
	freeMatrix(XV_vec_T);
	freeMatrix(temp_term_U);

    freeMatrix(sum_term_U1);
    freeMatrix(sum_term_U2);

    freeMatrix(XU);
	freeMatrix(XU_vec_T);
	freeMatrix(temp_term_V);

	freeMatrix(sum_term_V1);
    freeMatrix(sum_term_V2);

	printf("Training: Initial Objective = %f\n",obj_i);

	if(VALIDATE)
	{
		// Calculate Objective on Test set.
		obj = leastSquaredLoss(U, V, X_test, Y_test);
		obj = obj + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2));
		printf("Testing: Objective = %f\n",obj);
	}
	freeMatrix(U);
	freeMatrix(V);
}

// Predict on Unknown data with trained model and data from R
void predictMatrixRegression(double *p_X_test, double *p_Y_test, int *p_N, int *p_m, int *p_n, int *p_k, double *p_U, double *p_V, int *p_Y_present, double *p_Y_predict, double *accuracy)
{
	int i;
	int k = *p_k;																		// Desired low rank
	int N = *p_N;																		// Number of Samples	
	int m = *p_m;																		// Number of rows
	int n = *p_n;																		// Number of columns
	int Y_present = *p_Y_present;														// Are true responses present?

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
		setM(Y_predict,i,0,trace(UVx));
	}

	// Copy into R array
	for(i=0;i<N;i++)
		p_Y_predict[i] = getM(Y_predict,i,0);

	// If true Y present, report Accuracy
	if (Y_present == 1)
	{
		matrixSubtract(Y_test, Y_predict, Y_minus_Y_predict);
		*accuracy = frobNorm(Y_minus_Y_predict)/(double)N;
	}

	freeMatrix(V_T);
	freeMatrix(UV);
	freeMatrix(x_T);
	freeMatrix(UVx);
	freeMatrix(Y_predict);
	freeMatrix(Y_minus_Y_predict);
}






