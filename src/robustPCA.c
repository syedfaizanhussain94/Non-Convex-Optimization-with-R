# include <stdio.h>
# include "matrixAlgebra.h"

# define T_MAX 1000														// Maximum number of iterations of Alternating Minimization.
# define lambda 0.1														// Regularizer parameter for U and V.
# define mu 0.1															// Regularizer parameter for S.
# define ERROR_THRESHOLD 0.0001											// Error Threshold to quit Alternating Minimization if error is below a minimal threshold.
# define DEBUG 0



// Reads and copies Y from R array
void readRdata(double *p_Y, matrix *Y, int m, int n)
{
	reshapeM(Y, m, n);
	Y->mat = p_Y;
}

// Function to perform robust PCA on data from R
void robustPCA(double *p_Y, int *p_m, int *p_n, int *p_k, double *p_U, double *p_V, double *p_S)
{
	time_t tm;
	srand((unsigned) time(&tm));

	int k = *p_k;																		// Desired low rank
	int m = *p_m;																		// Number of rows
	int n = *p_n;																		// Number of columns

	int i,j,t;
	double b, obj, obj_i;
	t = 1;

	matrix *Y = emptyMatrix();

	readRdata(p_Y, Y, m, n);

	matrix *l_eye_k = newMatrix(k,k);
	eye(k,l_eye_k);
	scalarMultiplyInPlace(lambda, l_eye_k);

	matrix *U = newMatrix(m, k);
	matrix *V = newMatrix(n, k);
	matrix *S = newMatrix(m,n);									// Initially S is a zero matrix

	randomizeMatrix(U);											// Elements of U sampled from standard normal.
	randomizeMatrix(V);											// Elements of V sampled from standard normal.

	if(DEBUG)
	{
		ones(m,k,U);
		ones(n,k,V);
	}

	matrix *V_T = newMatrix(k,n);
	matrix *U_T = newMatrix(k,m);
	matrix *UV = newMatrix(m,n);
	matrix *Y_minus_S = newMatrix(m,n);
	matrix *Y_minus_S_T = newMatrix(n,m);
	matrix *Y_minus_UV_minus_S = newMatrix(m,n);

	matrix *U_term_temp1 = newMatrix(k,k);
	matrix *U_term_temp2 = newMatrix(n,k);

	matrix *V_term_temp1 = newMatrix(k,k);
	matrix *V_term_temp2 = newMatrix(m,k);

	matrix *UV_plus_S = newMatrix(m,n);

	transpose(V, V_T);
	matrixMultiply(U,V_T, UV);
	matrixAdd(UV,S,UV_plus_S);
	matrixSubtract(Y,UV_plus_S, Y_minus_UV_minus_S);
	obj_i = pow(frobNorm(Y_minus_UV_minus_S),2);
	obj_i = obj_i + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2)) + mu*norm(S,1);
	printf("Training: Initital Objective  = %f\n",obj_i);

	// Alternating minimization starts
	while(t <= T_MAX)
	{
		printf("T = %d\n",t);
		matrixSubtract(Y,S, Y_minus_S);								
		transpose(Y_minus_S, Y_minus_S_T);								

		// Calculate U
		transpose(V, V_T);												
		matrixMultiply(V_T,V, U_term_temp1) ;							
		scalarMultiplyInPlace(2, U_term_temp1);
		matrixAddInPlace(U_term_temp1,l_eye_k);
		inverse(U_term_temp1);
		matrixMultiply(V,U_term_temp1, U_term_temp2);					
		matrixMultiply(Y_minus_S,U_term_temp2, U);
		scalarMultiplyInPlace(2,U); 

		// Calculate V
		transpose(U, U_T);												
		matrixMultiply(U_T,U, V_term_temp1);							
		scalarMultiplyInPlace(2, V_term_temp1);
		matrixAddInPlace(V_term_temp1,l_eye_k);
		inverse(V_term_temp1);
		matrixMultiply(U,V_term_temp1, V_term_temp2);					
		matrixMultiply(Y_minus_S_T,V_term_temp2, V);		
		scalarMultiplyInPlace(2,V); 

		transpose(V, V_T);
		matrixMultiply(U,V_T, UV);

		// Calculate S
		for(i=0;i<m;i++)
		{
			for(j=0;j<n;j++)
			{
				b = getM(Y,i,j) - getM(UV,i,j);
				if(b > (mu/(double)2))
					setM(S,i,j,(b - mu/(double)2));
				else if(b < (-mu/2))
					setM(S,i,j,(b + mu/(double)2));
				else
					setM(S,i,j,0);
			}
		}

		// Calculate Objective
		matrixAdd(UV,S,UV_plus_S);
		matrixSubtract(Y,UV_plus_S, Y_minus_UV_minus_S);
		obj = pow(frobNorm(Y_minus_UV_minus_S),2);
		obj = obj + (lambda/2)*(pow(frobNorm(U),2) + pow(frobNorm(V),2)) + mu*norm(S,1);
		printf("Objective  = %f\n",obj);
		t++;
	}
	printf("Initial Objective = %f\n",obj_i);
	// Copy model arrays into R arrays
	for(i=0;i<m;i++)
	{
		for(j=0;j<k;j++)
			p_U[i + j*m] = getM(U,i,j);
	}

	for(i=0;i<n;i++)
	{
		for(j=0;j<k;j++)
			p_V[i + j*n] = getM(V,i,j);
	}

	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
			p_S[i + j*m] = getM(S,i,j);
	}

	freeMatrix(l_eye_k);
	freeMatrix(U_T);
	freeMatrix(V_T);
	freeMatrix(UV);
	freeMatrix(Y_minus_S);
	freeMatrix(Y_minus_S_T);
	freeMatrix(Y_minus_UV_minus_S);
	freeMatrix(U_term_temp1);
	freeMatrix(U_term_temp2);
	freeMatrix(V_term_temp1);
	freeMatrix(V_term_temp2);
	freeMatrix(UV_plus_S);

	freeMatrix(U);
	freeMatrix(V);
	freeMatrix(S);
}