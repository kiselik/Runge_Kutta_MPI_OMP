#include <stdio.h> 
#include <stdlib.h>
#include <limits>
#include <math.h>
#include "mpi.h" 
#include <ctime>

void DataInitialization(float *Matrix, float *Vector, int N) 
{  
	for(int i=0;i<N;i++) 
	{ 
		Vector[i] = 0.082f; 
	} 

	float tempSum = 0;
	for (int i = 0; i < N; i++)
	{
		int seed=3+i;
		srand(seed);
		for (int j = 0; j < N ; j++)
		{
			Matrix[i*N + j] = ((float)(rand() % 10 - 5)) / 10;
		}
	}
} 

void PrintVector(float *Vector, int N)
{
	for(int i=0;i<N;i++) 
	{ 
		printf(" %.3f",Vector[i]); 
	} 
}
void PrintVector (int *Vector, int N)
{
	for(int i=0;i<N;i++) 
	{ 
		printf(" %.3f",Vector[i]); 
	} 
}
void PrintMatrix(float *Matrix, int N)
{
	for(int i=0;i<N;i++) 
	{ 
		for(int j=0;j<N;j++)
		{
			printf(" %.3f",Matrix[i*N+j]); 
		}
		printf("\n");
	} 
}


float* vectXmatr(float *Matrix,float step, float *Vector, int row_count,int count)
{
	//size- количество элементов в строке
	int size=count/row_count;
	float* sResult = new float[row_count];
	for(int i=0;i<row_count;i++) 
	{ 
		float tempSum = 0;
		for(int j=0;j<size;j++)
		{
			tempSum += step * Matrix[i*size+j] * Vector[i];
		}
		sResult[i] = tempSum;
	} 
	return sResult;
	//delete []sResult;
}

float* vectF(float x,float step, int N)
{
	float* sResult = new float[N];
	for(int i=0;i<N;i++)
	{
		sResult[i] = step*x;
	}
	return sResult;
}

float* sumVect(float *V1, float *V2, int N)
{
	float* sResult = new float[N];
	for(int i=0;i<N;i++)
	{
		sResult[i] = V1[i] + V2[i];
	}
	return sResult;
}

float* get_dy(float *V1, float *V2, float *V3, float *V4, int N)
{
	float* sResult = new float[N];
	for(int i=0;i<N;i++)
	{
		sResult[i] = (V1[i] + 2*V2[i] + 2*V3[i] + V4[i])/6;
	}
	return sResult;
}

void fill_array(int*a,int size)
{
	for(int i=0;i<size;i++)
		a[i]=0;
}
void fill_array(float*a,int size)
{
	for(int i=0;i<size;i++)
		a[i]=0;
}

float*getKoeff( float coeff,float *Vect, int N)
{
	float* sResult = new float[N];
	for(int i=0;i<N;i++)
	{
		sResult[i] = coeff*Vect[i];
	}
	return sResult;
}

bool areEqual(float* vectA, float* vectB, int N)
{
	float r = std::numeric_limits<float>::epsilon();
	for(int i=0; i<N; i++)
	{
		if(abs(vectA[i] - vectB[i]) > r)
		{
			return false;
		}
	}
	return true;
}

int main(int argc, char* argv[]) 
{
	int rank, count_process;
	int size = atoi(argv[1]);
	float start, finish, h, x;
	int index;
	float *y_start = new float[size];
	float *y_result = new float[size];
	float *y_result2=new float[size];
	float *A = new float[size*size];
	double ParallelTimeStart;

	y_result=y_start;

	int *recvCounts=NULL;
	int * recvdispls=NULL;
	float *tmp=NULL;
	int count_Row_Scatterv;
	float *matrix=NULL;
	float *vector=NULL;

	int *send_Counts_Matrix=NULL;
	int *displs_Matrix=NULL;

	int *send_Counts_Vector=NULL;
	int *displs_Vector=NULL;

	MPI_Init(&argc,&argv); 
	MPI_Comm_size(MPI_COMM_WORLD,&count_process); 
	MPI_Comm_rank(MPI_COMM_WORLD,&rank); 

	if ( rank == 0 ) 
	{ 
		DataInitialization(A, y_start, size);
		start=-2.5;
		finish=2.5;
		h=0.5;
		x=start;
		int count=size*size;
		float *k1 = new float[size];
		float *k2 = new float[size];
		float *k3 = new float[size];
		float *k4 = new float[size];
		float *dy = new float[size];
		double SerialTimeStart = MPI_Wtime(); 

		for (index=0; x<finish; index++)
		{         
			x = start + index*h;
			k1 = sumVect(vectXmatr(A,h,y_result,size,count), vectF(x,h,size),size);
			k2 = sumVect(vectXmatr(A,h,sumVect(y_result,getKoeff(0.5, k1,size),size),size,count), vectF(x+h/2,h,size),size);
			k3 = sumVect(vectXmatr(A,h,sumVect(y_result,getKoeff(0.5, k2, size),size),size,count), vectF(h,x+h/2,size), size);
			k4 = sumVect(vectXmatr(A,h,sumVect(y_result, k3, size),size,count), vectF(h,x+h,size), size);
			dy = get_dy(k1,k2,k3,k4,size);
			y_result = sumVect(y_result, dy,size);
		}
		double SerialTime = MPI_Wtime() - SerialTimeStart; 
		printf("\nTime (serial) = %.6f",SerialTime); 
		printf("\n");

		x=start;
		delete [] k1;
		delete [] k2;
		delete [] k3;
		delete [] k4;
		delete [] dy;
		ParallelTimeStart=MPI_Wtime();
	}
	//parallel v
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&start, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&finish, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&h, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(y_start, size, MPI_INT, 0, MPI_COMM_WORLD);



	if (count_process>1){
		count_Row_Scatterv= size / (count_process-1);
	}
	else{
		count_Row_Scatterv= size *size;
	}
	{
		send_Counts_Matrix = new int[count_process]; // массив, в котором лежит количество строк, уходящих на каждый процесс 
		displs_Matrix = new int[count_process];//массив смещений, относительно начала буфера перессылки

		send_Counts_Vector = new int[count_process]; //аналогично
		displs_Vector = new int[count_process];

		recvCounts = new int[count_process]; //аналогично тому,что вверху,только для Gatherv
		recvdispls = new int[count_process];

		for (int i = 1; i < count_process-1; i++){

			send_Counts_Vector[i]=recvCounts[i] = count_Row_Scatterv;// count of rows which edit every process
			send_Counts_Matrix[i] = recvCounts[i]*size; // count of elements 
		}
		send_Counts_Matrix[0]=send_Counts_Vector[0]=recvCounts[0]=0;// Нулевой ранг не учавствует в подсчете, поэтому ему отсылаем ноль элементов
		send_Counts_Matrix[count_process-1]=  (size- (count_process-2)*count_Row_Scatterv)*size;// количество элементов, которые достанутся последнему процессу(для матриц неровно делящихся на количество процессов
		send_Counts_Vector[count_process-1]=recvCounts[count_process-1]= size- (count_process-2)*count_Row_Scatterv;// количество строк, которые достанутся последнему процессу(для матриц неровно делящихся на количество процессов

		fill_array(displs_Matrix,count_process);// занулили
		fill_array(recvdispls,count_process);// занулили
		fill_array(displs_Vector,count_process);

		for (int i = 2; i < count_process; i++){

			displs_Matrix[i]+=send_Counts_Matrix[i-1]+displs_Matrix[i-1]; // в зависимости от процесса берем разное смещение, зависящее от предыдущих смещений
			recvdispls[i]+= recvCounts[i-1]+recvdispls[i-1];
			displs_Vector[i]+=recvCounts[i-1]+recvdispls[i-1];
		}


	}
	matrix = new float[send_Counts_Matrix[rank]];// буфер  для приема , длина которого зависит от номера процесса. т.е. сколькоэлементов отправляем каждому процессу
	vector=new float[send_Counts_Vector[rank]];

	MPI_Scatterv(A, send_Counts_Matrix, displs_Matrix, MPI_INT, matrix, send_Counts_Matrix[rank], MPI_INT, 0, MPI_COMM_WORLD);//+
	MPI_Scatterv(y_start, send_Counts_Vector, displs_Vector, MPI_INT, vector, send_Counts_Vector[rank], MPI_INT, 0, MPI_COMM_WORLD);//+

	if(rank!=0)
	{

		tmp=new float[recvCounts[rank]];
		int tmp_size=recvCounts[rank];
		float *k1 = new float[tmp_size];
		float *k2 = new float[tmp_size];
		float *k3 = new float[tmp_size];
		float *k4 = new float[tmp_size];
		float *dy = new float[tmp_size];

		fill_array(tmp,recvCounts[rank]);

		for (index=0; x<finish; index++)
		{         
			x = start + index*h;
			k1 = sumVect(vectXmatr(matrix,h,vector,send_Counts_Vector[rank],send_Counts_Matrix[rank]), vectF(x,h,send_Counts_Vector[rank]),send_Counts_Vector[rank]);
			k2 = sumVect(vectXmatr(matrix,h,sumVect(vector,getKoeff(0.5, k1,send_Counts_Vector[rank]),send_Counts_Vector[rank]),send_Counts_Vector[rank],send_Counts_Matrix[rank]), vectF(x+h/2,h,send_Counts_Vector[rank]),send_Counts_Vector[rank]);
			k3 = sumVect(vectXmatr(matrix,h,sumVect(vector,getKoeff(0.5, k2,send_Counts_Vector[rank]), send_Counts_Vector[rank]),send_Counts_Vector[rank],send_Counts_Matrix[rank]), vectF(h,x+h/2,send_Counts_Vector[rank]), send_Counts_Vector[rank]);
			k4 = sumVect(vectXmatr(matrix,h,sumVect(vector, k3, send_Counts_Vector[rank]),send_Counts_Vector[rank],send_Counts_Matrix[rank]), vectF(h,x+h,send_Counts_Vector[rank]), send_Counts_Vector[rank]);
			dy = get_dy(k1,k2,k3,k4,send_Counts_Vector[rank]);
			vector = sumVect(vector, dy,send_Counts_Vector[rank]);
		}

		delete [] k1;
		delete [] k2;
		delete [] k3;
		delete [] k4;
		delete [] dy;

		for (int j = 0; j < send_Counts_Vector[rank]; j++)
		{
			tmp[j]=vector[j];
		}
	}
	delete [] vector;
	delete [] matrix;
	MPI_Barrier(MPI_COMM_WORLD);
	fill_array(y_result2,size);

	// шлем результат из буфера tmp, размер которого=количеству обработанных строк, зависит от ранга, элементы типа float
	//шлем в буфер result2
	MPI_Gatherv(tmp,recvCounts[rank],MPI_FLOAT,y_result2,recvCounts,recvdispls,MPI_FLOAT, 0, MPI_COMM_WORLD);

	if(rank==0)
	{
		double ParallelTime = MPI_Wtime() - ParallelTimeStart; 
		printf("\nTime (parallel) = %.6f",ParallelTime);
		if(size<=10){
			printf("\nVector y0: \n");
			PrintVector(y_result, size);
			printf("\nVector y_result2: \n");
			PrintVector(y_result2, size);
		}
		if(areEqual(y_result, y_result2, size)) printf("\n\nResults are equal");
		else printf("\n\nResults are NOT equal!");
	}
	delete [] send_Counts_Matrix;
	delete [] displs_Matrix;
	delete [] send_Counts_Vector;
	delete [] displs_Vector;
	delete [] recvCounts;
	delete [] recvdispls;
	//delete [] y_start;
	delete [] y_result;
	delete [] y_result2;
	delete []A;

	MPI_Finalize(); 

	return 0;    
}
