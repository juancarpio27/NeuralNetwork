#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//Number of elements of the inpu layers, that correspond to the number of pixels of a picture
#define PIXELS 3073
//Number of elements of the first hidden layer
#define HIDDEN_LAYER_1 2000
//Number of elements of the second hidden layer
#define HIDDEN_LAYER_2 450
//Number of elements of the output layer
#define OUTPUT_LAYER 10
//Learning rate of the algorithm
#define LEARNING_RATE 0.01
//Numbers of elements to use for training
#define ELEMENTS 1000
//Blocks 
#define BLOCKS 32

/* 
 * Function that given a vector and its size, print it
 * In: 
 * f: vector of doubles to be printed
 * N: size of the vector
 */
void print_vector(double *f, int N){
	//Move in all vector to print each value
	for (int i =0; i < N; ++i)
		printf("%f\n",f[i]);
}

__global__ void process_input(unsigned char *c, double *f, int N){

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N){
		f[i] = (c[i]-1)/254.0;
	}
}

__global__ void get_layer(double *input, double *matrix, double *result,int input_size, int hidden_size){

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x < hidden_size && y < input_size)
		result[x] += input[y]*matrix[y*hidden_size+x];
}

__global__ void sigmoid(double *f, int N){

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x < N)
		f[i] = 1.0 / (1.0 + exp(-f[i]));		
}

__global__ void assign_constant(double *f){
	f[0] = 1;
}

__global__ void calculate_error(double *error_vector, double* gradient, double *layer, int N, int M){

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x < N && y < M)
		gradient[x*M+y] = LEARNING_RATE * error_vector[y] * layer[x];

}

__global__ void error_hidden_layer(double *f, double* error_array, double *next_layer_error, int layer_size, int next_size, double *transition_matrix){

	//Calculate error of every neuron in a hidden layer
	//The error in a hidden layer is defined as Si = oi * (1 - oi) * SUM(Wij * Sj) where Sj is the error from next
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x < layer_size){
		double product = 0.0;
		for (int j = 0; j < next_size; ++j){
			product += transition_matrix[x*next_size+y]*next_layer_error[y];
		}
		error_array[x] = f[x]*(1-f[x])*product;
	}
	
}

__global__ void sum_matrix(double *transition, double *gradient, int N, int M){

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if  (x < N && y < M){
		transition[x*M+y] = transition[x*M+y] + gradient[x*M+y];
	}

}

void read_file(char *file, unsigned char* buffer,double *transition_matrix_1,double *transition_matrix_2, double *transition_matrix_3, int elements){
	//Read the file that is in binary mode
	FILE *f;
	f = fopen(file, "rb");
	//Variable for the expected output
	unsigned char expected_output;
	int i = 0;
	//Init the total time to get the average of every classification
	float total_time = 0.0;

	//While there still elements on the file, and i is less than elements number cycloe
	//Read PIXELS elements, because every pixel is represented by a byte, is enough to tell the reader to read exactly PIXELS bytes
	while(1 == fread(buffer,PIXELS,1,f) && i < elements){

		expected_output = buffer[0];
		buffer[0] = 1;

		double *hidden_layer_1;
		double *hidden_layer_2;
		double *output_layer;

		cudaMalloc( (void**)&hidden_layer_1, (HIDDEN_LAYER_1+1)*sizeof(double) );
		cudaMalloc( (void**)&hidden_layer_2, (HIDDEN_LAYER_2+1)*sizeof(double) );
		cudaMalloc( (void**)&output_layer, OUTPUT_LAYER*sizeof(double) );

		unsigned char *dev_input; 
		cudaMalloc( (void**)&dev_input, PIXELS*sizeof(unsigned char) );
		cudaMemcpy( dev_input, buffer, PIXELS*sizeof(unsigned char), cudaMemcpyHostToDevice );

		double *dev_input_normalized;
		cudaMalloc( (void**)&dev_input_normalized, PIXELS*sizeof(double) );

		process_input<<<BLOCKS,PIXELS/BLOCKS+1>>>(dev_input,dev_input_normalized,PIXELS);

		dim3 threads_1(HIDDEN_LAYER_1/BLOCKS+1,PIXELS/BLOCKS+1);
		dim3 blocks_1(BLOCKS,BLOCKS);
		get_layer<<<blocks_1,threads_1>>>(dev_input_normalized,transition_matrix_1,hidden_layer_1,PIXELS,HIDDEN_LAYER_1);
		sigmoid<<<BLOCKS,HIDDEN_LAYER_1/BLOCKS+1>>>(hidden_layer_1,HIDDEN_LAYER_1+1);
		assign_constant<<<1,1>>>(hidden_layer_1);

		dim3 threads_2(HIDDEN_LAYER_2/BLOCKS+1,HIDDEN_LAYER_1/BLOCKS+1);
		dim3 blocks_2(BLOCKS,BLOCKS);
		get_layer<<<blocks_1,threads_1>>>(hidden_layer_1,transition_matrix_2,hidden_layer_2,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);
		sigmoid<<<BLOCKS,HIDDEN_LAYER_2/BLOCKS+1>>>(hidden_layer_2,HIDDEN_LAYER_2+1);

		dim3 threads_3(OUTPUT_LAYER/BLOCKS+1,HIDDEN_LAYER_2/BLOCKS+1);
		dim3 blocks_3(BLOCKS,BLOCKS);
		get_layer<<<blocks_1,threads_1>>>(hidden_layer_2,transition_matrix_3,output_layer,HIDDEN_LAYER_2+1,OUTPUT_LAYER);
		sigmoid<<<BLOCKS,OUTPUT_LAYER/BLOCKS+1>>>(output_layer,OUTPUT_LAYER);

		double *errors_array;
		cudaMalloc( (void**)&errors_array, OUTPUT_LAYER*sizeof(double) );
		double *transition_matrix_3_gradient;
		cudaMalloc( (void**)&transition_matrix_3_gradient, (HIDDEN_LAYER_2+1)*OUTPUT_LAYER*sizeof(double) );

		dim3 e_threads_1((HIDDEN_LAYER_2+1)/BLOCKS+1,OUTPUT_LAYER/BLOCKS+1);
		dim3 e_blocks_1(BLOCKS,BLOCKS);
		calculate_error<<<e_blocks_1,e_threads_1>>>(errors_array,transition_matrix_3_gradient,hidden_layer_2,HIDDEN_LAYER_2+1,OUTPUT_LAYER);

		double *hidden_layer_array_2;
		cudaMalloc( (void**)&hidden_layer_array_2, (HIDDEN_LAYER_2+1)*sizeof(double) );
		error_hidden_layer<<<BLOCKS,OUTPUT_LAYER/BLOCKS+1>>>(hidden_layer_2,hidden_layer_array_2,errors_array,OUTPUT_LAYER,HIDDEN_LAYER_2+1,transition_matrix_3);
		double transition_matrix_2_gradient;
		cudaMalloc( (void**)&transition_matrix_2_gradient, (HIDDEN_LAYER_1+1)*HIDDEN_LAYER_2*sizeof(double) );

		dim3 e_threads_2((HIDDEN_LAYER_1+1)/BLOCKS+1,HIDDEN_LAYER_2/BLOCKS+1);
		dim3 e_blocks_2(BLOCKS,BLOCKS);
		calculate_error<<<e_blocks_2,e_threads_2>>>(hidden_layer_array_2,transition_matrix_2_gradient,hidden_layer_1,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);

		double *hidden_layer_array_1;
		cudaMalloc( (void**)&hidden_layer_array_1, (HIDDEN_LAYER_1+1)*sizeof(double) );
		error_hidden_layer<<<BLOCKS,(HIDDEN_LAYER_2+1)/BLOCKS+1>>>(hidden_layer_1,hidden_layer_array_1,hidden_layer_array_2,HIDDEN_LAYER_2,HIDDEN_LAYER_1+1,transition_matrix_2);
		double transition_matrix_1_gradient;
		cudaMalloc( (void**)&transition_matrix_1_gradient, PIXELS*HIDDEN_LAYER_1*sizeof(double) );

		dim3 e_threads_3(PIXELS/BLOCKS+1,HIDDEN_LAYER_1/BLOCKS+1);
		dim3 e_blocks_3(BLOCKS,BLOCKS);
		calculate_error<<<e_blocks_3,e_threads_3>>>(hidden_layer_array_1,transition_matrix_1_gradient,input,PIXELS,HIDDEN_LAYER_1);

		dim3 s_threads_1(PIXELS/BLOCKS+1,HIDDEN_LAYER_1/BLOCKS+1);
		dim3 s_blocks_1(BLOCKS,BLOCKS);
		sum_matrix<<<s_blocks_1,s_threads_1>>>(transition_matrix_1,transition_matrix_1_gradient,PIXELS,HIDDEN_LAYER_1);
		dim3 s_threads_2((HIDDEN_LAYER_1+1)/BLOCKS+1,HIDDEN_LAYER_2/BLOCKS+1);
		dim3 s_blocks_2(BLOCKS,BLOCKS);
		sum_matrix<<<s_blocks_2,s_threads_2>>>(transition_matrix_2,transition_matrix_2_gradient,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);
		dim3 s_threads_3((HIDDEN_LAYER_2+1)/BLOCKS+1,OUTPUT_LAYER/BLOCKS+1);
		dim3 s_blocks_3(BLOCKS,BLOCKS);
		sum_matrix<<<s_blocks_3,s_threads_3>>>(transition_matrix_3,transition_matrix_3_gradient,HIDDEN_LAYER_2+1,OUTPUT_LAYER);

		++i;

		cudaFree( hidden_layer_1 );
		cudaFree( hidden_layer_2 );
		cudaFree( output_layer );
		cudaFree( errors_array );
		cudaFree( hidden_layer_array_1 );
		cudaFree( hidden_layer_array_2 );
		cudaFree( errors_array );
		cudaFree( transition_matrix_3_gradient );
		cudaFree( transition_matrix_2_gradient );
		cudaFree( transition_matrix_1_gradient );
		cudaFree( dev_input_normalized);
		cudaFree( dev_input );

	}
}

__global__ void init_layer(double *matrix, int N, int M){

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < N && y < M){
		int sign = rand() % 2;
		//Random number between 0 and 1
		if (sign == 0)
		    matrix[i*M+j] = (rand() % 1000000) / 1000000.0;
		else
			matrix[i*M+j] = - ((rand() % 1000000) / 1000000.0);
	}

}

int main(int argc, char *argv[]){

	//Init the random
	srand(time(NULL));
	//Review if the arguments
	if ( argc != 2 ) {
        /* We print argv[0] assuming it is the program name */
        printf( "Error se debe ejecutar: %s <N>\n", argv[0] );
        exit(0);
    }
    //Transform the argv to int
    int elements = atoi(argv[1]);
    printf("Se va a entrenar con %d elementos\n",elements);

    //Initialize transition matrix. They will always reside in the GPU
    double *dev_transition_1, *dev_transition_2, *dev_transition_3;
	cudaMalloc( (void**)&dev_transition_1, PIXELS*HIDDEN_LAYER_1*sizeof(double) );
    cudaMalloc( (void**)&dev_transition_2, (HIDDEN_LAYER_1+1)*HIDDEN_LAYER_2*sizeof(double));
    cudaMalloc( (void**)&dev_transition_3, (HIDDEN_LAYER_2+1)*OUTPUT_LAYER*sizeof(double));

    dim3 threads_1(PIXELS/BLOCKS+1, HIDDEN_LAYER_1/BLOCKS+1);
    dim3 blocks_1(BLOCKS, BLOCKS);
    init_layer<<<blocks_1,threads_1>>>(dev_transition_1,PIXELS,HIDDEN_LAYER_1);

    dim3 threads_2(HIDDEN_LAYER_1/BLOCKS+1, HIDDEN_LAYER_1/BLOCKS+1);
    dim3 blocks_2(BLOCKS, BLOCKS);
    init_layer<<<blocks_2,threads_2>>>(dev_transition_2,HIDDEN_LAYER_1,HIDDEN_LAYER_2);

    dim3 threads_3(HIDDEN_LAYER_2/BLOCKS+1, OUTPUT_LAYER/BLOCKS+1);
    dim3 blocks_3(BLOCKS, BLOCKS);
    init_layer<<<blocks_3,threads_3>>>(dev_transition_3,HIDDEN_LAYER_2,OUTPUT_LAYER);

    float tiempo1;
	cudaEvent_t inicio1, fin1;

    unsigned char *buffer = (unsigned char*)malloc(PIXELS*sizeof(unsigned char));

    cudaEventCreate(&inicio1); // Se inicializan
   	cudaEventCreate(&fin1);
   	cudaEventRecord( inicio1, 0 );

    read_file("data_batch_1.bin",buffer,dev_transition_1,dev_transition_2,dev_transition_3,elements);

    cudaEventRecord( fin1, 0); // Se toma el tiempo final.
   	cudaEventSynchronize( fin1 ); // Se sincroniza
   	cudaEventElapsedTime( &tiempo1, inicio1, fin1 );

   	//Print the time
   	printf("Tiempo total del programa: %f ms\n", tiempo1);

}