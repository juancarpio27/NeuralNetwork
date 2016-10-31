#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define PIXELS 3073
#define HIDDEN_LAYER_1 2000
#define HIDDEN_LAYER_2 450
#define OUTPUT_LAYER 10
#define LEARNING_RATE 0.001

#define ELEMENTS 1000

void print_vector(double *f, int N){
	for (int i =0; i < N; ++i)
		printf("%f\n",f[i]);
}

void get_layer(double *input, double *matrix, double *result,int input_size, int hidden_size){
	for (int i = 0; i<hidden_size; ++i){
		result[i] = 0.0;
		for (int j = 0; j < input_size; ++j){
			result[i] += input[j] * matrix[j*hidden_size+i];
		}
	}
}

void sigmoid(double *f, int N){
	for (int i =0; i < N; ++i)
		f[i] = 1.0 / (1.0 + exp(-f[i]));
}

void process_input(unsigned char *c, double *f, int N){
	for (int i =0; i < N; ++i){
		f[i] = (c[i]-1)/254.0;
	}
		
}

int max_index(double *f, int N){

	int max_index = 0;
	for (int i = 1; i < N; ++ i){
		if (f[i] > f[max_index]){
			max_index = i;
		}
	}
	return max_index;

}

double error(double *f, int output, int N) {
	double *output_array = (double *) malloc(N * sizeof(double));
	for (int i = 0; i < N; ++i)
		output_array[i] = 0.0;
	output_array[output] = 1.0;

	double error = 0.0;
	for (int i = 0; i < N; ++i)
		error += (output_array[i] - f[i]) * (output_array[i] - f[i]);
	return 0.5*error;
}

void error_output(double *f, int output, int N, double *error_array){
	double *output_array = (double *) malloc(N * sizeof(double));
	for (int i = 0; i < N; ++i)
		output_array[i] = 0.0;
	output_array[output] = 1.0;

	for (int i = 0; i < N; ++i){
		error_array[i] = (output_array[i] - f[i])*f[i]*(1-f[i]);
	}
}

void error_hidden_layer(double *f, double* error_array, double *next_layer_error, int layer_size, int next_size, double *transition_matrix){

	for (int i = 0; i < layer_size; ++i){
		double product = 0.0;
		for (int j = 0; j < next_size; ++j){
			product += transition_matrix[i*next_size+j]*next_layer_error[j];
		}
		error_array[i] = f[i]*(1-f[i])*product;
	}

}

void calculate_error(double *error_vector, double* gradient, double *layer, int N, int M){

	for (int i =0; i<N; ++i){
		for (int j = 0; j<M; ++j){
			gradient[i*M+j] = LEARNING_RATE * error_vector[j] * layer[i];
		}
	}

}

void sum_matrix(double *transition, double *gradient, int N, int M){
	for (int i = 0; i < N; ++i){
		for (int j = 0; j < M; ++j){
			transition[i*M+j] = transition[i*M+j] + gradient[i*M+j];
		}
	}
}

void read_file(char *file, unsigned char* buffer,double *transition_matrix_1,double *transition_matrix_2, double *transition_matrix_3, int elements){
	FILE *f;
	f = fopen(file, "rb");
	unsigned char expected_output;
	int i = 0;

	float total_time = 0.0;

	while(1 == fread(buffer,PIXELS,1,f) && i < elements){

		float tiempo1;
		cudaEvent_t inicio1, fin1;

		cudaEventCreate(&inicio1); // Se inicializan
   		cudaEventCreate(&fin1);
   		cudaEventRecord( inicio1, 0 );

		expected_output = buffer[0];
		buffer[0] = 1;
		double *hidden_layer_1 = (double*)malloc((HIDDEN_LAYER_1+1)*sizeof(double));
		double *hidden_layer_2 = (double*)malloc((HIDDEN_LAYER_2+1)*sizeof(double));
		double *output_layer = (double*)malloc(OUTPUT_LAYER*sizeof(double));

		double *input = (double*)malloc(PIXELS*sizeof(double));
		process_input(buffer,input,PIXELS);

		//Se ejecutan las neuronas
		get_layer(input,transition_matrix_1,hidden_layer_1,PIXELS,HIDDEN_LAYER_1);
		sigmoid(hidden_layer_1,HIDDEN_LAYER_1+1);
		hidden_layer_1[HIDDEN_LAYER_1] = 1;

		get_layer(hidden_layer_1,transition_matrix_2,hidden_layer_2,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);
		sigmoid(hidden_layer_2,HIDDEN_LAYER_2+1);
		hidden_layer_2[HIDDEN_LAYER_2] = 1;

		get_layer(hidden_layer_2,transition_matrix_3,output_layer,HIDDEN_LAYER_2+1,OUTPUT_LAYER);
		sigmoid(output_layer,OUTPUT_LAYER);

		//Error transicion de la capa de salida a la anterior
		double *errors_array = (double*)malloc(OUTPUT_LAYER* sizeof(double));
		error_output(output_layer,expected_output,OUTPUT_LAYER,errors_array);

		double *transition_matrix_3_gradient = (double*)malloc((HIDDEN_LAYER_2+1)*OUTPUT_LAYER*sizeof(double));
		calculate_error(errors_array,transition_matrix_3_gradient,hidden_layer_2,HIDDEN_LAYER_2+1,OUTPUT_LAYER);

		double *hidden_layer_array_2 = (double*)malloc((HIDDEN_LAYER_2+1)* sizeof(double));
		error_hidden_layer(hidden_layer_2,hidden_layer_array_2,errors_array,OUTPUT_LAYER,HIDDEN_LAYER_2+1,transition_matrix_3);
		double *transition_matrix_2_gradient = (double*)malloc((HIDDEN_LAYER_1+1)*HIDDEN_LAYER_2*sizeof(double));
		calculate_error(hidden_layer_array_2,transition_matrix_2_gradient,hidden_layer_1,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);

		double *hidden_layer_array_1 = (double*)malloc((HIDDEN_LAYER_1+1)* sizeof(double));
		error_hidden_layer(hidden_layer_1,hidden_layer_array_1,hidden_layer_array_2,HIDDEN_LAYER_2,HIDDEN_LAYER_1+1,transition_matrix_2);
		double *transition_matrix_1_gradient = (double*)malloc(PIXELS*HIDDEN_LAYER_1*sizeof(double));
		calculate_error(hidden_layer_array_1,transition_matrix_1_gradient,input,PIXELS,HIDDEN_LAYER_1);

		sum_matrix(transition_matrix_1,transition_matrix_1_gradient,PIXELS,HIDDEN_LAYER_1);
		sum_matrix(transition_matrix_2,transition_matrix_2_gradient,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);
		sum_matrix(transition_matrix_3,transition_matrix_3_gradient,HIDDEN_LAYER_2+1,OUTPUT_LAYER);

		//Siguiente elemento 
		++i;

		free(hidden_layer_1);
		free(hidden_layer_2);
		free(output_layer);
		free(input);
		free(errors_array);

		cudaEventRecord( fin1, 0); // Se toma el tiempo final.
   		cudaEventSynchronize( fin1 ); // Se sincroniza
   		cudaEventElapsedTime( &tiempo1, inicio1, fin1 );

   		total_time += tiempo1;
		
	}

	total_time /= elements;
	printf ("Tiempo promedio por clasificacion: %f\n", total_time);
}

void init_layer(double *matrix, int N, int M){
	for (int i =0; i < N; ++ i){
		for (int j = 0; j < M; ++j){
		    int sign = rand() % 2;
		    if (sign == 0)
		    	matrix[i*M+j] = (rand() % 1000000) / 1000000.0;
			else
				matrix[i*M+j] = - ((rand() % 1000000) / 1000000.0);
		}
	}
		
}

void print_layer(double *matrix, int N, int M){
	for (int i =0; i < N; ++ i){
		for (int j = 0; j < M; ++j)
			printf("%f ",matrix[i*M+j]);
		printf("\n");
	}		
}

int main(int argc, char *argv[]){

	srand(time(NULL));

	 if ( argc != 2 ) {
        /* We print argv[0] assuming it is the program name */
        printf( "Error se debe ejecutar: %s <N>\n", argv[0] );
        exit(0);
    }

    int elements = atoi(argv[1]);

    printf("Se va a entrenar con %d elementos\n",elements);
	
	double *transition_matrix_1 = (double*)malloc(PIXELS*HIDDEN_LAYER_1*sizeof(double));
	double *transition_matrix_2 = (double*)malloc((HIDDEN_LAYER_1+1)*HIDDEN_LAYER_2*sizeof(double));
	double *transition_matrix_3 = (double*)malloc((HIDDEN_LAYER_2+1)*OUTPUT_LAYER*sizeof(double));

	init_layer(transition_matrix_1,PIXELS,HIDDEN_LAYER_1);
	init_layer(transition_matrix_2,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);
	init_layer(transition_matrix_3,HIDDEN_LAYER_2+1,OUTPUT_LAYER);

	float tiempo1;
	cudaEvent_t inicio1, fin1;

	unsigned char *buffer = (unsigned char*)malloc(PIXELS*sizeof(unsigned char));

	cudaEventCreate(&inicio1); // Se inicializan
   	cudaEventCreate(&fin1);
   	cudaEventRecord( inicio1, 0 );

	read_file("data_batch_1.bin",buffer,transition_matrix_1,transition_matrix_2,transition_matrix_3,elements);

	cudaEventRecord( fin1, 0); // Se toma el tiempo final.
   	cudaEventSynchronize( fin1 ); // Se sincroniza
   	cudaEventElapsedTime( &tiempo1, inicio1, fin1 );

   	printf("Tiempo total del programa: %f ms\n", tiempo1);

}

