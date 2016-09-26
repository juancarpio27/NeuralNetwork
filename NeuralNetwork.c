#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define PIXELS 3073

#define HIDDEN_LAYER_1 1500
#define HIDDEN_LAYER_2 450
#define OUTPUT_LAYER 10

void print_vector(double *f, int N){
	for (int i =0; i < N; ++i)
		printf("%f\n",f[i]);
}

void get_hidden_layer_1(unsigned char *input, double *matrix, double *result,int input_size, int hidden_size){
	for (int i = 0; i<hidden_size; ++i){
		result[i] = 0.0;
		for (int j = 0; j < input_size; ++j){
			result[i] += input[j] + matrix[j*hidden_size+i];
		}
	}
}

void get_layer(double *input, double *matrix, double *result,int input_size, int hidden_size){
	for (int i = 0; i<hidden_size; ++i){
		result[i] = 0.0;
		for (int j = 0; j < input_size; ++j){
			result[i] += input[j] + matrix[j*hidden_size+i];
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

void read_file(char *file, unsigned char* buffer,double *transition_matrix_1,double *transition_matrix_2, double *transition_matrix_3){
	FILE *f;
	f = fopen(file, "rb");
	unsigned char expected_output;
	int i = 0;
	while(1 == fread(buffer,PIXELS,1,f) && i < 1){

		expected_output = buffer[0];
		buffer[0] = 1;
		double *hidden_layer_1 = (double*)malloc(HIDDEN_LAYER_1*sizeof(double));
		double *hidden_layer_2 = (double*)malloc(HIDDEN_LAYER_2*sizeof(double));
		double *output_layer = (double*)malloc(OUTPUT_LAYER*sizeof(double));

		double *input = (double*)malloc(PIXELS*sizeof(double));
		process_input(buffer,input,PIXELS);

		//Se ejecutan las neuronas
		get_layer(input,transition_matrix_1,hidden_layer_1,PIXELS,HIDDEN_LAYER_1);
		sigmoid(hidden_layer_1,HIDDEN_LAYER_1);
		get_layer(hidden_layer_1,transition_matrix_2,hidden_layer_2,HIDDEN_LAYER_1,HIDDEN_LAYER_2);
		sigmoid(hidden_layer_2,HIDDEN_LAYER_2);
		get_layer(hidden_layer_2,transition_matrix_3,output_layer,HIDDEN_LAYER_2,OUTPUT_LAYER);
		sigmoid(output_layer,OUTPUT_LAYER);
		print_vector(output_layer,OUTPUT_LAYER);

		//Siguiente elemento 
		++i;
		
	}
}

void init_layer(double *matrix, int N, int M){
	for (int i =0; i < N; ++ i)
		for (int j = 0; j < M; ++j)
			matrix[i*M+j] = (rand() % 10000) / 10000.0;
}

void print_layer(double *matrix, int N, int M){
	for (int i =0; i < N; ++ i){
		for (int j = 0; j < M; ++j)
			printf("%f ",matrix[i*M+j]);
		printf("\n");
	}		
}



int main(){

	srand(time(NULL));
	
	double *transition_matrix_1 = (double*)malloc(PIXELS*HIDDEN_LAYER_1*sizeof(double)); 
	double *transition_matrix_2 = (double*)malloc(HIDDEN_LAYER_1*HIDDEN_LAYER_2*sizeof(double)); 
	double *transition_matrix_3 = (double*)malloc(HIDDEN_LAYER_2*OUTPUT_LAYER*sizeof(double)); 

	init_layer(transition_matrix_1,PIXELS,HIDDEN_LAYER_1);
	init_layer(transition_matrix_2,HIDDEN_LAYER_1,HIDDEN_LAYER_2);
	init_layer(transition_matrix_3,HIDDEN_LAYER_2,OUTPUT_LAYER);

	unsigned char *buffer = (unsigned char*)malloc(PIXELS*sizeof(unsigned char));
	read_file("data_batch_1.bin",buffer,transition_matrix_1,transition_matrix_2,transition_matrix_3);


}

