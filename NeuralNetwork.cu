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

/*
 * Function that given the value of the previous layer of a neural network, and its transition matrix
 * to the new layer, calculates the net value of the layer
 * In: 
 * input: vector that represents the previous layer of the layer to calculate
 * matrix: transition matrix with the weigths of the neural network
 * result: vector to store the results. It represents the layer to be calculated
 * input_size: size of the previous layer
 * hidden_size: size of the calculated layer
 *
 */
void get_layer(double *input, double *matrix, double *result,int input_size, int hidden_size){
	//Move thorugh every element of the layer
	for (int i = 0; i<hidden_size; ++i){
		//Init the neuron value in 0
		result[i] = 0.0;
		//Multiply each value of the previous neuron times its weigth and store it in result
		for (int j = 0; j < input_size; ++j){
			result[i] += input[j] * matrix[j*hidden_size+i];
		}
	}
}

/*
 * Function that apply the sigmoid function to every element of a vector
 * In:
 * double: vector to apply the signmoid function to every element
 * N: size of the vector
 */
void sigmoid(double *f, int N){
	//Move through all elements of the vector
	for (int i =0; i < N; ++i)
		//Apply the sigmoid function to every element
		//Sigmoid used: f(x) = 1 / (1 + e^(-x))
		f[i] = 1.0 / (1.0 + exp(-f[i]));
}

/*
 * Function that normalize the input, so all the values are equally important. Normalize is the process
 * to transform every element of a vector to its correspondent value beetwen 0 and 1
 * In:
 * c: vector with the numbers between 0 and 255, each one which corresponds to a pixel of the input image
 * f: vector so save the normalized vector
 * N: size of the vectors
 */
void process_input(unsigned char *c, double *f, int N){
	//Move through all elements of the vecotr
	for (int i =0; i < N; ++i){
		//Normalize x cosist of (x - Min)/ (Max-Min), in pixels Max is 255 and Min is 1
		f[i] = (c[i]-1)/254.0;
	}
		
}

/*
 * Function that returns the index corresponding to the maximum element of an array
 * In:
 * f: vector of values
 * N: size of vector
 * Out:
 * int corresponding to the index of the maximum value
 */
int max_index(double *f, int N){
	//The max is the first element
	int max_index = 0;
	for (int i = 1; i < N; ++ i){
		//If there is a new max, then substitute it
		if (f[i] > f[max_index]){
			max_index = i;
		}
	}
	//Return the index of the max element
	return max_index;
}

/*
 * Function that calculate the error of the neural network
 * In:
 * f: output vector of the neural network
 * output: expected value
 * N: size of the vector
 * Out:
 * double corresponding to the calculated error of the NN
 */
double error(double *f, int output, int N) {
	double *output_array = (double *) malloc(N * sizeof(double));
	//Init the expected answer in 0
	for (int i = 0; i < N; ++i)
		output_array[i] = 0.0;
	//Mark the expected answer 1
	output_array[output] = 1.0;

	//Init the error in 0
	double error = 0.0;
	//Calulate the total error, the rror is defined as
	//(1/2)*SUM(di - oi)^2 where di is expected value, oi is calculated value
	for (int i = 0; i < N; ++i)
		error += (output_array[i] - f[i]) * (output_array[i] - f[i]);
	return 0.5*error;
}

/*
 * Function that calculate the error of the output layers
 * In:
 * f: value of the output neurons
 * output: expected value
 * N: size of the vector
 * error_array: vector with the calculated error of every neuron
 *
 */
void error_output(double *f, int output, int N, double *error_array){
	double *output_array = (double *) malloc(N * sizeof(double));
	//Init the expected answer in 0
	for (int i = 0; i < N; ++i)
		output_array[i] = 0.0;
	//Mark the expected answer 1
	output_array[output] = 1.0;

	//Get the error for every neuron. The error in the output layer is defined as Si = (di - oi)*oi*(1-oi)
	for (int i = 0; i < N; ++i){
		error_array[i] = (output_array[i] - f[i])*f[i]*(1-f[i]);
	}
}

/* Function that calculates the error of the hidden layers
 * In: 
 * f: hidden layer calculated values
 * error_array: error vector, to save the error of every neuron in the hidden layer
 * next_layer_error: error vector of the next layer, neccessary to calculate the error of a hidden layer
 * layer_size: size of the hidden layer
 * next_size: size of the next layer
 * transition_matrix: transition matrix to propagate values from hidden to next layer 
 */
void error_hidden_layer(double *f, double* error_array, double *next_layer_error, int layer_size, int next_size, double *transition_matrix){

	//Calculate error of every neuron in a hidden layer
	//The error in a hidden layer is defined as Si = oi * (1 - oi) * SUM(Wij * Sj) where Sj is the error from next
	for (int i = 0; i < layer_size; ++i){
		//Inicial value of the sumatory
		double product = 0.0;
		for (int j = 0; j < next_size; ++j){
			//Add Wij * Sj to acumulator
			product += transition_matrix[i*next_size+j]*next_layer_error[j];
		}
		//Get the final product
		error_array[i] = f[i]*(1-f[i])*product;
	}

}

/*
 * Function that calculates the variation of weigths of a neural network
 * In:
 * error_vector: error vector of the layer
 * gradient: variance of the weights for every element
 * layer: value of the elements of the layer
 * N: rows of the transition matrix
 * M: columns ot the transition matrix
 *
 */
void calculate_error(double *error_vector, double* gradient, double *layer, int N, int M){
	//Iterate over the matrix
	for (int i =0; i<N; ++i){
		for (int j = 0; j<M; ++j){
			//The variance of the weigth is alpha * Sj * Oi
			gradient[i*M+j] = LEARNING_RATE * error_vector[j] * layer[i];
		}
	}

}

/*
 * Function that sums two matrix and store it directly in the first matrix
 * In:
 * transition: first matrix
 * gradient: second matrix
 * N: rows of the matrix
 * M: columns of the matrix
 */
void sum_matrix(double *transition, double *gradient, int N, int M){
	//Iterate over all the matrix 
	for (int i = 0; i < N; ++i){
		for (int j = 0; j < M; ++j){
			//Change the weight of every value of the transition matrix
			transition[i*M+j] = transition[i*M+j] + gradient[i*M+j];
		}
	}
}

/*
 * Function that reads a file, stores every vector of it, and then apply backpropagation
 * In:
 * file: name of the file with the data
 * buffer: vector where every pixel will be stored
 * transition_matrix_1: transition vector from input layer to hidden layer 1
 * transition_matrix_2: transition vector from hidden layer 1 to hidden layer 2
 * transition_matrix_3: transition vector from hidden layer 2 to output layer
 * elements: number of elements to use for training
 */
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

		//Start taking the time
		float tiempo1;
		cudaEvent_t inicio1, fin1;

		cudaEventCreate(&inicio1); // Se inicializan
   		cudaEventCreate(&fin1);
   		cudaEventRecord( inicio1, 0 );

   		//The first value of the vector is the expected time
		expected_output = buffer[0];
		//After the expected output is saved, it can be substituted by the bias
		buffer[0] = 1;
		//Init the layers of the network
		double *hidden_layer_1 = (double*)malloc((HIDDEN_LAYER_1+1)*sizeof(double));
		double *hidden_layer_2 = (double*)malloc((HIDDEN_LAYER_2+1)*sizeof(double));
		double *output_layer = (double*)malloc(OUTPUT_LAYER*sizeof(double));

		//Normalize the data
		double *input = (double*)malloc(PIXELS*sizeof(double));
		process_input(buffer,input,PIXELS);

		//Forward information from input layer to hidden layer 1
		get_layer(input,transition_matrix_1,hidden_layer_1,PIXELS,HIDDEN_LAYER_1);
		//Apply signmoid to hidden layer 1
		sigmoid(hidden_layer_1,HIDDEN_LAYER_1+1);
		//Assign the bias
		hidden_layer_1[HIDDEN_LAYER_1] = 1;

		//Forward information from hidden layer 1 to hidden layer 2
		get_layer(hidden_layer_1,transition_matrix_2,hidden_layer_2,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);
		//Apply signmoid to hidden layer 2
		sigmoid(hidden_layer_2,HIDDEN_LAYER_2+1);
		//Assign the bias
		hidden_layer_2[HIDDEN_LAYER_2] = 1;

		//Forward information from hidden layer 2 to output layer
		get_layer(hidden_layer_2,transition_matrix_3,output_layer,HIDDEN_LAYER_2+1,OUTPUT_LAYER);
		//Apply signmoid to output layer
		sigmoid(output_layer,OUTPUT_LAYER);

		//Get the error of the output
		double *errors_array = (double*)malloc(OUTPUT_LAYER* sizeof(double));
		error_output(output_layer,expected_output,OUTPUT_LAYER,errors_array);

		//Get the weight update for transision matrix 3
		double *transition_matrix_3_gradient = (double*)malloc((HIDDEN_LAYER_2+1)*OUTPUT_LAYER*sizeof(double));
		calculate_error(errors_array,transition_matrix_3_gradient,hidden_layer_2,HIDDEN_LAYER_2+1,OUTPUT_LAYER);

		//Get the weight update for transision matrix 2
		double *hidden_layer_array_2 = (double*)malloc((HIDDEN_LAYER_2+1)* sizeof(double));
		error_hidden_layer(hidden_layer_2,hidden_layer_array_2,errors_array,OUTPUT_LAYER,HIDDEN_LAYER_2+1,transition_matrix_3);
		double *transition_matrix_2_gradient = (double*)malloc((HIDDEN_LAYER_1+1)*HIDDEN_LAYER_2*sizeof(double));
		calculate_error(hidden_layer_array_2,transition_matrix_2_gradient,hidden_layer_1,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);

		//Get the weight update for transision matrix 1
		double *hidden_layer_array_1 = (double*)malloc((HIDDEN_LAYER_1+1)* sizeof(double));
		error_hidden_layer(hidden_layer_1,hidden_layer_array_1,hidden_layer_array_2,HIDDEN_LAYER_2,HIDDEN_LAYER_1+1,transition_matrix_2);
		double *transition_matrix_1_gradient = (double*)malloc(PIXELS*HIDDEN_LAYER_1*sizeof(double));
		calculate_error(hidden_layer_array_1,transition_matrix_1_gradient,input,PIXELS,HIDDEN_LAYER_1);

		//Update the value of the transitions matrix once all have been calculated
		sum_matrix(transition_matrix_1,transition_matrix_1_gradient,PIXELS,HIDDEN_LAYER_1);
		sum_matrix(transition_matrix_2,transition_matrix_2_gradient,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);
		sum_matrix(transition_matrix_3,transition_matrix_3_gradient,HIDDEN_LAYER_2+1,OUTPUT_LAYER);

		//Siguiente elemento 
		++i;

		//Free the information not required for the next iteration
		free(hidden_layer_1);
		free(hidden_layer_2);
		free(output_layer);
		free(input);
		free(errors_array);

		//Record the finish moment
		cudaEventRecord( fin1, 0); 
   		cudaEventSynchronize( fin1 );
   		cudaEventElapsedTime( &tiempo1, inicio1, fin1 );

   		//Add the time to the total
   		total_time += tiempo1;
		
	}

	//Take the average time
	total_time /= elements;
	printf ("Tiempo promedio por clasificacion: %f\n", total_time);
}

/*
 * Function that randomly initialize all values off the transiction matrix
 * In:
 * matrix: transition matrix of the neural network
 * N: rows of the matrix
 * M: columns of the matrix
 */
void init_layer(double *matrix, int N, int M){
	//Iterate over the matrix
	for (int i =0; i < N; ++ i){
		for (int j = 0; j < M; ++j){
			//Random number to see if its negative or positive
		    int sign = rand() % 2;
		    //Random number between 0 and 1
		    if (sign == 0)
		    	matrix[i*M+j] = (rand() % 1000000) / 1000000.0;
			else
				matrix[i*M+j] = - ((rand() % 1000000) / 1000000.0);
		}
	}
		
}

/*
 * Function that prints the value of the transition matrix
 * In:
 * matrix: transition matrix
 * N: rows of the matrix
 * M: columns of the matrix
 */
void print_layer(double *matrix, int N, int M){
	//Iterate over the matrix and print
	for (int i =0; i < N; ++ i){
		for (int j = 0; j < M; ++j)
			printf("%f ",matrix[i*M+j]);
		printf("\n");
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
	
	//Create the space for the transition matrix
	double *transition_matrix_1 = (double*)malloc(PIXELS*HIDDEN_LAYER_1*sizeof(double));
	double *transition_matrix_2 = (double*)malloc((HIDDEN_LAYER_1+1)*HIDDEN_LAYER_2*sizeof(double));
	double *transition_matrix_3 = (double*)malloc((HIDDEN_LAYER_2+1)*OUTPUT_LAYER*sizeof(double));

	//Initialize the values of the matrix
	init_layer(transition_matrix_1,PIXELS,HIDDEN_LAYER_1);
	init_layer(transition_matrix_2,HIDDEN_LAYER_1+1,HIDDEN_LAYER_2);
	init_layer(transition_matrix_3,HIDDEN_LAYER_2+1,OUTPUT_LAYER);

	//Start the time
	float tiempo1;
	cudaEvent_t inicio1, fin1;

	unsigned char *buffer = (unsigned char*)malloc(PIXELS*sizeof(unsigned char));

	//Start getting the time
	cudaEventCreate(&inicio1); // Se inicializan
   	cudaEventCreate(&fin1);
   	cudaEventRecord( inicio1, 0 );

   	//Start the training
	read_file("data_batch_1.bin",buffer,transition_matrix_1,transition_matrix_2,transition_matrix_3,elements);

	//Finish the time
	cudaEventRecord( fin1, 0); // Se toma el tiempo final.
   	cudaEventSynchronize( fin1 ); // Se sincroniza
   	cudaEventElapsedTime( &tiempo1, inicio1, fin1 );

   	//Print the time
   	printf("Tiempo total del programa: %f ms\n", tiempo1);

}

