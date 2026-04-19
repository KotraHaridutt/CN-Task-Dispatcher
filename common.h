/* 
 * common.h – Shared definitions for the Task Dispatcher system
 * This header is included by both master.c and worker.c.
 * It defines the network port, limits, task types, and the
 * two structures exchanged over the socket (Task and Result).
*/

#ifndef COMMON_H
#define COMMON_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>   

#include <unistd.h>       
#include <pthread.h>      
#include <arpa/inet.h>    


#define PORT        8080   
#define MAX_WORKERS 8      
#define MAX_DATA    512    
#define BUF_SIZE    1024   

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Task-type identifiers ---- */
#define TASK_DOT_PRODUCT  1   // Vector dot product (NN core)       
#define TASK_DCT          2   // 1D Discrete Cosine Transform coeff 
#define TASK_CONVOLUTION  3   // 1D signal convolution (edge detect)



/* Task: sent from master → worker */
typedef struct {
    int task_id;            
    int task_type;          
    int data[MAX_DATA];     
    int data_size;          
    int param;              
} Task;

/* Result: sent from worker → master */
typedef struct {
    int  task_id;           
    long result;            
} Result;

#endif /* COMMON_H */