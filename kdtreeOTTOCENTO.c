/**************************************************************************************
* 
* CdL Magistrale in Ingegneria Informatica
* Corso di Architetture e Programmazione dei Sistemi di Elaborazione - a.a. 2018/19
* 
* Progetto dell'algoritmo di Product Quantization for Nearest Neighbor Search
* in linguaggio assembly x86-32 + SSE
* 
* Fabrizio Angiulli, aprile 2019
* 
**************************************************************************************/

/*
* 
* Software necessario per l'esecuzione:
* 
*    NASM (www.nasm.us)
*    GCC (gcc.gnu.org)
* 
* entrambi sono disponibili come pacchetti software 
* installabili mediante il packaging tool del sistema 
* operativo; per esempio, su Ubuntu, mediante i comandi:
* 
*    sudo apt-get install nasm
*    sudo apt-get install gcc
* 
* potrebbe essere necessario installare le seguenti librerie:
* 
*    sudo apt-get install lib32gcc-4.8-dev (o altra versione)
*    sudo apt-get install libc6-dev-i386
* 
* Per generare il file eseguibile:
* 
* nasm -f elf32 kdtreepca32.nasm && gcc -O0 -m32 -msse kdtreepca32.o kdtreepca32c.c -o kdtreepca32c && ./kdtreepca32c
* 
* oppure
* 
* ./runkdtreepca32
* 
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>
#include <malloc.h>

#define	MATRIX		float*
#define	KDTREE		float* // modificare con il tipo di dato utilizzato

typedef struct {
    char* filename; //nome del file, estensione .ds per il data set, estensione .qs per l'eventuale query set
    MATRIX ds; //data set 
    MATRIX qs; //query set
    int n; //numero di punti del data set
    int k; //numero di dimensioni del data/query set
    int nq; //numero di punti del query set
    int h; //numero di componenti principali da calcolare 0 se PCA non richiesta
    int kdtree_enabled; //1 per abilitare la costruzione del K-d-Tree, 0 altrimenti
    KDTREE kdtree; //riferimento al K-d-Tree, NULL se costruzione non richiesta
    float r; //raggio di query, -1 se range query non richieste
    int silent; //1 per disabilitare le stampe, 0 altrimenti
    int display; //1 per stampare i risultati, 0 altrimenti
    MATRIX U; //matrice U restituita dall'algoritmo PCA
    MATRIX V; //matrice V restituita dall'algoritmo PCA
    
    //STRUTTURE OUTPUT MODIFICABILI
    int* QA; //risposte alle query in forma di coppie di interi (id_query, id_vicino)
    int nQA; //numero di risposte alle query
} params;

/*
* 
*	Le funzioni sono state scritte assumento che le matrici siano memorizzate 
* 	mediante un array (float*), in modo da occupare un unico blocco
* 	di memoria, ma a scelta del candidato possono essere 
* 	memorizzate mediante array di array (float**).
* 
* 	In entrambi i casi il candidato dovrà inoltre scegliere se memorizzare le
* 	matrici per righe (row-major order) o per colonne (column major-order).
*
* 	L'assunzione corrente è che le matrici siano in row-major order.
* 
*/

struct Node{
	float* point;
	struct Node* left;
	struct Node* right;
};

//struct Node* buildTree(MATRIX d,int rig,int col,int livello);
float* Punto(MATRIX d,int rig,int col,int c);
MATRIX matrixSX(MATRIX d,int rig,int col,int c);
MATRIX matrixDX(MATRIX d,int rig,int col,int c);
float* PuntoRif(MATRIX d,int liv,int rig,int col);
MATRIX assegnaDataset(MATRIX sx,MATRIX dx,int rig,int col,int index);
MATRIX MatriceUnaRiga(MATRIX d,int liv,int rig,int col);


void* get_block(int size, int elements) { 
    return _mm_malloc(elements*size,16); 
}


void free_block(void* p) { 
    _mm_free(p);
}


MATRIX alloc_matrix(int rows, int cols) {
    return (MATRIX) get_block(sizeof(float),rows*cols);
}


void dealloc_matrix(MATRIX mat) {
    free_block(mat);
}


/*
* 
* 	load_data
* 	=========
* 
*	Legge da file una matrice di N righe
* 	e M colonne e la memorizza in un array lineare in row-major order
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero a 32 bit
* 	successivi 4 byte: numero di colonne (M) --> numero intero a 32 bit
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri floating-point a precisione singola
* 
*****************************************************************************
*	Se lo si ritiene opportuno, è possibile cambiare la codifica in memoria
* 	della matrice. 
*****************************************************************************
* 
*/
MATRIX load_data(char* filename, int *n, int *k) {
    FILE* fp;
    int rows, cols, status, i;
    
    fp = fopen(filename, "rb");
    
    if (fp == NULL){
        printf("'%s': bad data file name!\n", filename);
        exit(0);
    }
    
    status = fread(&cols, sizeof(int), 1, fp);
    status = fread(&rows, sizeof(int), 1, fp);
    
    MATRIX data = alloc_matrix(rows,cols);
    status = fread(data, sizeof(float), rows*cols, fp);
    fclose(fp);
    
    *n = rows;
    *k = cols;
    
    return data;
}

/*
* 
* 	save_data
* 	=========
* 
*	Salva su file un array lineare in row-major order
*	come matrice di N righe e M colonne
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero a 32 bit
* 	successivi 4 byte: numero di colonne (M) --> numero intero a 32 bit
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri interi o floating-point a precisione singola
* 
*/
void save_data(char* filename, void* X, int n, int k) {
    FILE* fp;
    int i;
    fp = fopen(filename, "wb");
    if(X != NULL){
        fwrite(&n, 4, 1, fp);
        fwrite(&k, 4, 1, fp);
        for (i = 0; i < n; i++) {
            fwrite(X, 4, k, fp);
            //printf("%i %i\n", ((int*)X)[0], ((int*)X)[1]);
            X += 4*k;
        }
    }
    fclose(fp);
}



// PROCEDURE ASSEMBLY
extern void prova(params* input);


/*
*	PCA
* 	=====================
*/
void pca(params* input) {
    
    // -------------------------------------------------
    // Codificare qui l'algoritmo PCA
    // -------------------------------------------------
    //prova(input);
    // Calcola le matrici U e V
    // -------------------------------------------------
    
}

/*
*	K-d-Tree
* 	======================
*/
void kdtree(params* input) {
	int l = 0;
	int i;
	MATRIX d = input->ds;
	int rig = input->n;
	int col = input->k;
	int c;//taglio
	MATRIX sx;
	MATRIX dx;
	struct Node *nodes = (struct Node*)malloc(input->n*sizeof(struct Node));
	for(i = 0; i < input->n; i++){
		if(l<input->k){
			c=l%col;
			nodes[i].point=(float*)malloc(input->k*sizeof(float));
			nodes[i].left=(struct Node*)malloc(sizeof(struct Node*));
			nodes[i].left->point=(float*)malloc(input->k*sizeof(float));
			nodes[i].right=(struct Node*)malloc(sizeof(struct Node*));
			nodes[i].right->point=(float*)malloc(input->k*sizeof(float));
			printf("allocato!\n");
			nodes[i].point=Punto(d,rig,col,c);
			printf("punto trovato!\n");
			//VA, MA AD UN CERTO PUNTO NON MI CALCOLA LA MATRICE DI SX
			// E MI DA ERRORE DI SEGMENTAZIONE; MA SE LO FACCIO PARTIRE CON 
			//VALGRIND VA DANDOMI IL TEMPO, ANCHE SE DOPO MI SEGNALA GLI ERRORI
			MATRIX sx = matrixSX(d,rig,col,c);
			printf("Matrice SX trovata!\n");
			MATRIX dx = matrixDX(d,rig,col,c);
			printf("Matrice DX trovata!\n");
			//N.B le due matrice hanno rig/2
			if((rig/2)==1){//CONTROLLARE MEGLIO
				/*nodes[i].left->point=PuntoRif(sx,l++,rig/2,col);
				MATRIX m1 = MatriceUnaRiga(sx,l++,rig/2,col);
				printf("m1\n");
				if(*(nodes[i].left->point+c)<*(m1+c)){
					nodes[i+1].right->point=m1;
					nodes[i+1].left->point=NULL;
					printf("figlio sdx\n");				
				}//if
				else{
					nodes[i+1].right->point=NULL;
					nodes[i+1].left->point=m1;
					printf("figlio ssx\n");				
				}//else
				nodes[i].right->point=PuntoRif(dx,l++,rig/2,col);
				MATRIX m2 = MatriceUnaRiga(dx,l++,rig/2,col);
				if(*(nodes[i].right->point+c)<*(m2+c)){
					nodes[i+1].right->point=m2;
					nodes[i+1].left->point=NULL;
					printf("figlio dsx\n");					
				}//if
				else{
					nodes[i+1].right->point=NULL;
					nodes[i+1].left->point=m2;	
					printf("figlio sdx\n");				
				}//else*/
				printf("FANCULO!\n");
				break;
			}//if
			else {
				nodes[i].left->point=PuntoRif(sx,l++,rig/2,col);
				printf("Figlio sx trovato!\n");
				nodes[i].right->point=PuntoRif(dx,l++,rig/2,col);
				printf("Figlio dx trovato!\n");
			}//else
			if(i==0)
				printf("Nodo radice associato a d trovato!\n");
			d=assegnaDataset(sx,dx,rig/2,col,i++);
			rig=rig/2;		
			l++;
		}//if
	}//for
	for(i = 0; i < input->n; i++){
		free(nodes[i].point);
		free(nodes[i].left);
		free(nodes[i].right);	
	}//for
	free(nodes);
	printf("Successo!\n");
}//kdtree

float* Punto(MATRIX d,int rig,int col,int c){
	if(d==NULL){
		dealloc_matrix(d);
		exit(1);
	}
	else{
		for(int w=0; w<rig*col;w++){
			if((d+w)==NULL){
				dealloc_matrix(d);
				exit(1);			
			}
		}
	}
	float *P = (float*)malloc(col*sizeof(float));
	float somma, media;
	int i, j;
	for(i=c; i<rig*col; i+=col){
		somma+=*(d+i);	
	}
	media=somma/rig;
	for(j=c; j<rig*col; j+=col){
		if((media/(*(d+j)))>=1.0 && (media/(*(d+j)))<=1.25){
			for(i=0;i<col;i++){
				*(P+i)=*(d+j-(j%col)+i);
			}
			break;
		}	
	}
	return P;
}//Punto

MATRIX matrixSX(MATRIX d,int rig,int col,int c){
	if(d==NULL){
		dealloc_matrix(d);
		exit(1);
	}
	else{
		for(int w=0; w<rig*col;w++){
			if((d+w)==NULL){
				dealloc_matrix(d);
				exit(1);			
			}
		}
	}
	float somma, media;
	int i, j, index;
	int k = 0;
	for(i=c; i<rig*col; i+=col){
		somma+=*(d+i);	
	}
	media=somma/rig;
	for(j=c; j<rig*col; j+=col){
		if((media/(*(d+j)))>=1.0 && (media/(*(d+j)))<=1.25){
			index=j;
			break;
		}
	}
	MATRIX sx = alloc_matrix(rig/2,col);
	for(i=c; i<rig*col; i+=col){
		if((*(d+i))<(*(d+index))){
			if(k<col){
				for(j=k; j<(rig/2)*col; j++){
					*(sx+j)=*(d+i-(i%col)+j);			
				}
				k+=col;
			}
			else {
				break;
			}		
		}
		else if(index==i){
			continue;		
		}
	}
	return sx;
}//matrixSX

MATRIX matrixDX(MATRIX d,int rig,int col,int c){
	if(d==NULL){
		dealloc_matrix(d);
		exit(1);
	}
	else{
		for(int w=0; w<rig*col;w++){
			if((d+w)==NULL){
				dealloc_matrix(d);
				exit(1);			
			}
		}
	}
	float somma, media;
	int i, j, index;
	int k = 0;
	for(i=c; i<rig*col; i+=col){
		somma+=*(d+i);	
	}
	media=somma/rig;
	for(j=c; j<rig*col; j+=col){
		if((media/(*(d+j)))>=1.0 && (media/(*(d+j)))<=1.25){
			index=j;
			break;
		}
	}
	MATRIX dx = alloc_matrix(rig/2,col);
	for(i=c; i<rig*col; i+=col){
		if((*(d+i))>=(*(d+index))){
			if(k<col){
				for(j=k; j<(rig/2)*col; j++){
					*(dx+j)=*(d+i-(i%col)+j);			
				}
				k+=col;
			}
			else {
				break;
			}		
		}
		else if(index==i){
			continue;		
		}
	}
	return dx;
}//matrixSX

float* PuntoRif(MATRIX d,int liv,int rig,int col){
	int c = liv%col;
	float *P = (float*)malloc(col*sizeof(float));
	return P=Punto(d,rig,col,c);
}//PuntoRif

MATRIX assegnaDataset(MATRIX sx,MATRIX dx,int rig,int col,int index){
	if(sx==NULL && dx==NULL){
		dealloc_matrix(sx);
		dealloc_matrix(dx);
		exit(1); 
	}
	MATRIX d = alloc_matrix(rig,col);
	if(sx==NULL){
		d = dx;
		dealloc_matrix(sx);	
	}
	else if(dx==NULL){
		d = sx;
		dealloc_matrix(dx);	
	}
	if(index%2!=0){
		d = sx;	
	}//if
	else {
		d = dx;	
	}//else
	return d;
}//assegnaDataset

MATRIX MatriceUnaRiga(MATRIX d,int liv,int rig,int col){
	if(d==NULL){
		dealloc_matrix(d);
		exit(1);
	}
	else{
		for(int w=0; w<rig*col;w++){
			if((d+w)==NULL){
				dealloc_matrix(d);
				exit(1);			
			}
		}
	}
	MATRIX dt = alloc_matrix(rig,col);
	int i;
	for(i=0;i<rig*col;i++){
		*(dt+i)=*(d+i);
	}
	return dt;
}//MatriceUnaRiga

/*struct Node* buildTree(MATRIX d,int rig,int col,int livello){
	if(d==NULL){
		dealloc_matrix(d);
		exit(1);
	}
	else {
		for(int w=0; w<rig*col;w++){
			if((d+w)==NULL){
				dealloc_matrix(d);
				exit(1);			
			}
		}	
	}
	struct Node* n=(struct Node*)malloc(sizeof(struct Node*));
	int c = livello%col;
	float somma, media;
	int i, j;
	int k = 0;
	int z = 0;
	for(i=c; i<rig*col; i+=col){
		somma+=*(d+i);	
	}
	media=somma/rig;
	float* P = (float*) malloc(sizeof(float)*col);
	for(j=c; j<rig*col; j+=col){
		if((media/(*(d+j)))>=1.0 && (media/(*(d+j)))<=1.25){
			n->index=j;
			for(i=0;i<col;i++){
				*(P+i)=*(d+j-(j%col)+i);
			}
			break;
		}	
	}
	MATRIX ds1 = alloc_matrix(rig/2,col);//sx
	MATRIX ds2 = alloc_matrix(rig/2,col);//dx
	for(i=c; i<rig*col; i+=col){
		if((*(d+i))<(*(d+n->index))){
			for(j=k; j<(rig/2)*col; j++){
				*(ds1+j)=*(d+i-(i%col)+j);			
			}
			k+=col;		
		}
		else if((*(d+i))>=(*(d+n->index))){
			for(j=z; j<(rig/2)*col; j++){
				*(ds2+j)=*(d+i-(i%col)+j);			
			}
			z+=col;		
		}
		else if(n->index==i){
			continue;		
		}	
	}
	n->point=P;
	n->left=buildTree(ds1,rig/2,col,livello++);
	n->right=buildTree(ds2,rig/2,col,livello++);	
	return n;
}*/

/*
*	Range Query Search
* 	======================
*/
void range_query(params* input) {
    
    // -------------------------------------------------
    // Codificare qui l'algoritmo di ricerca
    // -------------------------------------------------

    // Calcola il risultato come una matrice di nQA coppie di interi
    // (id_query, id_vicino)
    // o in altro formato
    // -------------------------------------------------
}

int main(int argc, char** argv) {
    
    char fname[256];
    int i, j, k;
    clock_t t;
    float time;
    
    //
    // Imposta i valori di default dei parametri
    //
    
    params* input = malloc(sizeof(params));
    
    input->filename = NULL;
    input->h = 0;
    input->kdtree = NULL;
    input->r = -1;
    input->silent = 0;
    input->display = 0;
    input->QA = NULL;
    input->nQA = 0;
    
    //
    // Visualizza la sintassi del passaggio dei parametri da riga comandi
    //
    
    if (argc <= 1 && !input->silent) {
        printf("Usage: %s <data_name> [-pca <h>] [-kdtree [-rq <r>]]\n", argv[0]);
        printf("\nParameters:\n");
        printf("\t-d: display query results\n");
        printf("\t-s: silent\n");
        printf("\t-pca <h>: h-component PCA enabled\n");
        printf("\t-kdtree: kdtree building enabled\n");
        printf("\t-rq <r>: range query search with radius r enabled\n");
        printf("\n");
        exit(0);
    }
    
    //
    // Legge i valori dei parametri da riga comandi
    //
    
    int par = 1;
    while (par < argc) {
        if (par == 1) {
            input->filename = argv[par];
            par++;
        } else if (strcmp(argv[par],"-s") == 0) {
            input->silent = 1;
            par++;
        } else if (strcmp(argv[par],"-d") == 0) {
            input->display = 1;
            par++;
        } else if (strcmp(argv[par],"-pca") == 0) {
            par++;
            if (par >= argc) {
                printf("Missing h value!\n");
                exit(1);
            }
            input->h = atoi(argv[par]);
            par++;
        } else if (strcmp(argv[par],"-kdtree") == 0) {
            input->kdtree_enabled = 1;
            par++;
            if (par < argc && strcmp(argv[par],"-rq") == 0) {
                par++;
                if (par >= argc) {
                    printf("Missing radius value!\n");
                    exit(1);
                }
                input->r = atof(argv[par]);
                if(input->r < 0){
                    printf("Range query radius must be non-negative!\n");
                    exit(1);
                }
                par++;
            }
        } else{
            printf("WARNING: unrecognized parameter '%s'!\n",argv[par]);
            par++;
        }
    }
    
    //
    // Legge i dati e verifica la correttezza dei parametri
    //
    
    if(input->filename == NULL || strlen(input->filename) == 0){
        printf("Missing input file name!\n");
        exit(1);
    }
    
    sprintf(fname, "%s.ds", input->filename);
    input->ds = load_data(fname, &input->n, &input->k);

    if(input->h < 0){
        printf("Invalid value of PCA parameter h!\n");
        exit(1);
    }
    if(input->h > input->k){
        printf("Value of PCA parameter h exceeds data set dimensions!\n");
        exit(1);
    }
    
    if(input->r >= 0){
        sprintf(fname, "%s.qs", input->filename);
        input->qs = load_data(fname, &input->nq, &k);
        if(input->k != k){
            printf("Data set dimensions and query set dimensions are not compatible!\n");
            exit(1);
        }
    }
    
    //
    // Visualizza il valore dei parametri
    //
    
    if(!input->silent){
        printf("Input file name: '%s'\n", input->filename);
        printf("Data set size [n]: %d\n", input->n);
        printf("Number of dimensions [k]: %d\n", input->k);
        if(input->h > 0){
            printf("PCA search enabled\n");
            printf("Number of principal components [h]: %i\n",input->h);
        }else{
            printf("PCA search disabled\n");
        }
        if(input->kdtree_enabled){
            printf("Kdtree building enabled\n");
            if(input->r >= 0){
                printf("Range query search enabled\n");
                printf("Range query search radius [r]: %f\n",input->r);
            }else{
                printf("Range query search disabled\n");
            }
        }else{
            printf("Kdtree building disabled\n");
        }
    }

    //
    // Calcolo PCA
    //
    
    if(input->h > 0){
        t = clock();
        pca(input);
        t = clock() - t;
        time = ((float)t)/CLOCKS_PER_SEC;
        sprintf(fname, "%s.U", input->filename);
        save_data(fname, input->U, input->n, input->h);
        sprintf(fname, "%s.V", input->filename);
        save_data(fname, input->V, input->k, input->h);
    }else
        time = -1;
       
    if (!input->silent)
        printf("\nPCA time = %.3f secs\n", time);
    else
        printf("%.3f\n", time);
    
    //
    // Costruzione K-d-Tree
    //
    
    if(input->kdtree_enabled){
        t = clock();
        kdtree(input);
        t = clock() - t;
        time = ((float)t)/CLOCKS_PER_SEC;
    }else
        time = -1;
    if (!input->silent)
        printf("\nIndexing time = %.3f secs\n", time);
    else
        printf("%.3f\n", time);

    //
    // Range query search
    //
    
    if(input->r >= 0){
        t = clock();
        range_query(input);
        t = clock() - t;
        time = ((float)t)/CLOCKS_PER_SEC;
    }else
        time = -1;
    if (!input->silent)
        printf("\nQuerying time = %.3f secs\n", time);
    else
        printf("%.3f\n", time);
    
    //
    // Salva il risultato delle query
    // da modificare se si modifica il formato delle matrici di output
    //
    
    if(input->r >= 0){
        if(!input->silent && input->display) {
            //NB: il codice non assume che QA sia ordinata per query, in caso lo sia ottimizzare il codice
            printf("\nQuery Answer:\n");
            for(i = 0; i < input->nq; i++){
                printf("query %d: [ ", i);
                for(j = 0; j < input->nQA; j++)
                    if(input->QA[j*2] == i)
                        printf("%d ", input->QA[j*2+1]);
                printf("]\n");
            }
            printf("\n");
        }
        sprintf(fname, "%s.qa", input->filename);
        save_data(fname, input->QA, input->nQA, 2);
    }
    
    if (!input->silent)
        printf("\nDone.\n");

    return 0;
}
