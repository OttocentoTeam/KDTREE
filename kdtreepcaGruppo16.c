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
#include <libgen.h>
#include <xmmintrin.h>

#define	MATRIX		float*
#define	KDTREE		struct tree* // modificare con il tipo di dato utilizzato

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
  
    
    //STRUTTURE OUTPUT MODIFICABILI
    int* QA; //risposte alle query in forma di coppie di interi (id_query, id_vicino)
    int nQA; //numero di risposte alle query
} params;
    
    //matrici 
    MATRIX U; //matrice U restituita dall'algoritmo PCA
    MATRIX V; //matrice V restituita dall'algoritmo PCA
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

struct tree{
    float* point;
    struct tree *left;
    struct tree *right;
};//tree

struct tree* buildTree(MATRIX d,int livello,int inizio_matrice,int fine_matrice,int col);
void ordinaDataset(MATRIX d,int inizio_matrice,int fine_matrice,int col,int c);
void Scambia(float a,float b);
float EuclideanDistance(float* p, float* q, int dim);
float minCol (float* col, int dim);
float maxCol (float* col, int dim);


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
        fwrite(&k, 4, 1, fp);
        fwrite(&n, 4, 1, fp);
        for (i = 0; i < n; i++) {
            fwrite(X, 4, k, fp);
            //printf("%i %i\n", ((int*)X)[0], ((int*)X)[1]);
            X += 4*k;
        }
    }
    fclose(fp);
}

/*metodo per centrare il dataset sulla media */
MATRIX centraDS(int n, int k, MATRIX ds){
    //si suppone ds di dimensione n*k
    MATRIX dataC= alloc_matrix(n,k);

    float sum;
    float media;
    for(int i=0; i<k; i++){
        sum=0.0;
        media=0.0;
        for(int j=i; j<n*k; j+=k){
            sum+=ds[j];
        }
    media=sum/n;
        for(int ii=i; ii<n*k; ii+=k){
            dataC[ii]=ds[ii]-media;
        }
    }

    return dataC;
}//centraDS

/*estrae il vettore u dal dataset*/ 
MATRIX estraicolonna(int n, int k, MATRIX ds){
    //si suppone ds di dimensione n*k
   MATRIX u=alloc_matrix(n,1);
   
   int j;
   int i;
    for( j=0,i=0 ; j<n*k ; j+=k, i++){
            u[i]=ds[j];
    }

   return u;     
}//estraicolonna

/*calcolo del dataset trasposto*/
MATRIX datasetTrasposto(int n, int k, MATRIX ds){
    //si suppone ds di dimensione n*k
    MATRIX dt=alloc_matrix(k,n);
    int z=0;

    for( int i=0; i<k; i++){
        for(int j=i; j<n*k; j+=k){
            dt[z]=ds[j];
            z++;
        }
    }

    return dt;
}

/*calcolo del prodotto tra il dataset trasposto e il vettore u*/
MATRIX prodottoDtu(int n, int k, MATRIX dst, MATRIX u){
    //si assume dst di dimensione k*n e u di dimensione n*1
    MATRIX ris=alloc_matrix(k,1);
    float tmp;
    int z=0;
        for(int i=0; i<k;i++){
            tmp=0.0;
            for(int j=0;j<n;j++){
                tmp+=dst[z]*u[j];
                z++;
            }
            ris[i]=tmp;
        }
    return ris;    

}//prodottoDtu

/*calcolo del prodotto tra u e u trasposto*/
float prodottoutu(int n, MATRIX u){
    //si suppone u di dimensione n*1
    float ris;
        for(int i=0; i<n; i++){
            ris+=u[i]*u[i];
        }
    return ris;
}//prodottoutu

/*calcolo del prodotto tra ds e v*/
MATRIX prodottoDv(int n, int k, MATRIX ds, MATRIX v){
    //ds dimensione n*k, v dimensione k*1
    MATRIX dv=alloc_matrix(n,1);
    float sum;
    int z=0;
        for(int i=0; i<n; i++){
            sum=0.0;
            for(int j=0; j<k; j++){
               sum+=ds[z]*v[j];
               z++;
            }
            dv[i]=sum;
        }
    return dv;    
}

/*calcola il prodotto tra v trasposto e v*/
float prodottovtv(int k, MATRIX v){
    //si suppone v di dimensione k*1
    float ris=0.0;
     for(int i=0; i<k ; i++){
        ris+=v[i]*v[i];
     }

    return ris;
}

/*calcola il prodotto tra u e v trasposto*/
MATRIX prodottouvt(int n, int k, MATRIX u, MATRIX v){
    MATRIX uvt=alloc_matrix(n,k);
    int z=0;
    for(int i=0; i<n; i++){
        for(int j=0;j<k;j++){
            uvt[z]=u[i]*v[j];
            z++;
        }
    }
    return uvt;
}

// PROCEDURE ASSEMBLY
extern void prova(params* input);


/*
*	PCA
* 	=====================
*/
void pca(int n, int k, int h, MATRIX ds) {
  float norma,t1,t2,diff,vabs;
  
  float teta=0.00000001;  //imposta la soglia teta a 1e-8
  
  MATRIX dataC= alloc_matrix(n,k);
  dataC=centraDS(n,k,ds);  //centra il data set sulla media

  MATRIX u=alloc_matrix(n,1);
  u=estraicolonna(n,k,dataC); //sia u la prima colonna del ds

  MATRIX v=alloc_matrix(k,1);

  MATRIX dsT=datasetTrasposto(n,k,dataC);
  MATRIX dtu=alloc_matrix(k,1); //conterrà prodotto tra ds trasposto e u

  float utu;                    //conterrà prodotto tra u trasposto e u

  MATRIX dv=alloc_matrix(n,1);  //conterrà prodotto tra ds e v

  float vtv;                    //conterrà prodotto tra v trasposto e v

  U=alloc_matrix(n,h);
  V=alloc_matrix(k,h);

  MATRIX uvt=alloc_matrix(n,k);

   for(int j=0; j<h; j++){
              //calcola vettore dei load v
      
      label:  dtu=prodottoDtu(n,k,dsT,u);
              utu=prodottoutu(n,u);

               for(int i=0; i<k; i++){
                v[i]=dtu[i]/utu;
               }

               //normalizza il vettore dei load v

               norma=0.0;
               for(int i=0; i<k; i++){
                 norma+=pow(v[i],2);
               }

               norma=sqrt(norma);

               for(int i=0; i<k; i++){
                 v[i]=v[i]/norma;
               }

                //t=ut*u  (t=t1)
                t1=prodottoutu(n,u);

                //aggiorna il vettore degli score
                dv=prodottoDv(n,k,dataC,v);
                vtv=prodottovtv(k,v);
                
                for(int i=0; i<n ; i++){
                    u[i]=dv[i]/vtv;
                }

                 //t'=ut*u (t'=t2)
                 t2=prodottoutu(n,u);

                  //if
                  diff=t2-t1;
                  vabs=abs(diff);
                  if(vabs>=(teta*t2)){
                    goto label;
                  }

                //inserisci u come j-esima colonna di U

                int z=0;
                for(int i=j; i<n*h; i+=h){
                    U[i]=u[z];
                    z++;
                }

                //inserisci v come j-esima colonna di V
                z=0;
                for(int i=j; i<k*h; i+=h){
                    V[i]=u[z];
                    z++;
                }

                //aggiorna il dataset
                uvt=prodottouvt(n,k,u,v);
                for(int i=0; i<n*k;i++){
                    dataC[i]-=uvt[i];
                }
    }//for           
    //prova(input);
    
}

/*
*	K-d-Tree
* 	======================
*/
void kdtree(params* input) {
    int inizio_matrice = 0;
    int fine_matrice = input->n-1;
    int col = input->k-1;
    MATRIX d = input->ds;
    struct tree *root;
    printf("Creazione del kdtree iniziata\n");
    int l = 0;
    root = buildTree(d,l,inizio_matrice,fine_matrice,col);
    printf("Creazione del kdtree terminata\n");
    input->kdtree = root;
    printf("Successo!\n");
}//kdtree

struct tree* buildTree(MATRIX d,int livello,int inizio_matrice,int fine_matrice,int col){
    if(d==0){
        return NULL;
    }
    int c = livello%col;
    ordinaDataset(d,inizio_matrice,fine_matrice,col,c);
    int index = ((fine_matrice-inizio_matrice)/2)+inizio_matrice;
    int i;
    struct tree* node = (struct tree*)malloc(sizeof(struct tree));
    node->point = (float*)malloc(col*sizeof(float));
    for(i = 0; i < col; i++){
        node->point[i] = d[(index*col)+i];
    }
    if((fine_matrice-inizio_matrice)!=0 && (fine_matrice-inizio_matrice)!=1){
        node->left = buildTree(d,livello++,inizio_matrice,index-1,col);
        node->right = buildTree(d,livello++,index+1,fine_matrice,col);
    }
    else if((fine_matrice-inizio_matrice)==1){
        node->right = buildTree(d,livello++,index+1,fine_matrice,col);
    }
    free(node);
    return node;
}//buildTree

/*void ordinaDataset(MATRIX d,int inizio_matrice,int fine_matrice,int col,int c){
    int i, j, z;
    for(i = inizio_matrice; i < fine_matrice; i+=col){
        for(z = i+1; z < fine_matrice; z++){
            if((d[i*col+c])>(d[z*col+c])){
                for(j = 0; j < col; j++){
                    Scambia(d[i*col+j],d[z*col+j]);
                }
            }
        }
    }
}*///ordinaDataset-BUBBLESORT FUNZIONANTE

void ordinaDataset(MATRIX d,int inizio_matrice,int fine_matrice,int col,int c){
    if(inizio_matrice >= fine_matrice) return;
  	int i = inizio_matrice, j = fine_matrice;
  	int tmp;
	int x;
	float pivot = d[i*col+c];
  	for(;;) {
		while(d[i*col+c] < pivot) i++;
		while(pivot < d[j*col+c]) j--;
    	if(i >= j) break;
		for(x=0;x<col;x++){
            tmp = d[i*col+x]; 
			d[i*col+x] = d[j*col+x]; 
			d[j*col+x] = tmp;
		}
    		i++; j--;
  	}
  	ordinaDataset(d,inizio_matrice, i-1,col,c);
  	ordinaDataset(d, j+1, fine_matrice,col,c);
}//ordinaDataset-QUICKSORT FUNZIONANTE

void Scambia(float a,float b){
    float tmp;
    tmp = a;
    a = b;
    b = tmp;
}//Scambia

float EuclideanDistance(float* p, float* q, int dim) { //metodo per il calcolo della distanza tra due punti
      float somma=0;
      for(int i=0; i<dim; i++){
            int var = ((q[i])-(p[i]))*((q[i])-(p[i])); //differenza delle k dimensioni ed elevamento al 2
            somma+=var;
      }
      return sqrt(somma);
}

float minCol (float* col, int dim){ //metodo per il calcolo del minimo della colonna j
    float min = col[0]; //prima dimensione
    int i;
    for(i=0; i<(dim-1);i++){
        if(col[i]<=min){
            min=col[i];
        }
    }
    return min;
}

float maxCol (float* col, int dim){ //metodo per il calcolo del massimo della colonna j
    float max= col[0]; //prima dimensione
    int i;
    for(i=0;i<(dim-1);i++){
        if(col[i]>=max){
            max=col[i];
        }
    }
    return max;
}

//FINO A QUA E' CORRETTO

/*float Distance (float* H, float* Q){   //RIVEDERE
    //cerco Hj min e Hj max
    int dim = input->k;
    float* P =  alloc_matrix(1, dim);
    int j;
    for(j=0;j<dim; j++){
        if(Q[j]<=H[j]){
            P[j]=H[j];
        }
        else if (Q[j]>=*(H+j+1)){
            P[j]=*(H+j+1);
        }
        else{
            P[j] = Q[j];
        }    
    return EuclideanDistance(P, Q, dim);
    }
}    

float* build_region(KDTREE albero, MATRIX DS, int inizio_matrice,int fine_matrice,int dim){ //metodo per la costruzione della regione H a partire dal nodo n
    
    float* H = alloc_matrix(dim, 2);
    
    
}*/

/*
*	Range Query Search
* 	======================
*/
void range_query(params* input) {
    
    /*int raggio = input->r;

    float* Q = input->kdtree;
    float* L = alloc_matrix(1,/*dimensione dataset); //array dei punti la cui distanza da q è minore di r 
    
    if(Distance(Q, H(n))>raggio){
        return 0;
    }
    float* P = alloc_matrix(1, input->k);

    int Dis = EuclideanDistance(Q,P, input->k);
    if(Dis<=raggio){
       //append Q dentro L

    }
    if(n->figliosx)!=NULL{
        L=L+&(range_query(figliosx, Q, r));
    }
    if(n->figliodx)!=NULL{
        L=L+&(range_query(figliodx, Q, r));
    }
    return L;*/
}


int main(int argc, char** argv) {
    
    char fname[256];
    char* dsname;
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
    dsname = basename(strdup(input->filename));
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
        pca(input->n, input->k, input->h, input->ds);
        t = clock() - t;
        time = ((float)t)/CLOCKS_PER_SEC;
        sprintf(fname, "%s.U", dsname);
        sprintf(fname, "%s.V", dsname);
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
        sprintf(fname, "%s.qa", dsname);
        save_data(fname, input->QA, input->nQA, 2);
    }
    
    if (!input->silent)
        printf("\nDone.\n");

    return 0;
}
