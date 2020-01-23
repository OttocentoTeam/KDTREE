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
    int cont = 0;
    int punti = 0;
    int idQ = 0;
    int idN = 0;
    int indiceQA =0;
    int* ris;
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
	int id;
    float* point;
    int cut_dim;
    float* H;
    struct tree *left;
    struct tree *right;
};//tree



struct tree* buildTree(MATRIX d,int livello,int inizio_matrice,int fine_matrice,int col);
void ordinaDataset(MATRIX d,int inizio_matrice,int fine_matrice,int col,int c);
void Scambia(float a,float b);
float EuclideanDistance(float* p, float* q, int dim);
//float minCol (MATRIX D, int rig, int col);
//float maxCol (MATRIX D, int rig, int col);
//float* build_Matrix(KDTREE albero, struct tree *nodo, int k, int n,int index_supporto);
float Distance (MATRIX H, float* Q, int k);

float minDim(KDTREE albero, int dim);
float maxDim(KDTREE albero, int dim);
float* build_region(struct tree *nodo, int h);
void ListaPunti(KDTREE albero, float* Q, float r, int h, int i);

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

float* centraPunto(int n, int k, MATRIX ds, float* Q){
    float* puntoCentrato = alloc_matrix(1,k);

    float sum;
    float media;
    for(int i=0; i<k; i++){
        sum=0.0;
        media=0.0;
        for(int j=i; j<n*k; j+=k){
            sum+=ds[j];
        }
        media=sum/n;
        puntoCentrato[i]=Q[i]-media;
    }
    return puntoCentrato;
}

MATRIX mul(float* q, MATRIX v, int k, int h){
    MATRIX mat = alloc_matrix(1, h);
    for(int i = 0; i < h; i++){
        float somma = 0.0;
        int index = 0;
        for(int j = 0; j<k*h; j+=h){
            if(index<k){
                somma+=q[index]*v[i*k+j];
                index++;
            }
        }
        mat[i]=somma;
    }
    return mat;
}

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
                int w=0;
                for(int i=j; i<k*h; i+=h){
                    V[i]=v[w];
                    w++;
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
    int col = input->k;
    int pca = input->h;
    MATRIX d = input->ds;
    struct tree *root;
    //printf("Creazione del kdtree iniziata\n");
    int l = 0;
    if(pca>0){
        root = buildTree(U,l,inizio_matrice,fine_matrice,pca);
    }
    else {
        root = buildTree(d,l,inizio_matrice,fine_matrice,col);
    }
    //printf("Creazione del kdtree terminata\n");
    input->kdtree = root;
    //printf("Successo!\n");
}//kdtree

struct tree* buildTree(MATRIX d,int livello,int inizio_matrice,int fine_matrice,int col){
    if(d==0){
        return NULL;
    }
    int c = livello%col;
    /*if(livello < col){
        c = livello%col;
        ordinaDataset(d,inizio_matrice,fine_matrice,col,c);
    }*/
    ordinaDataset(d,inizio_matrice,fine_matrice,col,c);
    int index = ((fine_matrice-inizio_matrice)/2)+inizio_matrice;
    int i;
    struct tree* node = (struct tree*)malloc(sizeof(struct tree));
    node->point = (float*)malloc(col*sizeof(float));
    for(i = 0; i < col; i++){
        node->point[i] = d[(index*col)+i];
    }
    /*if(c==0 && livello==0){
        node->father = NULL;
    } else{
        printf("padri\n");
        node->left->father = node;
        node->right->father = node;
        printf("padri vecchi\n");
    }*/
    node->cut_dim = c;
    node->id=idN;
    idN++;
    for(int x = inizio_matrice; x < index; x++){
        if(d[x*col+c]==d[index*col+c]){
            for(int y = 0; y < col; y++){
                Scambia(d[x*col+y],d[index*col+y]);
            }
            index = x;
            break;
        }
    }
    /*if((fine_matrice-inizio_matrice)!=0 && (fine_matrice-inizio_matrice)!=1){
        node->left = buildTree(d,livello++,inizio_matrice,index-1,col);
        node->right = buildTree(d,livello++,index+1,fine_matrice,col);
    }*/
    int startDX = index+1;
    int endSX = index-1;
    /*if(endSX<0){
        node->right = buildTree(d,livello++,startDX,fine_matrice,col);
    }*/
    if(endSX>inizio_matrice && startDX<fine_matrice){
        node->left = buildTree(d,livello++,inizio_matrice,endSX,col);
        node->right = buildTree(d,livello++,startDX,fine_matrice,col);
    } 
    else if((fine_matrice-inizio_matrice)==1){
        node->right = buildTree(d,livello++,startDX,fine_matrice,col);
    }
    else if((fine_matrice-inizio_matrice)==0){
        node->left = NULL;
        node->right = NULL;
        return node;
    }
    //free(node);
    
    /*for(int i=0; i<col; i++){
        printf("%f ,", node->point[i]);
    }
    printf("\n");*/
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
}

float EuclideanDistance(float* p, float* q, int k) { //metodo per il calcolo della distanza tra due punti
      float somma=0.0;
      //float var = 0;
      /*for(int x = 0; x < k; x++){
          printf("%f %f\n",p[x],q[x]);
      }
      exit(1);*/
      for(int i=0; i<k; i++){
           // var = ((q[i])-(p[i]))*((q[i])-(p[i])); //differenza delle k dimensioni ed elevamento al 2
            if(p[i]>=q[i]){
                somma+=(p[i]-q[i])*(p[i]-q[i]);
            }
            else{
                somma+=(q[i]-p[i])*(q[i]-p[i]);
            }
            //somma+=(p[i]-q[i])*(p[i]-q[i]);
      }
      
      float val=sqrtf(somma);
      //printf("%f\n", val);
      return val;
}

float minDim(KDTREE albero, int dim){   //Metodo per il calcolo del minimo su una dimensione
    if(albero==NULL) return 0.0;
    float* punto = albero->point;
    float min = punto[dim];
    float tmp;
    if((albero->left)==NULL && (albero->right)==NULL){
        return min;
    }
    if((albero->left)!=NULL && (albero->right)!=NULL){
        //printf("sx e dx\n");
        float sx = minDim(albero->left, dim);
        //printf("sx\n");
        float dx = minDim(albero->right, dim);
        //printf("dx\n");
        if(sx<=dx){
            tmp = sx;
        }
        else{
           tmp = dx;
        }
    }
    if ((albero->left)==NULL){
    	
        tmp = minDim(albero->right, dim);
    }
    if ((albero->right)==NULL){
        tmp = minDim(albero->left, dim);
    }
    if (tmp<=min){
        return tmp;
    }
    return min;
}


float maxDim(KDTREE albero, int dim){   //Metodo per il calcolo del massimo su una dimensione
    //if(albero==NULL) return NULL;
    float* punto = albero->point;
    float max = punto[dim];
    float tmp;
    if((albero->left)==NULL && (albero->right)==NULL){
        return max;
    }
    if((albero->left)!=NULL && (albero->right)!=NULL){
        float sx = maxDim(albero->left, dim);
        float dx = maxDim(albero->right, dim);
        if(sx>=dx){
            tmp = sx;
        }
        else{
           tmp = dx;
        }
    }
    if ((albero->left)==NULL){
        tmp = maxDim(albero->right, dim);
    }
    if ((albero->right)==NULL){
        tmp = maxDim(albero->left, dim);
    }
    if (tmp>=max){
        return tmp;
    }
    return max;
}


MATRIX build_region_radice(KDTREE nodo, int k){ //metodo per la costruzione della regione H del nodo radice a partire dal nodo n
      MATRIX H = alloc_matrix(k, 2);
      int i;
      for(i=0; i<k; i++){
          //printf("trovo minimo e massimo\n");
          H[i*2]=minDim(nodo, i);
          //printf("min\n");
          H[i*2+1]=maxDim(nodo, i);
          //printf("max\n");
      }
      return H;
}

MATRIX build_region_figlio(KDTREE nodo, float* punto_padre, int c_padre, float* H_padre, int k){
        
        MATRIX H_figlio = alloc_matrix(k, 2);
        int i;
        for(i=0; i<k; i++){
            if(i!=c_padre){ //per dimensioni diverse da quella di taglio
               H_figlio[i*2] = H_padre[i*2];
               H_figlio[i*2+1] = H_padre[i*2+1];
            }
            else{
                if((nodo->point[c_padre])<(punto_padre[c_padre])){ //per la dimensione di taglio
                    //per il figlio sx
                    H_figlio[i*2] = H_padre[i*2];
                    H_figlio[i*2+1] = punto_padre[c_padre];
                }
                else{
                    //per il figlio dx
                    H_figlio[i*2] = punto_padre[c_padre];
                    H_figlio[i*2+1] = H_padre[i*2+1];
                }
                
            }
        }
        return H_figlio;
}

int assegnaRegioni(KDTREE albero, int k){
    //printf("TEST 0\n");
    if(albero==NULL){
    	//printf("TEST\n");
        return 0;
    }
    if(albero->left ==NULL && albero->right==NULL){
    	//printf("TEST 1\n");
        return 0;
    }
    
    if(cont==0){
    	
        float* H_radice = alloc_matrix(k,2);
        
        H_radice = build_region_radice(albero, k);
        
        albero->H=H_radice;
        dealloc_matrix(H_radice);
        if(albero->left!=NULL){
            float* H_figlio = alloc_matrix(k,2);
            H_figlio=build_region_figlio(albero->left, albero->point, albero->cut_dim, albero->H, k);
            albero->left->H= H_figlio;
            dealloc_matrix(H_figlio);
        }

        if(albero->right!=NULL){
            float* H_figlio = alloc_matrix(k,2);
            H_figlio=build_region_figlio(albero->right, albero->point, albero->cut_dim, albero->H, k);
            albero->right->H=H_figlio;
            dealloc_matrix(H_figlio);
        }
        
        /*for(int i=0; i<k; i++){
            printf("%f\n", albero->H[i]);
        }*/

        cont++;
        assegnaRegioni(albero->left, k);
        assegnaRegioni(albero->right, k);
    }
    else{
        if(albero->left!=NULL){
            float* H_figlio = alloc_matrix(k,2);
            H_figlio=build_region_figlio(albero->left, albero->point, albero->cut_dim, albero->H, k);
            albero->left->H= H_figlio;
            dealloc_matrix(H_figlio);
        }
        if(albero->right!=NULL){
            float* H_figlio = alloc_matrix(k,2);
            H_figlio=build_region_figlio(albero->right, albero->point, albero->cut_dim, albero->H, k);
            albero->right->H=H_figlio;
            dealloc_matrix(H_figlio);

            /*for(int i=0; i<k; i++){
            printf("%f\n", albero->right->H[i]);
            }*/
        }

        assegnaRegioni(albero->left, k);
        assegnaRegioni(albero->right, k);
    }
    return 1;    
}

float Distance (MATRIX H, float* Q, int k){ //prima c'era float* H al posto di MATRIX H
    // cerco il punto P della regione H piu` prossimo a Q 
    int j;
    float* P = alloc_matrix(1, k);
    for(j=0; j<k; j++){
        if(Q[j]<=H[j*2]){
            P[j]=H[j*2];
        }
        else if(Q[j]>=H[j*2+1]){
            P[j]=H[j*2+1];
        }
        else{
            P[j]=Q[j];
        }
    }
    return EuclideanDistance(P,Q,k);    
}

/*
*	Range Query Search
* 	======================
*/
void ListaPunti(KDTREE albero, float* Q, float r, int k, int idQ){
    if(albero==NULL) return;
    //if(r<0) return NULL;
    
    if(Distance(albero->H,Q,k)>=r){
        return; 
    }
    float* P = albero->point;

    //printf("p fatto\n");
    float ed = EuclideanDistance(P,Q,k);
    //printf("Distanza euclidea calcolata: %f\n", ed);
    if(ed<=r){
        if(indiceQA==0){
            ris[indiceQA]=idQ;
            ris[indiceQA+1]=albero->id;
            indiceQA+=2;
        }
        else{
            ris=realloc(ris, (indiceQA+2)*(sizeof(int)));
            ris[indiceQA]=idQ;
            ris[indiceQA+1]=albero->id;
            indiceQA+=2;
        }
        punti++;
        //printf("aggiunto punto nella lista\n");
    }
    if(albero->left != NULL){
        //printf("vado a sx\n");
        ListaPunti(albero->left, Q, r, k, idQ);
    }
    if(albero->right != NULL){
        //printf("vado a dx\n");
        ListaPunti(albero->right, Q, r, k, idQ);
    }
}

/*void stampaLista(struct list *lista){
    printf("%p\n", lista->point);
    stampaLista(lista->next);
};

void stampaAlbero(KDTREE albero){
    printf("%p\n", albero->point);
    stampaAlbero(albero->left);
    stampaAlbero(albero->right);
};*/

void stampaAlbero(KDTREE albero, int k){
    if(albero==NULL){
        exit(-1);
    }  
    float punto[k];
    for(int i=0; i<k; i++){
       printf("%f\n", albero->point[i]);
    }
    
    stampaAlbero(albero->left, k);
    stampaAlbero(albero->right, k);
}


void range_query(params* input) {
    int dim = input->k;
    int pca = input->h;
    int num = input->n;
    int dimqs = input->nq;
    MATRIX d = input->ds;
    MATRIX queryset = input->qs;
    float raggio = input->r;
    KDTREE albero = input->kdtree;
    float* punto; 
    ris=(int*)malloc(2*sizeof(int));   
    

    //stampaAlbero(albero, dim);

    
    if(pca>0){ //nel caso in cui fosse abilitata la pca
        assegnaRegioni(albero, pca);
        //printf("TEST\n");
        for(int i=0; i<dimqs*pca; i+=pca){
            //struct list* List;
            //printf("si prende punto da qs\n");
            float* puntoqs = alloc_matrix(1,pca);
            for(int j=0; j<pca; j++){
                puntoqs[j]=queryset[i*pca+j];
            } 

        /* In particolare, il punto Q dovr`a essere centrato rispetto alla media del dataset 
        e trasformato moltiplicandolo per la matrice dei load V . Quindi, sia b Q il punto 
        Q centrato sulla media del dataset, la range query deve essere eﬀettuata sul punto 
        ................................Q′ = Qcentrato*V........................ */
            //printf("si centra il punto\n");
            float* qc = centraPunto (num, pca, U, puntoqs);
            //printf("ok\n");
            float* q1 = mul(qc,V, pca, dim); //va fatta la moltiplicazione vettore matrice
            //printf("mul fatta\n");
          
            ListaPunti(albero, q1, raggio, pca, i);
            dealloc_matrix(puntoqs);
            
        }

    }
    else{
        assegnaRegioni(albero, dim);
        printf("CHECK\n");
        for(int i=0; i<dimqs*dim; i+=dim){
            
        //printf("si prende punto da qs\n");
            float* puntoqs = alloc_matrix(1,dim);
            for(int j=0; j<dim; j++){
                puntoqs[j]=queryset[i*dim+j];
            } 
	
            ListaPunti(albero, puntoqs, raggio, dim, i);

  			dealloc_matrix(puntoqs);
        }
    }
    input->QA=ris;
    input->nQA=punti;
    
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
        printf("\nIndexing time = %.6f secs\n", time);
    else
        printf("%.6f\n", time);

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
            printf("Trovati %d punti \n", punti);
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
