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

    /*AGGIUNTE MIE*/
    float* Radice;
    
} params;

//AGGIUNTE
MATRIX dataT; //dataset trasposto
float *pdt;
MATRIX u;
MATRIX v;
float *pu;
float *pv;

//FUNZIONI AGGIUNTIVE
KDTREE buildTree(params* input, MATRIX DT, int liv, int dimensione);
MATRIX dsTrasposto(params* input);
MATRIX centraDataSet(params* input);
MATRIX generau(params* input);
MATRIX prodottoDtu(params* input);
float prodottoutu(params* input);
MATRIX prodottoDv(params* input);
MATRIX prodottouvt(params* input);


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


void* get_block(int size, int elements) { 
    return _mm_malloc(elements*size,32); 
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
    int n=input->n;
    int k=input->k;
    int h=input->h;

    float norma;
    float t1,t2;  // t e t'
    float diff;   //differenza t'-t
    float vabs;

    MATRIX dtu; //prodotto Dt * u
    float *pdtu;
    float utu;  //prodotto ut *u

    MATRIX dv;  //prodotto D*v
    float *pdv;

    MATRIX uvt; //prodotto u*vt    
    float *puvt;

    float *pd;     //puntatore al dataset
    float *pmu;   //puntatore ad U
    float *pmv;   //puntatore a V
    
    MATRIX matu;
    MATRIX matv;

    matu=calloc(n*h,sizeof(float)); //è corretto  usare input->U?
    matv=calloc(k*h,sizeof(float));

    float teta=0.00000001;  //imposta la soglia teta a 1e-8
    input->ds=centraDataSet(input);   //centra il DataSet sulla media
    u=generau(input);            //sia u la prima colonna di D

    v=calloc(k,sizeof(float));
    float vtv;

    for(int j=0; j<h; j++){
    //calcola vettore dei load v
      
      label:  dtu=prodottoDtu(input);
              utu=prodottoutu(input);
    
              for(int i=0; i<k; i++){
                pdtu=dtu+i;
                pv=v+i;
                *pv=(*pdtu)/utu;
          
              }
              
              
      //normalizza il vettore dei load v

       norma=0.0;
       for(int i=0; i<k; i++){
         pv=v+i;
         norma+=pow(*pv,2);
       }

       norma=sqrt(norma);

       for(int i=0; i<k; i++){
        pv=v+i;
        *pv=(*pv)/norma;
       }

     //t=ut*u  (t=t1)
      t1=prodottoutu(input);

      //aggiorna il vettore degli score
      dv=prodottoDv(input);
      vtv=0.0;
      for(int i=0; i<k; i++){
        pv=v+i;
        vtv+=(*pv)*(*pv);
      }

      for(int i=0; i<n ; i++){
        pdv=dv+i;
        pu=u+i;
        *pu=(*pdv)/vtv;
      }

    //t'=ut*u (t'=t2)
      t2=prodottoutu(input);

    //if
    diff=t2-t1;
    vabs=abs(diff);
    if(vabs>=(teta*t2)){
      goto label;
    }

    //inserisci u come j-esima colonna di U
     pu=u;
        for(int i=j; i<n*h; i+=h){
            pmu=matu+i;
            *pmu=*pu;
            pu++;
        }
    //inserisci v come j-esima colonna di V
     pv=v;
        for(int i=j; i<k*h; i+=h){
            pmv=matv+i;
            *pmv=*pv;
            pv++;
        }
    //aggiorna il dataset
    uvt=prodottouvt(input);
    
    for(int i=0; i<n*k; i++){
        pd=(input->ds)+i;
        puvt=uvt+i;
        *pd-=*puvt;
    }
    }//for
  //  prova(input); //a cosa serve?
    //--------------------------------------------------
    // Calcola le matrici U e V
    // -------------------------------------------------
    input->U=matu;
    input->V=matv;

    
    
}

/*
*	K-d-Tree
* 	======================
*/
void kdtree(params* input) {
   //if(input->ds==0)
		//return NULL;
    int l;//livello corrente che ci aiuta a calcolare la dimensione su cui effettuare il taglio
	int c = l%input->k; //c è la dimensione su cui effettuar il taglio
    /*Andiamo ad individuare il punto mediano lungo la dimensione c. Procedo nel seguente modo:
    mi calcolo la somma di tutti gli elementi presenti nella dimensione su cui sto tagliando e mi ricavo la media,
    attraverso il risultato della media vado ad individuare l'elemento du cui devo tagliare, e in questo
    modo mi in un vettore vuoto il punto associato all'elemento su cui si effettua il taglio e lo restituisco alla fine,
    come scritto nello pseudo-codice del progetto*/
	float somma;
	float media;
	int i;
	int j;
    int z;
    int w;
    int n;//mi serve per poter scalare sulla riga del mediano
    /*con il for seguente parto da c e incremento ogni volta di k, in modo tale da andare a prendere
    sempre gli elementi sulla dimensione su cui sto lavorando*/ 
	for(i=c; i<(input->n)*(input->k); i+=input->k){
		somma+=*(input->ds)+i;//andiamo ad aggiungere ogni volta a somma il valore presente nella dimensione(o colonna se preferisci) c e riga i-esima
	media=somma/input->n;
    //trovato la media, vado a trovarmi esattamente il punto sulla dimensione c su cui lavorare
	int pc;//"puntatore" che ci indica la posizione del punto mediano sulla dimensione c
    float *P = (float*) malloc(sizeof(float)*input->k);
	for(j=c; j<(input->n)*(input->k); j+=input->k){
        if((media/(*(input->ds)+j))>=1.0 && (media/(*(input->ds)+j))<=1.25){
            pc=j;
            break;/*in questo modo, quando trovo quel valore che si avvicina a quello della media facendo il rapporto
            , mi salvo la sua posizione e fermo il for*/
        }
    }
    /*ora vado a riempire il vettore P che dovrò restituire alla fine.
    In questo caso rappresenterà la radice dell'albero*/
    for(z=c; z<(input->n)*(input->k); z+=input->k){
        if(pc==z){
            for(w=0; w<input->k; w++){
                P[w]=*(input->ds)+z+n;
                n++;
            }
        }
        break; //Esco dal for poichè ho già trovato il nodo radice del 
        //k-d-tree associato a D
    }
	MATRIX DS1; int x = 0;
    MATRIX DS2; int y = 0;
    for(i=c; i<(input->n)*(input->k); i+=input->k){
        if((*(input->ds)+i)<(*(input->ds)+pc)){
            for(j=0; j<input->k; j++){
                *(DS1)+j+x=*(input->ds)+i-(i%input->k)+j;
            }
            x+=input->k; //in questo modo evito di scrivere sulle celle già usate
            //inoltre posso usarlo per vedere la dimensione effettiva del nuovo dataset
            //visto come un array lungo
        }
         if((*(input->ds)+i)>=(*(input->ds)+pc)){
            for(j=0; j<input->k; j++){
                *(DS2)+j+y=*(input->ds)+i-(i%input->k)+j;
            }
            y+=input->k;
        }
        else if(pc==i){//pc è la "posizione" dove si trova il mediano
            continue;
        }
    }
    KDTREE sx = buildTree(input,DS1,l++,x);
    KDTREE dx = buildTree(input,DS2,l++,y);
    input->Radice=P;
}

KDTREE buildTree(params* input, MATRIX DT, int liv, int dimensione){
    int c = liv%input->k;
    int puntCorr;
    int i;
    int j;
    float somma;
    float media;
    int cont = 0;//mi conta quanti elementi sono presenti nel dataset attuale
    // e mi servirà anche per calcolare la media
    for(i=c; i<dimensione; i+=input->k){
        somma+=*(DT)+i;
        cont++;
    }
    media=somma/cont;
    for(j=c; j<dimensione; j+=input->k){
        if((media/(*(DT)+j))>=1.0 && (media/(*(DT)+j))<=1.25){
            puntCorr=j;
            break;
        }
    }
    MATRIX ds1; int dimX=0;
    MATRIX ds2; int dimY=0;
    for(i=c; i<dimensione; i+=input->k){
         if((*(DT)+i)<(*(DT)+puntCorr)){
            for(j=0; j<input->k; j++){
                *(ds1)+j+dimX==*(DT)+i-(i%input->k)+j;
            }
            dimX+=input->k;         
        }
         if((*(DT)+i)>=(*(DT)+puntCorr)){
            for(j=0; j<input->k; j++){
                *(ds2)+j+dimY==*(DT)+i-(i%input->k)+j;
            }
            dimY+=input->k;        
        }
        else if(puntCorr==i){
            continue;
        }
    }
    //NON SO COME RITORNARE DUE VALORI CONTEMPORANEAMENTE, RISOLVETE VOI SE NON VA BENE
    if(ds1!=NULL){
        return buildTree(input,ds1,liv++,dimX);
    }
    if(ds2!=NULL){
        return buildTree(input,ds2,liv++,dimY);
    }
}//buildTree

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

//funzioni aggiuntive

/*genera il dataset trasposto*/
MATRIX dsTrasposto(params* input){
  
  MATRIX data=input->ds;
  float *pd;
  int n=input->n;
  int k=input->k;

  dataT=calloc(n*k,sizeof(float));
  pdt=dataT;
  pd=data;
  for(int i=0; i<k; i++){
    for(int j=i; j<n*k; j+=k){
      pd=data+j;
      *pdt=*pd;
      pdt++;
    }
  }
return dataT;
}//dsTrasposto

/*centra il dataset sulla media*/
MATRIX centraDataSet(params* input){
  MATRIX dc;
  MATRIX data=input->ds;
  int n=input->n;
  int k=input->k;

  dc=calloc(n*k,sizeof(float));
  float *pdc;
  float media;
  float tmp;

  for(int i=0; i<k; i++){
    media=0.0;
    
      for(int m=i; m<n*k; m+=k){
        pdc=data+m;
        media+=*pdc;
      }
      media=media/n;
      for(int j=i; j<n*k; j+=k){
        pdc=data+j;
        tmp=(*pdc)-media;
        pdc=dc+j;
        *pdc=tmp;
      }
  }
return dc;
}//centraDataSet

MATRIX generau(params* input){
 
   int n=input->n;
   int k=input->k;
   MATRIX data=input->ds;
   float *pd;

   u=calloc(n,sizeof(float)); 
   pu=u; 
  for(int i=0; i<(n*k);i+=k){
        pd=data+i;   
        *pu=*pd;
        pu++;
    }

 return u;
}//generau

MATRIX prodottoDtu(params* input){
  int n=input->n;
  int k=input->k;

  MATRIX prod;
  prod= calloc(k,sizeof(float));

  float *pp;
  pp=prod;

  float fd=0.0;
  float fu=0.0;
  
  float tmp=0.0;

  for(int i=0; i<n*k; i+=n){
    tmp=0.0;
    
    
    for(int j=0; j<n; j++){
      pdt=dataT+i+j;
      pu=u+j;
      fd=*pdt;
      fu=*pu;
      tmp=fd*fu;
      *pp+=tmp;
      if(j==(n-1))pp++;
    }
  }
  return prod;
}//prodottoDtu

/*prodotto ut*u*/
float prodottoutu(params* input){
    int n=input->n;
  float utu=0.0;
    
    for(int i=0; i<n; i++){
      pu=u+i;
      utu+=(*pu)*(*pu);
    }
  return utu;
}//prodottoutu

/*prodotto tra dataset e v*/
MATRIX prodottoDv(params* input){
  int n=input->n;
  int k=input->k;

  MATRIX prod;
  prod= calloc(n,sizeof(float));

  MATRIX data =input->ds;
  float *pd;

  float *pp;
  pp=prod;

  float fd=0.0;
  float fv=0.0;
  
  float tmp=0.0;

  for(int i=0; i<n*k; i+=k){
    tmp=0.0;
    
    
    for(int j=0; j<k; j++){
      pd=data+i+j;
      pv=v+j;
      fd=*pd;
      fv=*pv;
      tmp=fd*fv;
      *pp+=tmp;
      if(j==(k-1))pp++;
    }
  }
  return prod;
}//prodottoDv

MATRIX prodottouvt(params* input){
    int n=input->n;
    int k=input->k;

  MATRIX uvt;
    float *puvt;
    uvt=calloc(n*k,sizeof(float));

    puvt=uvt;
    for(int i=0; i<n; i++){
        pu=u+i;
        for(int j=0; j<k; j++){
            pv=v+j;
            *puvt=(*pu)*(*pv);
            puvt++;
        }
    }
  return uvt;
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
        pca(input);
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
    
    if(input->kdtree){
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

