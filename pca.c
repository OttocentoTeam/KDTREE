#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

#define MATRIX float*

MATRIX data;            //DATASET
float *pd;              //puntatore al  dataset
int n;                  //numero righe
int k;                  //numero colonne
float teta=0.00000001;  //imposta la soglia teta a 1e-8
int h;                  //numero di componenti principali che si vogiono ottenere
MATRIX u;
float *pu;
MATRIX dataT;           //DATASET trasposto
float *pdt;



MATRIX generads(int n, int k);
void stampads();
MATRIX generau();
MATRIX dsTrasposto();
void stampadst();
MATRIX prodottoDtu();

int main(int argc, char const *argv[]){
    n=5;
    k=6;

     data=generads(n,k); //EASTER EGG 50x16=800
     stampads();
     
     u=generau();
    
     dataT=dsTrasposto();
     stampadst();


  


}

/*genera il dataset*/
MATRIX generads(int n, int k){
  
  data=calloc(n*k,sizeof(float)); //allocazione di n*k elementi di tipo float
 
  printf("GENERAZIONE DATASET...\n");
  printf("\n");
    /*
      Il riempimento del dataset è banale,
      si inseriscono nelle  locazioni di memoria 
      dedicate al dataset i numeri da 0 a (n*k)-1
    */
    for(int i=0; i<n*k;i++){
        pd=data+i; //pd punta alla i-esima locazione del dataset (data[0][0])
        *pd=i;     //inserisci il valore di i nella locazione puntata da pd
    }

    
    
  return data;
}//generads

/*stampa il DATASET*/
void stampads(){
    printf("STAMPA DATASET...\n");
    printf("\n");
   
    int j=0;
    while(j<n*k){
      for(int z=0;z<k;z++){
        pd=data+j; //pd punta alla j-esima locazione del dataset
        printf("%6.1f ",*pd); //stampa il contenuto della locazione puntata da pd
        if(z%(k-1)==0 && z!=0){
          printf("\n");
        }
        j++;
      }
    }
    
    printf("\n");
}//stampads

MATRIX generau(){
  /*
   Voglio inserire gli elementi della prima colonna del dataset nel vettore u.
   Poichè la matrice e nxk, allora, u avrà dimensione n
   */
   u=calloc(n,sizeof(float)); 
   printf("GENERAZIONE VETTORE u...\n");
   printf("\n");
   
   pu=u; //pu punta al primo elemento di u (u[0])
  for(int i=0; i<(n*k);i+=k){
        /*
        pd ad ogni iterazione punterà 3 elementi in avanti
        1 iter : pd = data+0 (data[0][0])
        2 iter : pd = data+10(data[1][0])
        ...*/
        pd=data+i; 
        //inserisco nella locazione puntata da pu il contenuto della locazione puntata da pd
        *pu=*pd;
        //sposto pu al successivo elemento di u
        pu++;
    }

printf("STAMPA VETTORE u...\n");
    printf("\n");
    
  for(int i=0;i<n;i++){
        pu=u+i;
        printf("%6.1f ",*pu);
        printf("\n");
    }
 
  printf("\n");
}//generau


MATRIX dsTrasposto(){

  printf("GENERAZIONE DATA SET TRASPOSTO...\n");
  printf("\n");
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

void stampadst(){
   printf("STAMPA DATASET TRASPOSTO...\n");
    printf("\n");
   
   int j=0;
    while(j<n*k){
      for(int z=0;z<n;z++){
        pdt=dataT+j; //pd punta alla j-esima locazione del dataset
        printf("%6.1f ",*pdt); //stampa il contenuto della locazione puntata da pd
        if(z%(n-1)==0 && z!=0){
          printf("\n");
        }
        j++;
      }
    }
    
    printf("\n");
    
}//stampads

MATRIX prodottoDtu(){
  //NOTA BENE: la dimensione del dataset trasposto è k*n!
  MATRIX prod;
  prod=calloc(k,sizeof(float));
  float *pp; //puntatore al prodotto
  pp=prod;

  float f1;
  float f2;

  float tmp;

  for(int i=0; i<n*k;i+=n){
    pdt=dataT+i;
    pp++;
    for(int j=0; j<n; j++){
        pu=u+j;
        f1=*pdt;
        f2=*pu;
        tmp=f1*f2;
        pdt++;
        *pp+=tmp;
    }
  }

  return prod;
}


  

