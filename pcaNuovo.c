#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>

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
MATRIX v;
float *pv;



MATRIX generads(int n, int k);
void stampads();
MATRIX generau();
MATRIX dsTrasposto();
void stampadst();
MATRIX prodottoDtu();
MATRIX prodottoDv();



int main(int argc, char const *argv[]){
    n=5;
    k=3;
    h=2;

     data=generads(n,k); 
     stampads();
     
     u=generau();
    
     dataT=dsTrasposto();
     stampadst();
    
     MATRIX dTu=prodottoDtu();
   
   printf("Dt*u:\n");
   printf("\n");
   //stampa prodotto Dt*u
    float *pdtu;
    pdtu=dTu;
    for(int i=0; i<k; i++){
      pdtu=dTu+i;
        printf("%6.1f \n",*pdtu);
       
    }

    printf("\n");
    
    //calcola ut * u
    printf("\n");
    float t1=0.0;
    
    for(int i=0; i<n; i++){
      pu=u+i;
      t1+=(*pu)*(*pu);
    }
    printf("ut*u: %6.1f\n", t1);
    printf("\n");

    
    
    pv=v;
    v=calloc(k, sizeof(float));
    //for (int j=0; j<h; j++){
        //calcola il vettore dei load
        printf("VETTORE v\n");
        printf("\n");
        for(int i=0; i<k; i++){
          pdtu=dTu+i;
          pv=v+i;
          *pv=(*pdtu)/t1;
          printf("%6.2f\n",*pv);
        }
        //normalizza il vettore dei load
       float norma=0.0;
       for(int i=0; i<k; i++){
         pv=v+i;
         norma+=pow(*pv,2);
       }
       norma=sqrt(norma);
       printf("\n");
       printf("norma(v): %6.2f\n",norma);
       printf("\n");
       printf("vettore v normalizzato:\n");
       printf("\n");
      for(int i=0; i<k; i++){
        pv=v+i;
        *pv=(*pv)/norma;
        printf("%6.2f\n",*pv);
      }
      printf("\n");

      //aggiorna il vettore degli score
      
      MATRIX dv=prodottoDv();
      float *pdv;
      printf("D*v:\n");
      printf("\n");
      for(int i=0; i<n; i++){
          pdv=dv+i;
          printf("%6.2f\n",*pdv);
      }

      float vtv=0.0;
      for(int i=0; i<k; i++){
        pv=v+i;
        vtv+=(*pv)*(*pv);
      }
      printf("\n");
      printf("vt*v: %6.1f\n", vtv);
      printf("\n");
     

    
    for(int i=0; i<n ; i++){
      pdv=dv+i;
      pu=u+i;
      *pu=(*pdv)/vtv;
    }

    printf("VETTORE u aggiornato:\n");
    printf("\n");
    for(int i=0; i<n; i++){
      pu=u+i;
      printf("%6.2f\n",*pu);
    }
    
    printf("\n");
    float t2=0.0;

    for(int i=0; i<n; i++){
      pu=u+i;
      t2+=(*pu)*(*pu);
    }
    printf("t'=ut*u: %6.1f\n", t2);
    printf("\n");
}//main

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

  return u;
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
        printf("%6.1f ",*pdt); //stampa il contenuto della locazione puntata da pdt
        if(z%(n-1)==0 && z!=0){
          printf("\n");
        }
        j++;
      }
    }
    
    printf("\n");
    
}//stampadst


MATRIX prodottoDtu(){
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

MATRIX prodottoDv(){
  MATRIX prod;
  prod= calloc(n,sizeof(float));

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




  


