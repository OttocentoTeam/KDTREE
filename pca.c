#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>

#define MATRIX float*

MATRIX data;            //DATASET
float *pd;              //puntatore al  dataset
MATRIX dataT;           //DataSet trasposto
float *pdt;             //puntatore al DataSet trasposto
int n;                  //numero righe del dataset
int k;                  //numero colonne del dataset
int h;                  //numero di componenti principali che si vogiono ottenere
MATRIX u;               //vettore degli score u
float *pu;              //puntatore a u
MATRIX v;               //vettore dei load v
float *pv;              //puntatore a v



MATRIX generads(int n, int k);
void stampads();
MATRIX generau();
MATRIX dsTrasposto();
void stampadst();
MATRIX prodottoDtu();
MATRIX prodottoDv();
MATRIX centraDataSet();
MATRIX nipals();
float prodottoutu();
MATRIX prodottouvt();



int main(int argc, char const *argv[]){
    n=10;
    k=4;
    h=2;

     data=generads(n,k); 
     
     dataT=dsTrasposto();    
    
}//main

/*genera il dataset*/
MATRIX generads(int n, int k){
  MATRIX d;
  d=calloc(n*k,sizeof(float)); //allocazione di n*k elementi di tipo float
 
  
    /*
      Il riempimento del dataset è banale,
      si inseriscono nelle  locazioni di memoria 
      dedicate al dataset i numeri da 0 a (n*k)-1
    */
    for(int i=0; i<n*k;i++){
        pd=d+i; //pd punta alla i-esima locazione del dataset (data[0][0])
        *pd=i;     //inserisci il valore di i nella locazione puntata da pd
    }

    
    
  return d;
}//generads


/*genera il vettore u e lo stampa*/
MATRIX generau(){
  /*
   Voglio inserire gli elementi della prima colonna del dataset nel vettore u.
   Poichè la matrice e nxk, allora, u avrà dimensione n
   */
    MATRIX u;
   u=calloc(n,sizeof(float)); 
   
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



  return u;
}//generau


/*genera il dataset trasposto*/
MATRIX dsTrasposto(){
  MATRIX dt;
 
  dt=calloc(n*k,sizeof(float));
  pdt=dt;
  pd=data;
  for(int i=0; i<k; i++){
    for(int j=i; j<n*k; j+=k){
      pd=data+j;
      *pdt=*pd;
      pdt++;
    }
  }
return dt;
}//dsTrasposto


/*prodotto tra dattaset trasposto e u*/
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


/*prodotto tra dataset e v*/
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


/*centra il dataset sulla media*/
MATRIX centraDataSet(){
  MATRIX dc;
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


/*prodotto ut*u*/
float prodottoutu(){
  float utu=0.0;
    
    for(int i=0; i<n; i++){
      pu=u+i;
      utu+=(*pu)*(*pu);
    }
  return utu;
}

MATRIX prodottouvt(){
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


/*algoritmo PCA
*Input: dataset D di dimensione (n × k), numero h < k di componenti principali desiderate
*Output: matrice (n × h) degli score U, matrice (k × h) dei load V
*/
MATRIX nipals(){
  float norma;
  float t1,t2;
  float diff;
  float vabs;

  MATRIX dtu;
  float *pdtu;
  float utu;

  MATRIX dv;
  float *pdv;

  MATRIX uvt;
  float *puvt;

  MATRIX matu;  //matrice U degli score
  float *pmu;   //puntatore ad U
  MATRIX matv;  //matrice V dei load
  float *pmv;   //punatore a V

  uvt=calloc(n*k,sizeof(float));
  matu=calloc(n*h,sizeof(float));
  matv=calloc(k*h,sizeof(float));


  float teta=0.00000001;  //imposta la soglia teta a 1e-8
  data=centraDataSet();   //centra il DataSet sulla media
  u=generau();            //sia u la prima colonna di D
 

   v=calloc(k,sizeof(float));
   float *pv;
   float vtv;
      

  for(int j=0; j<h; j++){
    //calcola vettore dei load v
      
      label:  dtu=prodottoDtu();
              utu=prodottoutu();
    
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
      t1=prodottoutu();

    //aggiorna il vettore degli score
      dv=prodottoDv();
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
      t2=prodottoutu();

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
    uvt=prodottouvt();
    
    for(int i=0; i<n*k; i++){
        pd=data+i;
        puvt=uvt+i;
        *pd-=*puvt;
    }
  }//for

return matu,matv; //si può fare???
}//nipals
  



