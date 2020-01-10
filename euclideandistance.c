#include <stdio.h>
#include <stdlib.h>
#include <math.h>


float EuclideanDistance(float *p, float *q) { //metodo per il calcolo della distanza tra due punti
      float somma=0;
      for(int i=0, i<k, i++){
            int var = ((q+i)-(p+i))*((q+i)-(p+i)); //differenza delle k dimensioni ed elevamento al 2
            somma+=var;
      }
      return sqrt(somma);
}
