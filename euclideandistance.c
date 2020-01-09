#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float EuclideanDistance(Punto *p, Punto *q);

typedef struct point {

        int x,
            y;

} Punto; 

float EuclideanDistance(Punto *p, Punto *q) { //metodo per il calcolo della distanza tra due punti

      return sqrt((q->x - p->x)*(q->x - p->x) + (q->y - p->y)*(q->y - p->y)); //radice quadrata di (qx-px)^2 + (qy-qy)^2
}

int main() { //main di prova 

    Punto *p,
          *q;
   
    p = (Punto*)malloc(sizeof(Punto));
    q = (Punto*)malloc(sizeof(Punto));
     
    float res;
 
    printf("Primo punto\n");
    printf("x: ");
    scanf("%d", &p->x); 
    printf("y: ");
    scanf("%d", &p->y); 

    printf("Secondo punto\n");
    printf("x: ");
    scanf("%d", &q->x); 
    printf("y: ");
    scanf("%d", &q->y); 

    res = EuclideanDistance(p,q);
    printf("Distanza = %f", res);

    free(p);
    free(q);

 return(0);

};