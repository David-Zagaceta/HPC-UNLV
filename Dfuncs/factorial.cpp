#include <math.h>
#include "factorial.h"

double factorial( double N){
        int n = round(N);
	return (n<0 ||n==1 || n==0) ? 1: factorial(n-1) * n;
}
