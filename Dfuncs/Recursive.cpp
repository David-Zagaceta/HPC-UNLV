#include <iostream>
#include <complex>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "factorial.h"

std::complex <double> WignerDRec(std::complex<double> Ra, std::complex<double> Rb, int twolmax, int* idx, std::complex<double>* arr);
double read_timer();

int main ()
{
 double x = 1;
 double y = 1;
 double z = 1;
 double r = sqrt(x*x + y*y + z*z);
 double omega = 0.32;

 std::complex<double> a(cos(omega),sin(omega)/r);
 std::complex<double> b(sin(omega)/r*y,sin(omega)/r*x);
 std::complex<double> result;

 int lmax = 64;

 double start = read_timer();
 int twolmax = 2*lmax;
 int ldim = twolmax + 1;

 int idx[ldim];

 int idx_count = 0;

 for( int l = 0; l <= twolmax; ++l){
  idx[l] = idx_count;
  for( int mb = 0; mb <= l; ++mb)
    for( int ma = 0; ma <= l; ++ma)
      idx_count++;
 }
 std::complex<double> arr[idx_count];

 WignerDRec(a, b, twolmax, idx, arr);

 double end = read_timer();
 std::cout << end-start << std::endl;


  return 0;
}
std::complex<double> WignerDRec(std::complex<double> Ra, std::complex<double> Rb, int twolmax, int* idx, std::complex<double>* arr){

    std::complex<double> one(1,0);
    std::complex<double> zero(0,0);
    arr[0] = one;

    for( int l=1; l<=twolmax; ++l){
        int llu = idx[l];
        int llup = idx[l-1];
        for( int mb = 0; 2*mb <= l; ++mb){
            arr[llu] = 0;
            for( int ma = 0; ma < l; ++ma){
                double rootpq = sqrt(factorial(l-ma)/static_cast<double>(factorial(l-mb)));
                arr[llu] += rootpq*std::conj(Ra)*arr[llup];
                rootpq = sqrt(factorial(ma+1)/static_cast<double>(factorial(l-mb)));
                arr[llu+1] = -rootpq*std::conj(Rb)*arr[llup];
                llu++;
                llup++;
            }
            llu++;
        }

        llu = idx[l];
        llup = llu+(l+1)*(l+1)-1;
        int mbpar = 1;
        for( int mb = 0; 2*mb <= l; ++mb){
            int mapar = mbpar;
            for( int ma = 0; ma <= l; ++ma){
                if( mapar == 1){
                    arr[llup] = std::conj(arr[llu]);
                }
                else{
                    arr[llup] = -std::conj(arr[llu]);
                }
                mapar = -mapar;
                llu++;
                llup--;
            }
            mbpar = -mbpar;
       }
    }
}

double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}
