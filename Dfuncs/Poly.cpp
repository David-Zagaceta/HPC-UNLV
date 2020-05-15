#include <iostream>
#include <complex>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "factorial.h"
#include "omp.h"

std::complex <double> WignerDPoly(std::complex<double> Ra, std::complex<double> Rb, int twol, int twomp, int twom);
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
 int numthreads;
 #pragma omp parallel
 {
     #pragma omp master
     {
         numthreads = omp_get_num_threads();
     }
     #pragma omp for
     for( int twol = 0; twol <= 2*lmax; ++twol)
       for( int twomp = -twol; twomp <= twol; ++twomp)
         for( int twom = -twol; twom <= twol; ++twom)
           result = WignerDPoly(a,b,twol,twomp, twom);
 }
 double end = read_timer();
 std::cout << end-start << std::endl;


  return 0;
}
std::complex <double> WignerDPoly(std::complex<double> Ra, std::complex<double> Rb, int twol, int twomp, int twom){
    double ra = std::abs(Ra);
    double rb = std::abs(Rb);
    double phia = std::arg(Ra);
    double phib = std::arg(Rb);

    double epsilon = std::pow(10,-15);
    std::complex <double> zero(0,0);

    if( ra <= epsilon){
        if( twomp != -twom || std::abs(twomp) > twol || std::abs(twom) > twol){
            return zero;
        }
        else{
            if( (twol-twom)%4 == 0){
                return std::pow(Rb, twom);
            }
            else{
                return -std::pow(Rb, twom);
            }
        }
    }
    
    else if( rb <= epsilon){
        if( twomp != -twom || std::abs(twomp) > twol || std::abs(twom) > twol){
            return zero;
        }
        else{
            return std::pow(Ra, twom);
        }
    }

    else if( ra < rb){

        double x = -ra*ra/rb/rb;
        if( std::abs(twomp) > twol || std::abs(twom) > twol){
            return zero;
        }

        else{
            std::complex<double> Prefactor = std::polar(std::pow(ra, (twom+twomp)/2) * std::pow(rb, twol-(twom+twomp)/2),
                                                        phia*(twom+twomp)/2 + phib*(twom-twomp)/2);

            Prefactor *= sqrt(factorial((twol+twom)/2)*factorial((twol-twom)/2)*factorial((twol+twomp)/2)*factorial((twol-twomp)/2));

            double l = twol/2;
            double mp = twomp/2;
            double m = twom/2;
            int kmax = round(std::min(l-mp, l-m));
            int kmin = round(std::max(0.0,-mp-m));

	    if( (twol-twom)%4 != 0){
                Prefactor *= -1;
            }

            double Sum = 1/factorial(kmax)/factorial((twol-twom)/2-kmax)/factorial((twomp+twom)/2+kmax)/factorial((twol-twomp)/2-kmax);

            for( int k=kmax-1; k > kmin; --k){
                Sum *= x;
                Sum += 1/factorial(k)/factorial((twol-twom)/2-k)/factorial((twomp+twom)/2+k)/factorial((twol-twomp)/2-k);
            }
            Sum *= std::pow(x,kmin);
            return Prefactor*Sum;

        }

    }

    else{
        double x = -rb*rb/ra/ra;
        if( std::abs(twomp) > twol || std::abs(twom) > twol){
            return zero;
        }

        else{
            std::complex<double> Prefactor = std::polar(std::pow(ra,twol-(twom-twomp)/2) * std::pow(rb, (twom-twomp)/2),
                                                        phia*(twom+twomp)/2 + phib*(twom-twomp)/2);

            Prefactor *= sqrt(factorial((twol+twom)/2)*factorial((twol-twom)/2)*factorial((twol+twomp)/2)*factorial((twol-twomp)/2));

            double l = twol/2;
            double mp = twomp/2;
            double m = twom/2;
            int kmax = round(std::min(l+mp, l-m));
            int kmin = round(std::max(0.0,mp-m));

            double Sum = 1/factorial(kmax)/factorial((twol-twom)/2-kmax)/factorial((-twomp+twom)/2+kmax)/factorial((twol+twomp)/2-kmax);

            for( int k=kmax-1; k > kmin; --k){
                Sum *= x;
                Sum += 1/factorial(k)/factorial((twol-twom)/2-k)/factorial((-twomp+twom)/2+k)/factorial((twol+twomp)/2-k);
            }
            Sum *= std::pow(x,kmin);
            return Prefactor*Sum;

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
