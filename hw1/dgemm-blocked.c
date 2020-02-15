/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = icc
OPT = -Ofast -mavx2
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* restrict C)
{
  /* For each row i of A */
  int numUnrolls = 8;
  int i,j,k,n;
  double cij;
  __m256d a,b;
  __m256d c[numUnrolls];
  // Taking into account the 4 data points loaded from SIMD
  // and the number of loops we are unrolling
    /* For each column j of B */ 
  for (j = 0; j < N; ++j){
    for (i = 0; i < M-M%(4*numUnrolls); i+=4*numUnrolls){
      /* Get C(i,j) */
      for (n=0; n<numUnrolls; ++n)
        c[n] = _mm256_loadu_pd(C+i+j*lda+n*4); 
      /* Product of a partial column of A
         With an element of B
         to get a partial sum of 1 column
         of C since we are in column major*/
      for (k = 0; k < K; ++k){
        b = _mm256_broadcast_sd(B+k+j*lda);
        for (n=0; n<numUnrolls; ++n){
          a = _mm256_loadu_pd(A+i+k*lda+n*4);
          c[n] = _mm256_fmadd_pd(a,b,c[n]);
          }
        }
      for (n=0; n<numUnrolls; ++n)
        _mm256_store_pd(C+i+j*lda+n*4, c[n]);
    }
  }

  // clean up remaining elements with naive matrix mult
  for ( ; i<M; ++i)
    for (j=0; j<N; ++j){
      cij = C[i+j*lda];
      for (k=0; k<K; ++k)
        cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
