#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"


double blockSize;
int numBlocks;

void buildBlocks(std::vector<std::vector<particle_t>>& blocks, particle_t* particles, int n, double boxLength)
{
    double cutoff = 0.01;
    blockSize = cutoff * 5;  
    numBlocks = int(boxLength / blockSize)+1; 

    blocks.resize(numBlocks * numBlocks);

    for (int i = 0; i < n; i++)
    {
        int x = int(particles[i].x / blockSize);
        int y = int(particles[i].y / blockSize);
        blocks[x*numBlocks + y].push_back(particles[i]);
    }
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    std::vector<std::vector<particle_t>> particle_blocks;
    std::vector<particle_t> block;

    double size = set_size( n );
    init_particles( n, particles);
    
    // init blocks
    buildBlocks(particle_blocks, particles, n, size);
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    
    for (int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;
     
        for (int i = 0; i < numBlocks; i++)
        {
            for (int j = 0; j < numBlocks; j++)
            {
                std::vector<particle_t>& vec1 = particle_blocks[i*numBlocks+j];
                for (int k = 0; k < vec1.size(); k++)
                    vec1[k].ax = vec1[k].ay = 0;
                for (int dx = -1; dx <= 1; dx++) // check all 8 neighbor blocks
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        if (i + dx >= 0 && i + dx < numBlocks && j + dy >= 0 && j + dy < numBlocks)
                        {
                            std::vector<particle_t>& vec2 = particle_blocks[(i+dx) * numBlocks + j + dy];
                            for (int k = 0; k < vec1.size(); k++) // iterate through both vectors from neighbor blocks
                                for (int l = 0; l < vec2.size(); l++)
                                    apply_force( vec1[k], vec2[l], &dmin, &davg, &navg);
                        }
                    }
                }
            }
        }
 
        for (int i = 0; i < numBlocks; i++)
        {
            for(int j = 0; j < numBlocks; j++)
            {
                std::vector<particle_t>& vec = particle_blocks[i * numBlocks + j];
                int vec_len = vec.size();
		int k = 0;
                for(; k < vec_len; )
                {
                    move( vec[k] );
                    int x = int(vec[k].x / blockSize);  // get block position
                    int y = int(vec[k].y / blockSize);
                    if (x == i && y == j)  // if still in original block do nothing
                        k++;
                    else // otherwise remove from current block and store it in the appropriate block ( temporary for now)
                    {
                        block.push_back(vec[k]);
                        vec[k] = vec[--vec_len];
                    }
                }
                vec.resize(k);
            }
        }
	// store particles in temporary block (from movement) into appropriate blocks
        for (int i = 0; i < block.size(); i++) 
        {
            int x = int(block[i].x / blockSize);
            int y = int(block[i].y / blockSize);
            particle_blocks[x*numBlocks+y].push_back(block[i]);
        }
        block.clear();
        
        if( find_option( argc, argv, "-no" ) == -1 )
        {
            if (navg) 
            {
                absavg +=  davg/navg;
                nabsavg++;
            }
              
            if (dmin < absmin) 
                absmin = dmin;

            if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, particles );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
        if (nabsavg) absavg /= nabsavg;
        // 
        //  -The minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");     

    //
    // Printing summary data
    //
    if( fsum ) 
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    
    if( fsave )
        fclose( fsave );
    
    return 0;
}
