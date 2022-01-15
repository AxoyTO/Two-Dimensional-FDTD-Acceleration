#include <mpi.h>
#include "fdtd-2d.h"

int main(int argc, char **argv){
    int rank, comm_size;
    int first_row, last_row;
    MPI_Request requests[2];
    MPI_Status statuses[2];

    double EX[NX][NY];
    double EY[NX][NY];
    double HZ[NX][NY];
    int i, j, it;
    int tmax = TMAX;
    int nx = NX;
    int ny = NY;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Barrier(MPI_COMM_WORLD);
    first_row = (rank * nx) / comm_size;
    last_row = (((rank + 1) * nx) / comm_size) - 1;

    //init_array
    for(i = first_row; i <= last_row; i++){
        for(j = 0; j < ny; j++){
            EX[i][j] = ((double) i * (j + 1)) / nx;
            EY[i][j] = ((double) i * (j + 2)) / ny;
            HZ[i][j] = ((double) i * (j + 3)) / nx;
        }
    }

    double start = MPI_Wtime();

    //kernel_fdtd_2d
    for(it = 1; it <= tmax; it++){
        //Row -> Ranks
        if(rank){
            MPI_Irecv(HZ[first_row - 1], NY, MPI_DOUBLE, rank - 1, 215, MPI_COMM_WORLD, requests);
        }
        if(rank != comm_size - 1){
            MPI_Isend(HZ[last_row], NY, MPI_DOUBLE, rank + 1, 215, MPI_COMM_WORLD, requests + 1);
        }

        int count = 2, shift = 0;
        if(!rank){
            count -= 1;
            shift = 1;
        }
        if(rank == comm_size - 1){
            count -= 1;
        }
        MPI_Waitall(count, requests + shift, statuses);

        //EY, EX
        if(!rank){
            for(j = 0; j < ny; j++)
                EY[0][j] = (double) it - 1;
        }

        for(i = first_row; i <= last_row; i++){
            if(i == 0)
                continue;
            for(j = 0; j < ny; j++)
                EY[i][j] = EY[i][j] - 0.5 * (EY[i][j] - HZ[i - 1][j]);
        }

        for(i = first_row; i <= last_row; i++){
            for(j = 1; j < ny; j++)
                EX[i][j] = EX[i][j] - 0.5 * (HZ[i][j] - HZ[i][j - 1]);
        }


        //Row -> ranks
        if(rank != comm_size - 1){
            MPI_Irecv(EY[last_row + 1], NY, MPI_DOUBLE, rank + 1, 216, MPI_COMM_WORLD, requests + 1);
        }

        if(rank){
            MPI_Isend(EY[first_row], NY, MPI_DOUBLE, rank - 1, 216, MPI_COMM_WORLD, requests);
        }

        count = 2, shift = 0;
        if(!rank){
            count -= 1;
            shift = 1;
        }
        if(rank == comm_size - 1){
            count -= 1;
        }
        MPI_Waitall(count, requests + shift, statuses);
        //HZ
        for(i = first_row; i <= last_row; i++){
            if(i == nx)
                continue;
            for(j = 0; j < ny - 1; j++)
                HZ[i][j] = HZ[i][j] - 0.7 * (EX[i][j + 1] - EX[i][j] + EY[i + 1][j] - EY[i][j]); //need next row EY
        }

    }
    double end = MPI_Wtime();
    if(rank == 0){
        printf("==========================\n");
        printf("PROCESSES: %d\n", comm_size);
        printf("Time in seconds = %0.6lf\n", end - start);
        printf("==========================\n");
    }
    MPI_Finalize();
    return 0;
}