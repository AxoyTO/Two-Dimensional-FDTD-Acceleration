cmake_minimum_required(VERSION 3.21)
project(.)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS -fopenmp)

add_executable(Supercomputer omp_fdtd-2d.c mpi_fdtd-2d.c)