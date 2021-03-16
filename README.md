# Compilation

To compile this project, run
`./{Project directory}/Parallelized/run_test.sh -n {Number of MPI Processes} -N {Number of nodes} [-t|optional] {mpi || mpi_omp} [-o|(used when -t mpi_omp is selected)] {Number of OpenMP threads}`

# [Optional] Profiling

You can visualize the performance of the application.

-   To run the visualizer, first run this command
`source {Project directory}/profiling/scripts/run_evaluations.sh`
-   Then you can run this command to run the visualizer
`profiler -h` to show how to pass arguments.
