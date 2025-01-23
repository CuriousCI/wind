#!/bin/bash

benchmark=false
debug=false
integration=false
perf=false
scaling=false
valgrind=false

while getopts ":bgipsv" option; do
    case $option in
        b) benchmark=true;;
        g) debug=true;;
        i) integration=true;;
        p) perf=true;;
        s) scaling=true;;
        v) valgrind=true;;
    esac
done

# ---

rows=250 columns=250 max_iter=2000 var_threshold=0.5
inlet_pos=0 inlet_size=$rows
particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=0.5
particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=0.5
short_rnd1=$((RANDOM % 10000 + 1)) short_rnd2=$((RANDOM % 10000 + 1)) short_rnd3=$((RANDOM % 10000 + 1))
rows=1000

update_args() {
    args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"
}

update_args

# ---

if $benchmark; then
    make wind_seq wind_mpi wind_omp wind_cuda wind_pthread wind_pthread_2
    ./wind_seq $args
    ./wind_omp $args
    ./wind_pthread $args
    ./wind_pthread_2 $args
    mpirun ./wind_mpi $args
    # ./wind_cuda $args
fi

if $perf; then 
    make wind_seq wind_omp wind_pthread wind_pthread_2
    perf stat $bonus ./wind_seq $args 
    perf stat $bonus ./wind_omp $args 
    perf stat $bonus ./wind_pthread $args 
    perf stat $bonus ./wind_pthread_2 $args 
fi

if $debug; then
    make wind_mpi_debug
    mpirun --get-stack-traces ./wind_mpi_debug $args
    # mpirun -np 4 alacritty --hold -e gdb ./wind_mpi_debug $args 
    # make wind_pthread_2_debug
    # gdb --args ./wind_pthread_2_debug $args
    # DEBUGGING MPI # pmpiexec -n 4 ./test : -n 1 ddd ./test : -n 1 ./test
fi

if $integration; then
    rows=100 columns=100 max_iter=100 inlet_size=$rows
    particles_f_band_size=$((rows-1)) particles_m_band_size=$((rows-1))
    echo "$short_rnd1 $short_rnd2 $short_rnd3"

    update_args
    make clean
    make debug
    ./wind_seq $args > seq.txt
    ./wind_omp $args > omp.txt
    ./wind_pthread $args > pthread.txt
    ./wind_pthread_2 $args > pthread_2.txt
    mpirun ./wind_mpi $args > mpi.txt
fi

if $valgrind; then
    rows=100
    columns=100
    max_iter=100
    var_threshold=0.0

    inlet_pos=0
    inlet_size=$rows

    particles_f_band_pos=1
    particles_f_band_size=99
    particles_f_density=0.5 

    particles_m_band_pos=1
    particles_m_band_size=99
    particles_m_density=0.5

    update_args

    make wind_seq_debug wind_pthread_debug wind_omp_debug
    rm target/* -f

    valgrind -s --tool=massif ./wind_seq_debug $args
    valgrind -s --tool=callgrind ./wind_seq_debug $args
    valgrind -s --tool=cachegrind ./wind_seq_debug $args

    for f in *.out.*; do 
        mv $f target/$f.seq
    done

    valgrind -s --log-file=./target/helgrind.omp.txt --tool=helgrind ./wind_omp_debug $args
    valgrind -s --tool=massif ./wind_omp_debug $args
    valgrind -s --tool=callgrind ./wind_omp_debug $args
    valgrind -s --tool=cachegrind ./wind_omp_debug $args

    for f in *.out.*; do 
        mv $f target/$f.omp
    done

    valgrind -s --log-file=./target/helgrind.pthread.txt --tool=helgrind ./wind_pthread_debug $args 
    valgrind -s --tool=massif ./wind_pthread_debug $args 
    valgrind -s --tool=callgrind ./wind_pthread_debug $args
    valgrind -s --tool=cachegrind ./wind_pthread_debug $args

    for f in *.out.*; do 
        mv $f target/$f.pthread
    done

    # ----

    for f in target/massif.out.*; do 
        ms_print $f > $f.txt
    done

    for f in target/callgrind.out.*; do 
        callgrind_annotate $f > $f.txt
    done

    for f in target/cachegrind.out.*; do 
        callgrind_annotate $f > $f.txt
    done
fi

get_time="awk 'NR==2 {printf \"%s \", \$2}' | tr -d '\n'"
get_rest="awk 'NR==3 {printf \"%s \", \$2}' | tr -d '\n'" # TODO

matrix_ratio() {
    local prog=$1
    local file=$2
    local iter=$3

    # 100000000000 rows=100000000 columns=10
    particles_f_band_size=0 particles_m_band_size=0 inlet_size=0
    echo "rows,columns,time" >> $file
    for ((i=1; i<100000000; i*=2)); do
        rows=$((100000000/i)) columns=$i
        update_args

        for ((j=0; j<$iter; j++)); do
            echo "$prog - rows: $rows - columns: $columns - iter: $j"
            echo -n "$rows,$columns, " >> $file 
            eval $prog $args | eval $get_time >> $file
            echo >> $file 
        done
    done
}

matrix_size() {
    local prog=$1
    local file=$2
    local iter=$3

    # 10000
    particles_f_band_size=0 particles_m_band_size=0 inlet_size=0
    echo "cells,time" >> $file
    for ((i=1; i<10000; i*=2)); do
        rows=$i columns=$i inlet_size=$i
        update_args

        for ((j=0; j<$iter; j++)); do
            echo "$prog - rows: $rows - columns: $columns - iter: $j"
            echo -n "$((rows*columns)), " >> $file 
            eval $prog $args | eval $get_time >> $file
            echo >> $file 
        done
    done
}

particles_m() {
    local prog=$1
    local file=$2
    local iter=$3

    particles_f_band_size=0 particles_m_band_size=$((rows-1))
    echo "density,time" >> $file
    for particles_m_density in $(seq 0 0.1 1); do
        update_args
        for ((j=0; j<$iter; j++)); do
            echo "$prog - density: $particles_m_density - iter: $j"
            echo -n "$particles_m_density," >> $file 
            eval $prog $args | eval $get_time >> $file
            echo >> $file 
        done
    done
}

particles_f() {
    local prog=$1
    local file=$2
    local iter=$3

    particles_f_band_size=$((rows-1)) particles_m_band_size=0
    echo "density,time" >> $file
    for particles_f_density in $(seq 0 0.1 1); do
        update_args
        for ((j=0; j<$iter; j++)); do
            echo "$prog - density: $particles_f_density - iter: $j"
            echo -n "$particles_f_density," >> $file 
            eval $prog $args | eval $get_time >> $file
            echo >> $file 
        done
    done
}

inlet_perf() {
    local prog=$1
    local file=$2
    local iter=$3

    particles_f_band_size=0 particles_m_band_size=0
    echo "inlet_size,time" >> $file
    for inlet_size in $(seq 0 100 $rows); do
        update_args
        for ((j=0; j<$iter; j++)); do
            echo "$prog - inlet_size: $inlet_size - iter: $j"
            echo -n "$inlet_size," >> $file 
            eval $prog $args | eval $get_time >> $file
            echo >> $file 
        done
    done
}

system() {
    local prog=$1
    local file=$2
    local iter=$3

    # 10000
    echo "rows,time" >> $file
    particles_f_density=0.5
    particles_m_density=0.5
    # 10000
    for ((i=1; i<=4096; i*=2)); do
    # for ((i=1; i<=2048; i*=2)); do
        rows=$i columns=$i inlet_size=$i
        particles_f_band_size=$((rows-1)) particles_m_band_size=$((rows-1)) inlet_size=$((rows-1))
        update_args

        for ((j=0; j<$iter; j++)); do
            echo "$prog - rows: $rows - columns: $columns - iter: $j"
            echo -n "$((rows)), " >> $file 
            eval $prog $args | eval $get_time >> $file
            echo >> $file 
        done
    done

}

if $scaling; then
    rm target/*.csv
    make wind_seq wind_mpi wind_omp wind_cuda wind_pthread wind_pthread_2
    
    iter=1
    max_iter=100

    system ./wind_seq "target/system.seq.csv" $iter
    for ((OMP_NUM_THREADS=1; OMP_NUM_THREADS<=16; OMP_NUM_THREADS*=2)); do
         system "OMP_STACKSIZE=999m OMP_NUM_THREADS=$OMP_NUM_THREADS ./wind_omp" "target/system.omp.$OMP_NUM_THREADS.csv" $iter
    done

    # ulimit -s unlimited
    system "OMP_STACKSIZE=999m OMP_NUM_THREADS=6 ./wind_omp" "target/system.omp.6.csv" $iter

    # matrix_ratio ./wind_seq "target/matrix_ratio.seq.csv" $iter
    # matrix_size ./wind_seq "target/matrix_size.seq.csv" $iter

    # OMP_NUM_THREADS=1
    # matrix_ratio ./wind_omp "target/matrix_ratio.omp.$OMP_NUM_THREADS.csv" $iter
    #

    # for ((OMP_NUM_THREADS=1; OMP_NUM_THREADS<=16; OMP_NUM_THREADS*=2)); do
    #     echo $OMP_NUM_THREADS
    #      matrix_size "OMP_NUM_THREADS=$OMP_NUM_THREADS ./wind_omp" "target/matrix_size.omp.$OMP_NUM_THREADS.csv" $iter
    # done
    #
    # rows=1000 columns=1000 inlet_size=$rows max_iter=100
    # particles_m ./wind_seq "target/particles_m_density.seq.csv" $iter
    # particles_f ./wind_seq "target/particles_f_density.seq.csv" $iter 
    #
    # for ((OMP_NUM_THREADS=1; OMP_NUM_THREADS<=16; OMP_NUM_THREADS*=2)); do
    #     echo $OMP_NUM_THREADS
    #     particles_m "OMP_NUM_THREADS=$OMP_NUM_THREADS ./wind_omp" "target/particles_m_density.omp.$OMP_NUM_THREADS.csv" $iter
    #     particles_f "OMP_NUM_THREADS=$OMP_NUM_THREADS ./wind_omp" "target/particles_f_density.omp.$OMP_NUM_THREADS.csv" $iter 
    # done

    # matrix_size ./wind_omp 'target/matrix_size.omp.$OMP_NUM_THREADS.csv' $iter
    # inlet_perf ./wind_seq 'target/inlet_perf.seq.csv' $iter 
fi


    # for OMP_NUM_THREADS in $(seq 1 1 16); do
    # inlet size performance
    
    # save="awk 'NR==2 {printf \"%s \", \$2}' | tr -d '\n' >> target/matrix_dimensions.csv"
    # max_iter=10 particles_f_band_size=0 particles_m_band_size=0
    # echo "rows,cols,time"
    # for ((i=10; i<100000000; i*=2)); do
    #     rows=$((100000000/i))
    #     columns=$i
    #     inlet_size=$rows
    #     echo -e "\n\nrows: $rows, cols: $columns"
    #     update_args
    #     for ((j=0; j<1; j++)); do
    #         echo "iter: $j"
    #         echo -n "$rows, $columns, " >> target/matrix_dimensions.csv
    #         ./wind_seq $args | eval $save
    #         echo >> target/matrix_dimensions.csv
    #     done
    # done


    # next: how do rows an 
    # save="awk 'NR==2 {printf \"%s \", \$2}' | tr -d '\n' >> target/weak_scaling_rows.csv"
    # echo -n "rows, seq" >> target/weak_scaling_rows.csv
    # for rows in $(seq 1000 1000 1000000); do
    #     echo "rows: $rows"
    #     inlet_size=$rows
    #
    #     update_args
    #
    #     for iter in $(seq 0 1 20); do
    #         echo "- iter: $iter"
    #         echo -n "$rows, " >> target/weak_scaling_rows.csv
    #         ./wind_seq $args | eval $save
    #         echo >> target/weak_scaling_rows.csv
    #     done
    # done

    # save="awk 'NR==2 {printf \"%s \", \$2}' | tr -d '\n' >> target/particles_density.csv"
    # save="awk 'NR==2 {printf \"%s \", \$2}' | tr -d '\n' >> target/fixed_density.csv"
    # rows=3000 columns=3000 inlet_size=$rows
    # max_iter=10 particles_f_band_size=$((rows-1)) particles_m_band_size=0
    # echo "density,time" >> target/fixed_density.csv
    # for density in $(seq 0 0.1 1); do
    #     echo -e "\n\ndensity: $density"
    #     particles_f_density=$density
    #     update_args
    #     for ((j=0; j<1; j++)); do
    #         echo "iter: $j"
    #         echo -n "$density," >> target/fixed_density.csv
    #         ./wind_seq $args | eval $save
    #         echo >> target/fixed_density.csv
    #     done
    # done
    #

# save="awk 'NR==2 {printf \"%s \", \$2}' | tr -d '\n' >> target/weak_scaling_density.csv"
# echo -n "density, seq, mpi, omp, cuda, pthread, pthread_2" >> target/weak_scaling_density.csv
# for particles_m_density in $(seq 0 0.1 1); do
#     echo $particles_m_density
#     args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"
#
#     echo -n "$particles_m_density, " >> target/weak_scaling_density.csv
#     ./wind_seq $args | eval $save
#     echo -n ", " >> target/weak_scaling_density.csv
#     mpirun ./wind_mpi $args | eval $save
#     echo -n ", " >> target/weak_scaling_density.csv
#     ./wind_omp $args | eval $save
#     echo -n ", " >> target/weak_scaling_density.csv
#     ./wind_cuda $args | eval $save
#     echo -n ", " >> target/weak_scaling_density.csv
#     ./wind_pthread $args | eval $save 
#     echo -n ", " >> target/weak_scaling_density.csv
#     ./wind_pthread_2 $args | eval $save 
#
#     echo "" >> target/weak_scaling_density.csv
# done
#
# Rscript graph.r

# rows=40
# columns=35
# max_iter=3710
# var_threshold=0.8
#
# inlet_pos=2
# inlet_size=33
#
# particles_f_band_pos=1
# particles_f_band_size=0
# particles_f_density=0 # [0.0, 0.1]
#
# particles_m_band_pos=5
# particles_m_band_size=10
# particles_m_density=0.3 # [0.0, 0.1]
#
# short_rnd1=3434
# short_rnd2=1242
# short_rnd3=965
#
# make debug
#
# # ./wind_seq $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 8 17 0.9 9 16 0.9 9 17 0.9 9 18 0.9 10 16 0.9 10 17 0.9 10 18 0.9 11 16 0.9 11 17 0.9 11 18 0.9 12 16 0.9 12 17 0.9 12 18 0.9 13 16 0.9 13 17 0.9 13 18 0.9 14 7 0.9 14 8 0.9 14 9 0.9 14 10 0.9 14 11 0.9 14 12 0.9 14 13 0.9 14 14 0.9 14 15 0.9 14 16 0.9 14 17 0.9 14 18 0.9 14 19 0.9 14 20 0.9 14 21 0.9 14 22 0.9 14 23 0.9 14 24 0.9 14 25 0.9 14 26 0.9 14 27 0.9 15 6 0.9 15 7 0.9 15 8 0.9 15 9 0.9 15 10 0.9 15 11 0.9 15 12 0.9 15 13 0.9 15 14 0.9 15 15 0.9 15 16 0.9 15 17 0.9 15 18 0.9 15 19 0.9 15 20 0.9 15 21 0.9 15 22 0.9 15 23 0.9 15 24 0.9 15 25 0.9 15 26 0.9 15 27 0.9 15 28 0.9 16 5 0.9 16 6 0.9 16 7 0.9 16 8 0.9 16 9 0.9 16 10 0.9 16 11 0.9 16 12 0.9 16 13 0.9 16 14 0.9 16 15 0.9 16 16 0.9 16 17 0.9 16 18 0.9 16 19 0.9 16 20 0.9 16 21 0.9 16 22 0.9 16 23 0.9 16 24 0.9 16 25 0.9 16 26 0.9 16 27 0.9 16 28 0.9 16 29 0.9 17 5 0.9 17 6 0.9 17 7 0.9 17 8 0.9 17 9 0.9 17 10 0.9 17 11 0.9 17 12 0.9 17 13 0.9 17 14 0.9 17 15 0.9 17 16 0.9 17 17 0.9 17 18 0.9 17 19 0.9 17 20 0.9 17 21 0.9 17 22 0.9 17 23 0.9 17 24 0.9 17 25 0.9 17 26 0.9 17 27 0.9 17 28 0.9 17 29 0.9 18 6 0.9 18 7 0.9 18 8 0.9 18 9 0.9 18 10 0.9 18 11 0.9 18 12 0.9 18 13 0.9 18 14 0.9 18 15 0.9 18 16 0.9 18 17 0.9 18 18 0.9 18 19 0.9 18 20 0.9 18 21 0.9 18 22 0.9 18 23 0.9 18 24 0.9 18 25 0.9 18 26 0.9 18 27 0.9 18 28 0.9 19 16 0.9 19 17 0.9 19 18 0.9 20 16 0.9 20 17 0.9 20 18 0.9 21 16 0.9 21 17 0.9 21 18 0.9 22 16 0.9 22 17 0.9 22 18 0.9 23 16 0.9 23 17 0.9 23 18 0.9 24 16 0.9 24 17 0.9 24 18 0.9 25 16 0.9 25 17 0.9 25 18 0.9 26 16 0.9 26 17 0.9 26 18 0.9 27 16 0.9 27 17 0.9 27 18 0.9 28 17 0.9 29 17 0.9 30 17 0.9 31 16 0.9 31 17 0.9 31 18 0.9 32 15 0.9 32 16 0.9 32 17 0.9 32 18 0.9 32 19 0.9 33 15 0.9 33 16 0.9 33 17 0.9 33 18 0.9 33 19 0.9 34 17 0.9
