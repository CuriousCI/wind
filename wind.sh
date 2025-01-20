#!/bin/bash

benchmark=false
debug=false
perf=false
scaling=false
valgrind=false

while getopts ":bgpsv" opt; do
  case $opt in
    b) benchmark=true;;
    g) debug=true;;
    p) perf=true;;
    s) scaling=true;;
    v) valgrind=true;;
  esac
done

rows=200000
columns=2000
max_iter=300
var_threshold=0.0

inlet_pos=0
inlet_size=$rows

particles_f_band_pos=1
particles_f_band_size=0
particles_f_density=0.0

particles_m_band_pos=1
particles_m_band_size=200
particles_m_density=1.0

short_rnd1=$((RANDOM % 10000 + 1))
short_rnd2=$((RANDOM % 10000 + 1))
short_rnd3=$((RANDOM % 10000 + 1))

if $scaling; then
    make wind_seq wind_mpi wind_omp wind_cuda wind_pthread wind_pthread_2

    rm target/*.csv

    save="awk 'NR==2 {printf \"%s \", \$2}' | tr -d '\n' >> target/weak_scaling_density.csv"

    echo -n "density, seq, mpi, omp, cuda, pthread, pthread_2" >> target/weak_scaling_density.csv

    for particles_m_density in $(seq 0 0.1 1); do
        echo $particles_m_density
        args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"

        echo -n "$particles_m_density, " >> target/weak_scaling_density.csv
        ./wind_seq $args | eval $save
        echo -n ", " >> target/weak_scaling_density.csv
        mpirun ./wind_mpi $args | eval $save
        echo -n ", " >> target/weak_scaling_density.csv
        ./wind_omp $args | eval $save
        echo -n ", " >> target/weak_scaling_density.csv
        ./wind_cuda $args | eval $save
        echo -n ", " >> target/weak_scaling_density.csv
        ./wind_pthread $args | eval $save 
        echo -n ", " >> target/weak_scaling_density.csv
        ./wind_pthread_2 $args | eval $save 

        echo "" >> target/weak_scaling_density.csv
    done

    Rscript graph.r
fi

if $benchmark; then
    args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"

    make wind_seq wind_mpi wind_omp wind_cuda wind_pthread wind_pthread_2

    echo -e "seq"
    ./wind_seq $args | tee -a check
    # echo -e "\n\nMPI"
    # mpirun ./wind_mpi $args | tee -a check
    echo -e "\n\nOpenMP"
    ./wind_omp $args | tee -a check
    # echo -e "\n\nCUDA"
    # ./wind_cuda $args | tee -a check 
    echo -e "\n\npthread"
    ./wind_pthread $args | tee -a check
    echo -e "\n\npthread 2"
    ./wind_pthread_2 $args | tee -a check

    rm check
fi

if $perf; then 
    args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"
    make wind_seq wind_omp wind_pthread 

    perf stat $bonus ./wind_seq $args 
    perf stat $bonus ./wind_omp $args 
    perf stat $bonus ./wind_pthread $args 
fi

if $valgrind; then
    rows=100
    columns=100
    max_iter=30
    var_threshold=0.0

    inlet_pos=0
    inlet_size=$rows

    particles_f_band_pos=1
    particles_f_band_size=10
    particles_f_density=1.0 

    particles_m_band_pos=20
    particles_m_band_size=30
    particles_m_density=1.0

    args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"

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

if $debug; then
    rows=200
    columns=200
    max_iter=3710
    var_threshold=0.0

    inlet_pos=2
    inlet_size=33

    particles_f_band_pos=1
    particles_f_band_size=10
    particles_f_density=0.5 # [0.0, 0.1]

    particles_m_band_pos=10
    particles_m_band_size=80
    particles_m_density=0.5 # [0.0, 0.1]

    short_rnd1=3434
    short_rnd2=1242
    short_rnd3=965

    args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"

    make wind_pthread_2_debug
    gdb --args ./wind_pthread_2_debug $args
fi

# -np 6 --oversubscribe
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
#
# rows=40
# columns=10
# max_iter=1000
# var_threshold=0.0
#
# inlet_pos=2
# inlet_size=6
#
# particles_f_band_pos=0
# particles_f_band_size=0
# particles_f_density=0 # [0.0, 1.0]
#
# particles_m_band_pos=0
# particles_m_band_size=0
# particles_m_density=0 # [0.0, 1.0]
#
# # particles_m_band_pos=1
# # particles_m_band_size=30
# # particles_m_density=1.0 # [0.0, 1.0]
#
# short_rnd1=3434
# short_rnd2=1242
# short_rnd3=965
#
# ./wind_seq $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3
    # if $cuda; then 
        # make wind_cuda 
    # fi
        # if $cuda; then 
        # fi
# cuda=false
    # c) cuda=true;;
    # if $cuda; then 
    #     make wind_cuda 
    # fi
    # if $cuda; then
        # echo -n "cuda, " >> target/weak_scaling_density.csv
    # fi
    # echo "pthread" >> target/weak_scaling_density.csv

    # if $cuda; then 
    # fi
