#!/bin/bash

run_benchmark=false
run_perf=false
run_valgrind=false
run_cuda=false
run_stats=false

while getopts ":pvbcs" opt; do
  case $opt in
    p) run_perf=true;;
    v) run_valgrind=true;;
    b) run_benchmark=true;;
    c) run_cuda=true;;
    s) run_stats=true;;
  esac
done

rows=200
columns=200
max_iter=1000
var_threshold=0.0

inlet_pos=0
inlet_size=55

particles_f_band_pos=1
particles_f_band_size=0
particles_f_density=0.0 # [0.0, 1.0]

particles_m_band_pos=1
particles_m_band_size=99
particles_m_density=1.0 # [0.0, 1.0]

short_rnd1=$((RANDOM % 10000 + 1))
short_rnd2=$((RANDOM % 10000 + 1))
short_rnd3=$((RANDOM % 10000 + 1))

echo $short_rnd1
echo $short_rnd2
echo $short_rnd3


if $run_stats; then
    make wind_seq wind_mpi wind_omp wind_pthread
    if $run_cuda; then 
        make wind_cuda 
    fi

    rm target/*.csv

    save="awk 'NR==2 {printf \"%s \", \$2}' | tr -d '\n' >> target/data.csv"

    echo -n "density, seq, mpi, omp, " >> target/data.csv
    if $run_cuda; then
        echo -n "cuda, " >> target/data.csv
    fi
    echo "pthread" >> target/data.csv

    for particles_m_density in $(seq 0 0.1 1); do
        echo $particles_m_density

        args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"

        echo -n "$particles_m_density, " >> target/data.csv
        ./wind_seq $args | eval $save
        echo -n ", " >> target/data.csv
        mpirun ./wind_mpi $args | eval $save
        echo -n ", " >> target/data.csv
        ./wind_omp $args | eval $save
        echo -n ", " >> target/data.csv
        if $run_cuda; then 
            ./wind_cuda $args | eval $save
            echo -n ", " >> target/data.csv
        fi
        ./wind_pthread $args | eval $save 

        echo "" >> target/data.csv
    done

    Rscript graph.r
fi

if $run_benchmark; then
    make wind_seq wind_mpi wind_omp wind_pthread
    if $run_cuda; then 
        make wind_cuda 
    fi

    ./wind_seq $args | tee -a results
    mpirun ./wind_mpi $args | tee -a results
    ./wind_omp $args | tee -a results
    if $run_cuda; then 
        ./wind_cuda $args | tee -a results 
    fi
    ./wind_pthread $args | tee -a results

    ./check_results
    rm results
fi

if $run_perf; then 
    make wind_seq wind_pthread wind_omp

    perf stat ./wind_seq $args
    perf stat ./wind_omp $args 
    perf stat ./wind_pthread $args 
fi

if $run_valgrind; then
    rows=100
    columns=100
    max_iter=30
    var_threshold=0.0

    inlet_pos=2
    inlet_size=10

    particles_f_band_pos=1
    particles_f_band_size=10
    particles_f_density=1.0 # [0.0, 1.0]

    particles_m_band_pos=20
    particles_m_band_size=30
    particles_m_density=1.0 # [0.0, 1.0]

    args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"

    make wind_seq_gdb wind_pthread_gdb wind_omp_gdb
    rm target/* -f

    valgrind -s --tool=massif ./wind_seq_gdb $args
    valgrind -s --tool=callgrind ./wind_seq_gdb $args
    valgrind -s --tool=cachegrind ./wind_seq_gdb $args

    for f in *.out.*; do 
        mv $f target/$f.seq
    done

    valgrind -s --log-file=./target/helgrind.omp.txt --tool=helgrind ./wind_omp_gdb $args
    valgrind -s --tool=massif ./wind_omp_gdb $args
    valgrind -s --tool=callgrind ./wind_omp_gdb $args
    valgrind -s --tool=cachegrind ./wind_omp_gdb $args

    for f in *.out.*; do 
        mv $f target/$f.omp
    done

    valgrind -s --log-file=./target/helgrind.pthread.txt --tool=helgrind ./wind_pthread_gdb $args 
    valgrind -s --tool=massif ./wind_pthread_gdb $args 
    valgrind -s --tool=callgrind ./wind_pthread_gdb $args
    valgrind -s --tool=cachegrind ./wind_pthread_gdb $args

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

#TODO: cuda performance generation


# for rng1, rng2, rng3 in ..., prime number steps and prime number start, or just list of prime numbers
# for size in $(seq 10 10 40);
# do
# it would be nice to extract time from output
# short_rnd1=5901
# short_rnd2=1242
# short_rnd3=965
# -np 6 --oversubscribe
