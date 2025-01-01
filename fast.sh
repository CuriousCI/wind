#!/bin/bash

run_perf=false
run_valgrind=false

while getopts ":pv" opt; do
  case $opt in
    p) run_perf=true;;
    v) run_valgrind=true;;
  esac
done

rows=100
columns=100
max_iter=3710
var_threshold=0.0

inlet_pos=20
inlet_size=150

particles_f_band_pos=10
particles_f_band_size=90
particles_f_density=1.0 # [0.0, 1.0]

particles_m_band_pos=1
particles_m_band_size=90
particles_m_density=1.0 # [0.0, 1.0]

short_rnd1=$((RANDOM % 10000 + 1))
short_rnd2=$((RANDOM % 10000 + 1))
short_rnd3=$((RANDOM % 10000 + 1))

echo $short_rnd1
echo $short_rnd2
echo $short_rnd3

make wind_seq wind_mpi wind_omp wind_pthread

./wind_seq $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 | tee -a results

mpirun ./wind_mpi $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 | tee -a results

./wind_omp $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 | tee -a results 

./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 | tee -a results 
./check_results

rm results

if $run_perf; then 
    perf stat ./wind_seq $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 | tee -a results

    perf stat ./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3
fi

if $run_valgrind; then
    rows=100
    columns=100
    max_iter=1000
    var_threshold=0.0

    inlet_pos=2
    inlet_size=10

    particles_f_band_pos=1
    particles_f_band_size=10
    particles_f_density=1.0 # [0.0, 1.0]

    particles_m_band_pos=20
    particles_m_band_size=30
    particles_m_density=1.0 # [0.0, 1.0]

    ./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 

    valgrind -s --tool=massif ./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 

    valgrind -s --tool=callgrind ./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 2> data_cachegrind

    valgrind -s --tool=cachegrind ./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 2> data_cachegrind

    rm target/* -f
    mv massif.out.* callgrind.out.* cachegrind.out.* target

    ./wind_seq $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 

    valgrind -s --tool=massif ./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 

    valgrind -s --tool=callgrind ./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 2> data_cachegrind

    valgrind -s --tool=cachegrind ./wind_pthread $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 2> data_cachegrind

    for f in *.out.*; do 
        mv $f target/$f.seq
    done

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














# for rng1, rng2, rng3 in ..., prime number steps and prime number start, or just list of prime numbers

# for size in $(seq 10 10 40);
# do
#
# echo "\n\nsize: $size"
#
# ./wind_seq $size $size $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3
#
# ./wind_mpi $size $size $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3
#
# ./wind_omp $size $size $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3
#
# done

# it would be nice to extract time from output

# valgrind ./wind_seq $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3
#
# perf stat ./wind_seq $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3
#
# valgrind ./wind_mpi $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3
#
# perf stat ./wind_mpi $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3

# short_rnd1=5901
# short_rnd2=1242
# short_rnd3=965

# -np 6 --oversubscribe
