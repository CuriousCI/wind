#!/bin/bash

quick=false
debug=false
valgrind=false
integration=false
submit_seq=false
submit_omp=false
submit_mpi=false
submit_cuda=false

while getopts ":bgipxvomd" option; do
    case $option in
        b) quick=true;;
        g) debug=true;;
        v) valgrind=true;;
        i) integration=true;;
        x) submit_seq=true;;
        o) submit_omp=true;;
        m) submit_mpi=true;;
        d) submit_cuda=true;;
    esac
done

# ---

rows=500 columns=500 max_iter=1000 var_threshold=0.5
inlet_pos=0 inlet_size=$columns
particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=1.0
particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=1.0
# short_rnd1=$((RANDOM % 10000 + 1)) short_rnd2=$((RANDOM % 10000 + 1)) short_rnd3=$((RANDOM % 10000 + 1))
short_rnd1=1902 short_rnd2=9019 short_rnd3=2901

update_args() {
    args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"
}

update_args
echo $args

get_time() {
    read
    read stdin; echo $stdin 1>&2
    echo $stdin | awk '{printf $2}'
    read stdin; echo $stdin 1>&2
}

div() {
    echo "$(bc -l <<< "$1/$2")"
}


# make wind_omp_2
# export OMP_NUM_THREADS=6
# perf stat -d ./wind_omp_2 $args


if $quick; then
    make wind_seq wind_omp wind_cuda_2 wind_omp_2 wind_omp_cuda wind_mpi_2 wind_mpi_omp

    echo -e "\nseq"; seq_t=$(./wind_seq $args | get_time)

    # echo -e "\nomp.1.1"; omp_1_t=$(OMP_NUM_THREADS=1 ./wind_omp $args | get_time)
    # echo "S: $(div $seq_t $omp_1_t)"
    # echo "E: $(div $omp_1_t $omp_1_t)"
    #
    # echo -e "\nomp.1.6"; t=$(OMP_NUM_THREADS=6 ./wind_omp $args | get_time)
    # echo "S: $(div $seq_t $t)"
    # echo "E: $(div $omp_1_t $t)"

    echo -e "\nomp.2.1"; omp_2_t=$(OMP_NUM_THREADS=1 ./wind_omp_2 $args | get_time)
    echo "S: $(div $seq_t $omp_2_t)"
    echo "E: $(div $omp_2_t $omp_2_t)"

    echo -e "\nomp.2.6"; t=$(OMP_NUM_THREADS=6 ./wind_omp_2 $args | get_time)
    echo "S: $(div $seq_t $t)"
    echo "E: $(div $omp_2_t $t)"

    echo -e "\ncuda.2"; t=$(./wind_cuda_2 $args | get_time)
    echo "S: $(div $seq_t $t)"

    echo -e "\nmpi.2.1"; mpi_2_t=$(mpirun -np 1 ./wind_mpi_2 $args | get_time)
    echo "S: $(div $seq_t $mpi_2_t)"
    echo "E: $(div $mpi_2_t $mpi_2_t)"

    echo -e "\nmpi.2.3"; t=$(mpirun -np 3 ./wind_mpi_2 $args | get_time)
    echo "S: $(div $seq_t $t)"
    echo "E: $(div $mpi_2_t $t)"

    echo -e "\nmpi.2.6"; t=$(mpirun -np 6 ./wind_mpi_2 $args | get_time)
    echo "S: $(div $seq_t $t)"
    echo "E: $(div $mpi_2_t $t)"

fi

if $debug; then
    rows=500 columns=500 max_iter=10000 var_threshold=0.5
    inlet_pos=0 inlet_size=$columns
    particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=0.0
    particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=0.0
    short_rnd1=$((RANDOM % 10000 + 1)) short_rnd2=$((RANDOM % 10000 + 1)) short_rnd3=$((RANDOM % 10000 + 1))
    rows=1000
    columns=1000

    update_args

    make wind_omp_2_debug wind_omp_2
    export OMP_NUM_THREADS=1
    gdb --args ./wind_omp_2 $args
    # mpirun --get-stack-traces ./wind_mpi_debug $args
    # mpirun -np 4 alacritty --hold -e gdb ./wind_mpi_debug $args 
    # make wind_pthread_2_debug
    # gdb --args ./wind_pthread_2_debug $args
    # DEBUGGING MPI # pmpiexec -n 4 ./test : -n 1 ddd ./test : -n 1 ./test
fi

if $integration; then
    rows=10 columns=10 max_iter=100 inlet_size=$rows
    inlet_pos=0 inlet_size=$columns
    particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=0.2
    particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=0.2
    echo "$short_rnd1 $short_rnd2 $short_rnd3"

    update_args
    make clean debug

    ./wind_seq $args > seq.txt
    OMP_NUM_THREADS=6 ./wind_omp_2 $args > omp_2.txt
    ./wind_cuda_2 $args > cuda_2.txt
    mpirun -np 2 ./wind_mpi_2 $args > mpi_2.txt
fi

if $valgrind; then
    rows=250 columns=250 max_iter=100 var_threshold=0.5
    inlet_pos=0 inlet_size=$columns
    particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=1.0
    particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=1.0
    short_rnd1=$((RANDOM % 10000 + 1)) short_rnd2=$((RANDOM % 10000 + 1)) short_rnd3=$((RANDOM % 10000 + 1))
    rows=500

    update_args
    make wind_seq_debug wind_omp_debug wind_omp_2_debug

    rm target/* -f

    valgrind -s --tool=massif ./wind_seq_debug $args
    valgrind -s --tool=callgrind ./wind_seq_debug $args
    valgrind -s --tool=cachegrind ./wind_seq_debug $args

    for f in *.out.*; do 
        mv $f target/$f.seq
    done

    export OMP_NUM_THREADS=1
    valgrind -s --log-file=./target/helgrind.omp.txt --tool=helgrind ./wind_omp_debug $args
    valgrind -s --tool=massif ./wind_omp_debug $args
    valgrind -s --tool=callgrind ./wind_omp_debug $args
    valgrind -s --tool=cachegrind ./wind_omp_debug $args

    for f in *.out.*; do 
        mv $f target/$f.omp
    done

    export OMP_NUM_THREADS=1
    valgrind -s --log-file=./target/helgrind.omp_2.txt --tool=helgrind ./wind_omp_2_debug $args
    valgrind -s --tool=massif ./wind_omp_2_debug $args
    valgrind -s --tool=callgrind ./wind_omp_2_debug $args
    valgrind -s --tool=cachegrind ./wind_omp_2_debug $args

    for f in *.out.*; do 
        mv $f target/$f.omp2
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

# 100000000000 rows=100000000 columns=10

data() {
    for ((j=0; j<$iter; j++)); do
        echo "$prog - size: $i - iter: $j"
        echo -n "$name,$id,$i," >> $file 
        time=$(eval $prog $args | get_time) 
        echo $time >> $file
    done
}

tunnel_size_4() {
    local name="tunnel_size_4"

    max_iter=10000 var_threshold=0.0
    inlet_pos=0
    particles_f_band_pos=1 particles_f_band_size=0 particles_f_density=0.0
    particles_m_band_pos=1 particles_m_band_size=0 particles_m_density=0.0
    for ((i=1; i<=1024; i*=2)); do
        rows=$i columns=$i inlet_size=$i
        update_args
        data
    done
}

tunnel_size_3() {
    local name="tunnel_size_3"

    max_iter=1000 var_threshold=0.0
    inlet_pos=0
    particles_f_band_pos=1 particles_f_band_size=0 particles_f_density=0.0
    particles_m_band_pos=1 particles_m_band_size=0 particles_m_density=0.0
    for ((i=1; i<=16384; i*=2)); do
        rows=$i columns=$i inlet_size=$i
        update_args
        data
    done
}

particles_f_4() {
    local name="particles_f_4"

    max_iter=10000 var_threshold=0.0
    rows=512 columns=512 
    inlet_pos=0 inlet_size=512
    particles_m_band_pos=1 particles_m_band_size=0 particles_m_density=0.0
    for ((i=2; i<=512; i*=2)); do
        particles_f_band_pos=1 particles_f_band_size=$((i-1)) particles_f_density=1.0
        update_args
        data
    done
}

particles_f_3() {
    local name="particles_f_3"

    max_iter=1000 var_threshold=0.0
    rows=1024 columns=1024 
    inlet_pos=0 inlet_size=1024
    particles_m_band_pos=1 particles_m_band_size=0 particles_m_density=0.0
    for ((i=2; i<=1024; i*=2)); do
        particles_f_band_pos=1 particles_f_band_size=$((i-1)) particles_f_density=1.0
        update_args
        data
    done
}

particles_m_4_once() {
    local iter=1
    local name="particles_m_4"

    max_iter=10000 var_threshold=0.0
    rows=512 columns=512 
    inlet_pos=0 inlet_size=512
    particles_f_band_pos=1 particles_f_band_size=0 particles_f_density=0.0
    for ((i=2; i<=512; i*=2)); do
        particles_m_band_pos=1 particles_m_band_size=$((i-1)) particles_m_density=1.0
        update_args
        data
    done
}

particles_m_3() {
    local name="particles_m_3"

    max_iter=1000 var_threshold=0.0
    rows=512 columns=512 
    inlet_pos=0 inlet_size=512
    particles_f_band_pos=1 particles_f_band_size=0 particles_f_density=0.0
    for ((i=2; i<=512; i*=2)); do
        particles_m_band_pos=1 particles_m_band_size=$((i-1)) particles_m_density=1.0
        update_args
        data
    done
}

full_system_4_once() {
    local iter=1
    local name="full_system_4_once"

    max_iter=10000 var_threshold=0.0
    rows=512 columns=512 
    inlet_pos=0 inlet_size=512
    for ((i=2; i<=512; i*=2)); do
        particles_f_band_pos=1 particles_f_band_size=$((i-1)) particles_f_density=1.0
        particles_m_band_pos=1 particles_m_band_size=$((i-1)) particles_m_density=1.0
        update_args
        data
    done
}

full_system_3() {
    local name="full_system_3"

    max_iter=1000 var_threshold=0.0
    rows=512 columns=512 
    inlet_pos=0 inlet_size=512
    for ((i=2; i<=512; i*=2)); do
        particles_f_band_pos=1 particles_f_band_size=$((i-1)) particles_f_density=1.0
        particles_m_band_pos=1 particles_m_band_size=$((i-1)) particles_m_density=1.0
        update_args
        data
    done
}

file="data/logs.csv" iter=10
short_rnd1=3434 short_rnd2=1242 short_rnd3=965

# short_rnd1=3434
# short_rnd2=1242
# short_rnd3=965
# if $submit; then
#     echo "name,id,size,time" >> $file 
# fi

if $submit_seq; then
    prog="./wind_seq" id="seq"
    tunnel_size_4
    tunnel_size_3
    particles_f_4
    particles_f_3
    particles_m_3
    full_system_3
    particles_m_4_once
    full_system_4_once
fi

if $submit_omp; then
    for ((OMP_NUM_THREADS=1; OMP_NUM_THREADS<=64; OMP_NUM_THREADS*=2)); do
        prog="OMP_NUM_THREADS=$OMP_NUM_THREADS ./wind_omp_2" id="omp_$OMP_NUM_THREADS"
        tunnel_size_4
        tunnel_size_3
        particles_f_4
        particles_f_3
        particles_m_3
        full_system_3
        particles_m_4_once
        full_system_4_once
    done
fi

if $submit_mpi; then
    for ((MPI_PROC=1; MPI_PROC<=8; MPI_PROC*=2)); do
        prog="mpirun -np $MPI_PROC ./wind_mpi_2" id="mpi_$MPI_PROC"
        tunnel_size_4
        tunnel_size_3
        particles_f_4
        particles_f_3
        particles_m_3
        full_system_3
        particles_m_4_once
        full_system_4_once
    done
fi

if $submit_cuda; then
    prog="./wind_cuda_2" id="cuda"
    tunnel_size_4
    tunnel_size_3
    particles_f_4
    particles_f_3
    particles_m_3
    full_system_3
    particles_m_4_once
    full_system_4_once
fi

# if $scaling; then



# fi


    # system ./wind_seq "target/system.seq$pref.csv" $iter
    # for ((OMP_NUM_THREADS=1; OMP_NUM_THREADS<=16; OMP_NUM_THREADS*=2)); do
    #      system "OMP_NUM_THREADS=$OMP_NUM_THREADS ./wind_omp" "target/system.omp.$OMP_NUM_THREADS$pref.csv" $iter
    # done
    # system "OMP_NUM_THREADS=6 ./wind_omp" "target/system.omp.6$pref.csv" $iter


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
    # ulimit -s unlimited
    # OMP_STACKSIZE=999m 

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
#
# rows=40 columns=35 max_iter=3710 var_threshold=0.8
# inlet_pos=2 inlet_size=33
# particles_f_band_pos=1 particles_f_band_size=0 particles_f_density=0
# particles_m_band_pos=5 particles_m_band_size=10 particles_m_density=0.3
# short_rnd1=3434 short_rnd2=1242 short_rnd3=965
# bonus="8 17 0.9 9 16 0.9 9 17 0.9 9 18 0.9 10 16 0.9 10 17 0.9 10 18 0.9 11 16 0.9 11 17 0.9 11 18 0.9 12 16 0.9 12 17 0.9 12 18 0.9 13 16 0.9 13 17 0.9 13 18 0.9 14 7 0.9 14 8 0.9 14 9 0.9 14 10 0.9 14 11 0.9 14 12 0.9 14 13 0.9 14 14 0.9 14 15 0.9 14 16 0.9 14 17 0.9 14 18 0.9 14 19 0.9 14 20 0.9 14 21 0.9 14 22 0.9 14 23 0.9 14 24 0.9 14 25 0.9 14 26 0.9 14 27 0.9 15 6 0.9 15 7 0.9 15 8 0.9 15 9 0.9 15 10 0.9 15 11 0.9 15 12 0.9 15 13 0.9 15 14 0.9 15 15 0.9 15 16 0.9 15 17 0.9 15 18 0.9 15 19 0.9 15 20 0.9 15 21 0.9 15 22 0.9 15 23 0.9 15 24 0.9 15 25 0.9 15 26 0.9 15 27 0.9 15 28 0.9 16 5 0.9 16 6 0.9 16 7 0.9 16 8 0.9 16 9 0.9 16 10 0.9 16 11 0.9 16 12 0.9 16 13 0.9 16 14 0.9 16 15 0.9 16 16 0.9 16 17 0.9 16 18 0.9 16 19 0.9 16 20 0.9 16 21 0.9 16 22 0.9 16 23 0.9 16 24 0.9 16 25 0.9 16 26 0.9 16 27 0.9 16 28 0.9 16 29 0.9 17 5 0.9 17 6 0.9 17 7 0.9 17 8 0.9 17 9 0.9 17 10 0.9 17 11 0.9 17 12 0.9 17 13 0.9 17 14 0.9 17 15 0.9 17 16 0.9 17 17 0.9 17 18 0.9 17 19 0.9 17 20 0.9 17 21 0.9 17 22 0.9 17 23 0.9 17 24 0.9 17 25 0.9 17 26 0.9 17 27 0.9 17 28 0.9 17 29 0.9 18 6 0.9 18 7 0.9 18 8 0.9 18 9 0.9 18 10 0.9 18 11 0.9 18 12 0.9 18 13 0.9 18 14 0.9 18 15 0.9 18 16 0.9 18 17 0.9 18 18 0.9 18 19 0.9 18 20 0.9 18 21 0.9 18 22 0.9 18 23 0.9 18 24 0.9 18 25 0.9 18 26 0.9 18 27 0.9 18 28 0.9 19 16 0.9 19 17 0.9 19 18 0.9 20 16 0.9 20 17 0.9 20 18 0.9 21 16 0.9 21 17 0.9 21 18 0.9 22 16 0.9 22 17 0.9 22 18 0.9 23 16 0.9 23 17 0.9 23 18 0.9 24 16 0.9 24 17 0.9 24 18 0.9 25 16 0.9 25 17 0.9 25 18 0.9 26 16 0.9 26 17 0.9 26 18 0.9 27 16 0.9 27 17 0.9 27 18 0.9 28 17 0.9 29 17 0.9 30 17 0.9 31 16 0.9 31 17 0.9 31 18 0.9 32 15 0.9 32 16 0.9 32 17 0.9 32 18 0.9 32 19 0.9 33 15 0.9 33 16 0.9 33 17 0.9 33 18 0.9 33 19 0.9 34 17 0.9"

    # export OMP_NUM_THREADS=1
    # echo -e "\nmpi.omp.1.1"; mpi_omp_t=$(mpirun -np 1 --bind-to none ./wind_mpi_omp $args | get_time)
    # echo "S: $(div $seq_t $mpi_omp_t)"
    # echo "E: $(div $mpi_omp_t $mpi_omp_t)"
    #
    # export OMP_NUM_THREADS=3
    # echo -e "\nmpi.omp.2.3"; t=$(mpirun -np 2 --bind-to none ./wind_mpi_omp $args | get_time)
    # echo "S: $(div $seq_t $t)"
    # echo "E: $(div $mpi_omp_t $t)"
    #
    # export OMP_NUM_THREADS=2
    # echo -e "\nmpi.omp.3.2"; t=$(mpirun -np 3 --bind-to none ./wind_mpi_omp $args | get_time)
    # echo "S: $(div $seq_t $t)"
    # echo "E: $(div $mpi_omp_t $t)"
    
    # echo -e "\nomp.cuda"; OMP_NUM_THREADS=6 ./wind_omp_cuda $args 

    # echo -e "\nomp.cuda"; t=$(./wind_omp_cuda $args | get_time)
    # echo "S: $(div $seq_t $t)"
    # echo -e "\nmpi"; t=$(mpirun ./wind_mpi $args | get_time)

    # echo -e "\n\nomp 3"; OMP_NUM_THREADS=3 ./wind_omp $args
    # val=$(echo -e "\n\nomp 1"; OMP_NUM_THREADS=1 ./wind_omp $args | tee >(awk 'NR==2 {printf $2}'))
    # echo "speed-up: $((seq_time/val))"
    # val=$(echo -e "\n\nomp 6"; OMP_NUM_THREADS=6 ./wind_omp $args | eval $get_time)
    # echo "speed-up: $((seq_time/val))"

    # val=$(echo -e "\n\nomp-2 1"; OMP_NUM_THREADS=1 ./wind_omp_2 $args | eval $get_time)
    # echo "speed-up: $((seq_time/val))"
    # val=$(echo -e "\n\nomp-2 6"; OMP_NUM_THREADS=6 ./wind_omp_2 $args | eval $get_time)
    # echo "speed-up: $((seq_time/val))"

    # for ((OMP_NUM_THREADS=1; OMP_NUM_THREADS<=16; OMP_NUM_THREADS*=2)); do
    #     echo -e "\n\nomp $OMP_NUM_THREADS"; OMP_NUM_THREADS=$OMP_NUM_THREADS ./wind_omp $args
    # done
    # echo -e "\n\npthread"; ./wind_pthread $args
    # echo -e "\n\npthread 2"; ./wind_pthread_2 $args
    # echo -e "\n\nmpi"; mpirun ./wind_mpi $args
    # echo -e "\n\ncuda"; ./wind_cuda $args

# perf=false
# p) perf=true;;
# if $perf; then 
#     make wind_seq wind_omp
#     perf stat ./wind_seq $args 
#     OMP_NUM_THREADS=6 perf stat ./wind_omp $args 
# fi

    # make wind_omp_2_debug
    # gdb --args ./wind_omp_2_debug $args
    # rows=100 columns=100 max_iter=1000 inlet_size=$rows
    # particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=1.0
    # particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=0.0


    # valgrind -s --log-file=./target/helgrind.pthread_2.txt --tool=helgrind ./wind_pthread_2_debug $args 
    # valgrind -s --tool=massif ./wind_pthread_2_debug $args 
    # valgrind -s --tool=callgrind ./wind_pthread_2_debug $args
    # valgrind -s --tool=cachegrind ./wind_pthread_2_debug $args
    #
    # for f in *.out.*; do 
    #     mv $f target/$f.pthread
    # done

    # ----


# matrix_size() {
#     local prog=$1
#     local file=$2
#     local iter=$3
#
#     # 10000
#     particles_f_band_size=0 particles_m_band_size=0 inlet_size=0
#     echo "cells,time" >> $file
#     for ((i=1; i<10000; i*=2)); do
#         rows=$i columns=$i inlet_size=$i
#         update_args
#
#         for ((j=0; j<$iter; j++)); do
#             echo "$prog - rows: $rows - columns: $columns - iter: $j"
#             echo -n "$((rows*columns)), " >> $file 
#             eval $prog $args | eval $get_time >> $file
#             echo >> $file 
#         done
#     done
# }

# rows=10000 columns=10000 max_iter=1000 var_threshold=0.5
# inlet_pos=0 inlet_size=$columns
# particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=0.0
# particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=0.0
# short_rnd1=$((RANDOM % 10000 + 1)) short_rnd2=$((RANDOM % 10000 + 1)) short_rnd3=$((RANDOM % 10000 + 1))

# particles_m() {
#     local prog=$1
#     local file=$2
#     local iter=$3
#
#     echo "density,time" >> $file
#     for particles_m_density in $(seq 0 0.1 1); do
#         update_args
#         for ((j=0; j<$iter; j++)); do
#             echo "$prog - density: $particles_m_density - iter: $j"
#             echo -n "$particles_m_density," >> $file 
#             eval $prog $args | eval $get_time >> $file
#             echo >> $file 
#         done
#     done
# }

# particles_f() {
#     local prog=$1
#     local file=$2
#     local iter=$3
#
#     particles_f_band_size=$((rows-1)) particles_m_band_size=0
#     echo "density,time" >> $file
#     for particles_f_density in $(seq 0 0.1 1); do
#         update_args
#         for ((j=0; j<$iter; j++)); do
#             echo "$prog - density: $particles_f_density - iter: $j"
#             echo -n "$particles_f_density," >> $file 
#             eval $prog $args | eval $get_time >> $file
#             echo >> $file 
#         done
#     done
# }

# inlet_perf() {
#     local prog=$1
#     local file=$2
#     local iter=$3
#
#     particles_f_band_size=0 particles_m_band_size=0
#     echo "inlet_size,time" >> $file
#     for inlet_size in $(seq 0 100 $rows); do
#         update_args
#         for ((j=0; j<$iter; j++)); do
#             echo "$prog - inlet_size: $inlet_size - iter: $j"
#             echo -n "$inlet_size," >> $file 
#             eval $prog $args | eval $get_time >> $file
#             echo >> $file 
#         done
#     done
# }

# system() {
#     local prog=$1
#     local file=$2
#     local iter=$3
#
#     # 10000
#     echo "rows,time" >> $file
#     particles_f_density=0.5
#     particles_m_density=0.5
#     # 10000
#     for ((i=1; i <= 20000; i*=2)); do
#     # for ((i=1; i<=4096; i*=2)); do
#     # for ((i=1; i<=2048; i*=2)); do
#         rows=$i columns=$((i/10 + 1)) inlet_size=$i
#         particles_f_band_size=$((rows-1)) particles_m_band_size=$((rows-1)) inlet_size=$((rows-1))
#         update_args
#
#         for ((j=0; j<$iter; j++)); do
#             echo "$prog - rows: $rows - columns: $columns - iter: $j"
#             echo -n "$((rows)), " >> $file 
#             eval $prog $args | eval $get_time >> $file
#             echo >> $file 
#         done
#     done
#
# }
