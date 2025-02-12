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

rows=512 columns=512 max_iter=1000 var_threshold=0.5
inlet_pos=0 inlet_size=$columns
particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=1.0
particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=1.0
# short_rnd1=$((RANDOM % 10000 + 1)) short_rnd2=$((RANDOM % 10000 + 1)) short_rnd3=$((RANDOM % 10000 + 1))
short_rnd1=1902 short_rnd2=9019 short_rnd3=2901
# rows=1024

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

# args="538 60 1397 0.5 30 29 0 0 0 0 0 0 3431 9012 6432"
# args="456 812 1004 2.2 21 745 0 0 0 0 0 0 684 384 1292"
# args="38000 32 31000 0.5 3 24 0 0 0 0 0 0 583 1943 2345"
# args="32 2100000 118 0.1 0 2100000 0 0 0 0 0 0 673 3902 43"

# args="102 80 352 0.1 10 50 0 0 0 0 0 0 3431 9012 12432 20 12 0.712 20 13 0.713 20 14 0.714 20 15 0.715 20 16 0.716 20 17 0.717 20 18 0.718 20 19 0.719 20 20 0.720 30 16 0.516 30 18 0.518 30 20 0.520 30 22 0.522 40 20 0.420 40 30 0.430 40 40 0.440 40 50 0.450 40 60 0.460 40 70 0.470"
# args="102 80 352 0.1 10 50 15 16 0.1 0 0 0 3431 9012 12432 20 12 0.712 20 13 0.713 20 14 0.714 20 15 0.715 20 16 0.716 20 17 0.717 20 18 0.718 20 19 0.719 20 20 0.720 30 16 0.516 30 18 0.518 30 20 0.520 30 22 0.522 40 20 0.420 40 30 0.430 40 40 0.440 40 50 0.450 40 60 0.460 40 70 0.470"
# args="2100 457 6300 0.4 1 452 20 2000 0.001 16 50 0.2 583 223 712"

if $quick; then
    make wind_seq wind_omp wind_mpi wind_cuda wind_mpi_omp wind_mpi_cuda wind_omp_cuda

    echo -e "\nseq"; seq_t=$(./wind_seq $args | get_time)

    # echo -e "\nomp.1.1"; omp_1_t=$(OMP_NUM_THREADS=1 ./wind_omp $args | get_time)
    # echo "S: $(div $seq_t $omp_1_t)"
    # echo "E: $(div $omp_1_t $omp_1_t)"
    #
    # echo -e "\nomp.1.6"; t=$(OMP_NUM_THREADS=6 ./wind_omp $args | get_time)
    # echo "S: $(div $seq_t $t)"
    # echo "E: $(div $omp_1_t $t)"

    # export GOMP_CPU_AFFINITY="0 1 2 3 4 5"

    echo -e "\nomp_1"; omp_t=$(OMP_NUM_THREADS=1 ./wind_omp $args | get_time)
    echo "S: $(div $seq_t $omp_t)"
    echo "E: $(div $omp_t $omp_t)"
    
    echo -e "\nomp_6"; t=$(OMP_NUM_THREADS=6 ./wind_omp $args | get_time)
    echo "S: $(div $seq_t $t)"
    echo "E: $(div $omp_t $t)"

    echo -e "\ncuda"; t=$(./wind_cuda $args | get_time)
    echo "S: $(div $seq_t $t)"

    echo -e "\nomp_cuda_6"; t=$(OMP_NUM_THREADS=6 ./wind_omp_cuda $args | get_time)
    echo "S: $(div $seq_t $t)"

    echo -e "\nmpi_1"; mpi_t=$(mpirun -np 1 ./wind_mpi $args | get_time)
    echo "S: $(div $seq_t $mpi_t)"
    echo "E: $(div $mpi_t $mpi_t)"

    echo -e "\nmpi_3"; t=$(mpirun -np 3 ./wind_mpi $args | get_time)
    echo "S: $(div $seq_t $t)"
    echo "E: $(div $mpi_t $t)"

    echo -e "\nmpi_6"; t=$(mpirun -np 6 ./wind_mpi $args | get_time)
    echo "S: $(div $seq_t $t)"
    echo "E: $(div $mpi_t $t)"

    # echo -e "\nmpi_cuda_1"; mpi_cuda_t=$(mpirun -np 1 ./wind_mpi $args | get_time)
    # echo "S: $(div $seq_t $mpi_cuda_t)"
    # echo "E: $(div $mpi_cuda_t $mpi_cuda_t)"
    #
    # echo -e "\nmpi_cuda_6"; t=$(mpirun -np 6 ./wind_mpi $args | get_time)
    # echo "S: $(div $seq_t $t)"
    # echo "E: $(div $mpi_cuda_t $t)"

    echo -e "\nmpi_omp_1_6"; mpi_t=$(OMP_NUM_THREADS=6 mpirun --bind-to none -np 1 ./wind_mpi_omp $args | get_time)
    echo "S: $(div $seq_t $mpi_t)"
    echo "E: $(div $mpi_t $mpi_t)"

    echo -e "\nmpi_omp_2_3"; t=$(OMP_NUM_THREADS=3 mpirun --bind-to none -np 2 ./wind_mpi_omp $args | get_time)
    echo "S: $(div $seq_t $t)"
    echo "E: $(div $mpi_t $t)"

    echo -e "\nmpi_omp_3_2"; t=$(OMP_NUM_THREADS=2 mpirun --bind-to none -np 3 ./wind_mpi_omp $args | get_time)
    echo "S: $(div $seq_t $t)"
    echo "E: $(div $mpi_t $t)"
fi

if $integration; then
    rows=128 columns=128 max_iter=512 inlet_size=$rows
    inlet_pos=0 inlet_size=$columns
    particles_f_band_pos=1 particles_f_band_size=$((rows-1)) particles_f_density=1.0
    particles_m_band_pos=1 particles_m_band_size=$((rows-1)) particles_m_density=1.0
    echo "$short_rnd1 $short_rnd2 $short_rnd3"

    update_args
    make clean debug

    ./wind_seq $args > seq.txt
    OMP_NUM_THREADS=6 ./wind_omp $args > omp.txt
    ./wind_cuda $args > cuda.txt
    mpirun -np 2 ./wind_mpi $args > mpi.txt
    OMP_NUM_THREADS=3 mpirun --bind-to none -np 2 ./wind_mpi_omp $args > mpi_omp.txt
    OMP_NUM_THREADS=6 ./wind_omp_cuda $args > omp_cuda.txt
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
        update_args; data
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
        update_args; data
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
        update_args; data
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
        update_args; data
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
        update_args; data
    done
}

file="data/data.csv" iter=1
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
    for ((OMP_NUM_THREADS=1; OMP_NUM_THREADS<=32; OMP_NUM_THREADS*=2)); do
        prog="OMP_NUM_THREADS=$OMP_NUM_THREADS ./wind_omp" id="omp_$OMP_NUM_THREADS"
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
        prog="mpirun -np $MPI_PROC ./wind_mpi" id="mpi_$MPI_PROC"
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
    prog="./wind_cuda" id="cuda"
    tunnel_size_4
    tunnel_size_3
    particles_f_4
    particles_f_3
    particles_m_3
    full_system_3
    particles_m_4_once
    full_system_4_once
fi
