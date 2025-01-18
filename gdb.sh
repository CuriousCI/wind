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

make wind_pthread_gdb 

# ./wind_seq $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 | tee -a results
#
# # -np 6 --oversubscribe
#
# mpirun --oversubscribe ./wind_mpi $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 | tee -a results
#
# ./wind_omp $rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3 | tee -a results 
args="$rows $columns $max_iter $var_threshold $inlet_pos $inlet_size $particles_f_band_pos $particles_f_band_size $particles_f_density $particles_m_band_pos $particles_m_band_size $particles_m_density $short_rnd1 $short_rnd2 $short_rnd3"
gdb --args ./wind_pthread_gdb $args
