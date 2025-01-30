set encoding utf8
set datafile separator ','
set terminal X11
set multiplot layout 2,3

prog="omp mpi cuda pthread"
file="1 2 4 6 8 16"

set key outside right top box width 0 height 0 vertical spacing 0.5 samplen 1.5 noenhanced
set key at screen 0.95, 0.95

plot 'target/system.seq.csv' using 1:2 with lines, \
     for [i=1:words(file)] sprintf("target/system.omp.%s.csv", word(file, i)) using 1:2 with lines title sprintf("omp %s", word(file, i))

set key outside right top box width 0 height 0 vertical spacing 0.5 samplen 1.5 noenhanced
set key at screen 0.95, 0.15
# set yr[0:3]

plot 'target/system.seq.new.csv' using 1:2 with lines, \
     for [i=1:words(file)] sprintf("target/system.omp.%s.new.csv", word(file, i)) using 1:2 with lines title sprintf("omp %s", word(file, i))

# set key outside right bottom box width 0 height 0 vertical spacing 0.5 samplen 1.5 noenhanced
# set key at screen 2, 2 

# set format x "%.0f"
# set logscale x 2
# 
# plot 'target/matrix_ratio.seq.csv' using 1:3 with lines, \
#      for [i=1:16] sprintf("target/matrix_ratio.omp.%d.csv", i) using 1:3 with lines
# 
# plot 'target/matrix_size.seq.csv' using 1:2 with lines, \
#      for [i=1:16] sprintf("target/matrix_size.omp.%d.csv", i) using 1:2 with lines
# 
# reset
# set datafile separator ','

# plot 'target/particles_m_density.seq.csv' using 1:2 with lines, \
     # for [i=1:words(file)] sprintf("target/particles_m_density.omp.%d.csv", i) using 1:2 with lines

# plot 'target/particles_f_density.seq.csv' using 1:2 with lines, \
     # for [i=1:16] sprintf("target/particles_f_density.omp.%d.csv", i) using 1:2 with lines

# plot 'target/inlet_perf.seq.csv' using 1:2 with lines, \
     # 'target/inlet_perf.omp.csv' using 1:2 with lines

# plot 'target/particles_m_density.seq.csv' using 1:2 with lines, \
     # 'target/particles_m_density.omp.csv' using 1:2 with lines
    

unset multiplot

# pause -1

# set datafile separator ","
# set table 'stats.dat'
#     plot 'data.csv' using ($1):($2) with table
# unset table
# 
# # Calculate group statistics
# set table 'group_stats.dat'
#     plot 'stats.dat' using ($1):($2) with table group by column(1)
# unset table
# 
# # Extract statistics
# set datafile separator ","
# stats "group_stats.dat" u 2 name "mean"
# stats "group_stats.dat" u 3 name "stddev"
# 
# # Plot results
# set xlabel "x"
# set ylabel "y"
# plot \
#     'data.csv' using ($1):($2) with points title "Data", \
#     mean with lines title "Mean", \
#     stddev with filledcurves y1=mean lt -1 title "Standard Deviation"
