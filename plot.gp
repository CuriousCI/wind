set encoding utf8
set datafile separator ','

set terminal X11 1 title 'float' enhanced font "New Computer Modern,12"
set format x "%.0f"
set logscale x 2
plot 'target/matrix_dimensions.csv' using 1:3 with lines

set terminal X11 2 title 'float' enhanced font "New Computer Modern,12"
plot 'target/particles_density.csv' using 1:2 with lines

set terminal X11 3 title 'float' enhanced font "New Computer Modern,12"
plot 'target/fixed_density.csv' using 1:2 with lines

# plot 'target/weak_scaling_rows.csv' using 0:1
