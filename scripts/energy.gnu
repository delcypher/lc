#GNUplot script to display the "energy" as a function of Monte Carlo Step that is outputted by sim-state
#It should be called as follows from within an interactive GNUplot session:
# call "energy.gnu" "filename"
# e.g. call "energy.gnu" "energy.dump"

#enable mouse
set mouse

set title "Free energy of lattice progression through Monte Carlo Algorithm"
set xlabel "Monte Carlo Step"
set ylabel "Energy/k_1 (Joules/Newton)"

#Hide key as it's "every" is misleading as it's not quite correct.
set key off

#Don't have border cover up y-axis

#plot from -1 so that border doesn't cover x-axis (could do ``unset border'') but this looks nicer
set yrange [-1:*]

#plot data
plot "$0"  with linespoints
