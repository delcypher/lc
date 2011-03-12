#GNUplot script to display the output of sim-state into the file annealing.dump
#It outputs the value of iTk () again 
#It should be called as follows from within an interactive GNUplot session:
# call "aangle.gnu" "filename"
# e.g. call "aangle.gnu" "/path/to/annealing.dump"

#enable mouse
set mouse

#set labels
set xlabel "Monte Carlo Step"
set ylabel "iTK"

#set 1:1 aspect ratio
#set size ratio -1

set title "Cooling progress in Monte Carlo Simulation"

#select columns 1 & 2 for plotting data 
plot "$0" using 1:2 with linespoints
