#GNUplot script to display the output of sim-state into the file coning.dump
# It outputs the Accentance angle as a function of monte carlo step
#It should be called as follows from within an interactive GNUplot session:
# call "aangle.gnu" "filename"
# e.g. call "aangle.gnu" "/path/to/coning.dump"

#enable mouse
set mouse

#set labels
set xlabel "Monte Carlo Step"
set ylabel "Acceptance angle (degrees)"

#set 1:1 aspect ratio
set size ratio 1

#define radians to degrees convert function
f(x) = x*(180/pi)

#set y-axis ticks every 5degrees
set ytics 5

set title "Coning algorithm progress in Monte Carlo Simulation"

#select columns 1 & 2 for plotting data (processing column 2 which is in radians and converting to degrees)
plot "$0" using 1:(f(column(2))) with linespoints
