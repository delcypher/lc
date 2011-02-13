#GNUplot script to display the output of uniform-rnd
#It should be called as follows from within an interactive GNUplot session:
# call "rndhistogram.gnu" "filename"

#enable mouse
set mouse

#make sure the key isn't displayed
set key off

#set the increment of x & y major axis tics
set xtics 0.1 

#set the number of intervals between the major axis tics to make the minor axis tics
set mxtics 2 ; set mytics 2

#setup grid
set grid mxtics mytics noxtics noytics

#set axis ranges
set xrange [0:1]

set xlabel "Pseudorandom number"
set ylabel "Count"

#enable bar borders
set style fill solid

#set style fill border 0

plot "$0" index 0 with boxes lt 1
