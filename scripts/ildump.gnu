#GNUplot script to display the output of Lattice::indexedNDump() i.e. an "Indexed Lattice Dump"
#It should be called as follows from within an interactive GNUplot session:
# call "ildump.gnu" "filename" width height
# e.g. call "ildump.gnu" "data" 17 17

#enable mouse
set mouse

#make sure the key isn't displayed
set key off

#set the increment of x & y major axis tics
set xtics 1 ; set ytics 1

#set the number of intervals between the major axis tics to make the minor axis tics
set mxtics 2 ; set mytics 2

#setup grid
set grid mxtics mytics noxtics noytics

#Make sure the borders can't draw over the vectors - disabled not support in gnuplot 4.0
#set border back

#set axis ranges
set xrange [-2:$1]
set yrange [-2:$2]

#set 1:1 aspect ratio
set size ratio -1

plot "$0" index 0 with vectors nohead, "$0" index 1 with vectors nohead, "$0" index 2 with vectors nohead
