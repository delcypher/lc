#GNUplot script to display the output of the command en-den-plot
#It should be called as follows from within an interactive GNUplot session:
# call "energy-den-colour-map.gnu" "filename" width height
# e.g. call "ildump.gnu" "data-file" 17 17

#enable mouse
set mouse

#make sure the key isn't displayed
set key off

#set the increment of x & y major axis tics
set xtics 5 ; set ytics 5


#setup grid
set grid mxtics mytics noxtics noytics

#Make sure the borders can't draw over the vectors - disabled not support in gnuplot 4.0
#set border back

#set axis ranges
set xrange [0:$1]
#flip y-axis so it's the right way round
set yrange [0:$2]

#you may need this command if axes look the wrong way round
#set yrange [0:$2] reverse

#force z range
set zrange [0:*]

#set 1:1 aspect ratio
set size ratio 1

#set axis titles
set xlabel "x'"
set ylabel "y'"
set title "Colour map of energy density"

#setup pm3d
set pm3d at s
set view 60,30

#possible palette that you could use that highlights disclinations well.
#set palette define (0 "grey", 0.1 "#700000", 0.3 "red")

#plot data
splot "$0" with pm3d

#set ranges to auto for other plots
set xrange [*:*]
set yrange [*:*]
set zrange [*:*]

#set xtics and ytics to auto for other plots
set xtics autofreq
set ytics autofreq

unset mxtics
unset mytics

