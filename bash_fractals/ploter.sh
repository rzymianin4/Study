#!/bin/bash
#Daiwd Tracz

filename=$5
re_sec=($1 $2)
im_sec=($3 $4)

`echo -e "#!/usr/bin/gnuplot --persist \n
set style line 3 lc rgb '#0060ad' lt 1 lw 2 pt 5 ps 0.1 # --- blue
set style line 1 lc rgb '#00ff00' lt 1 lw 2 pt 5 ps 0.1 # --- green
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 0.1 # --- red
set terminal png size 800,800 enhanced font 'Helvetica,20'
set xrange [${re_sec[0]}:${re_sec[1]}]
set yrange [${im_sec[0]}:${im_sec[1]}]
set output 'figure1.png'
unset key \n
plot '$filename' index 0 with points ls 1, \
     ''                  index 1 with points ls 2, \
     ''                  index 2 with points ls 3 " > "ploter.gnu"`
     
chmod u+x "ploter.gnu"
./ploter.gnu
rm ploter.gnu