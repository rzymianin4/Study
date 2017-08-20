#!/bin/bash
#Dawid Tracz

function get_h {
	echo -e "Script calculating roots of 'z^3 - 1 = 0' equation by Newton's method and drawing the fractal for a given area." ;}

function get_help {
	echo -e "Available flags:"
	echo -e "-h | --help\t shows manual."
	echo -e "-t | --threads\t (1 agrument) sets number of quadrants of the Gauss Plane that will be calculating in the same time \
(in each case all available cores will be used, but with different load. deafult: 1)"
	echo -e "-r | --real\t (2 arguments) sets real range of Gauss Plane for drawing fractal. Range HAS TO contains 0. (default: [-5, 5])"
	echo -e "-i | --imag\t (2 arguments) sets imaginary range of Gauss Plane for drawing fractal. Range HAS TO contains 0. (default: [-5, 5])"
	echo -e "-s | --step\t (1 agrument) sets distance between next two start points for both axes (default: 0.25)"
	echo -e "-p | --prec\t (1 agrument) sets precision of calculating roots to specified decimal position (default: 9)"
	echo -e "-o | --output\t (1 argument) sets name of output file (default: \"output.dat\")"
}

#-------------------------------------------------------

path=`pwd`
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

th="1"
prec=`echo "10^(-9)" | bc -l`
step="0.25"
out="output.dat"
re_sec=(-5 5)
im_sec=(-5 5)

while [[ $1 ]]; do
	case "$1" in
		-h )
			get_h
			echo "put '--help' to see more"
			exit 0 ;;
		--help )
			get_h
			echo
			get_help
			exit ;;
		-t | --threads )
			th=$2
			shift 2 ;;
		-r | --raal )
			re_sec=($2 $3)
			shift 3 ;;
		-i | --imaginary )
			im_sec=($2 $3)
			shift 3 ;;
		-s | --step )
			step=$2
			shift 2 ;;
		-p | --prec )
			prec=`echo "10^(-$2)" | bc -l`
			shift 2 ;;
		-o | --output )
			out=$2
			shift 2 ;;
		* )
			echo "bad option $1"
			echo "put \"--help\" to see the manual"
			exit -1 ;;
	esac
done

if [[ `which "gnuplot"` == "" ]]; then
	echo "gnuplot is not installed, so program cannot draw the fractal, only output data file will apear."
	read -p "Press enter to continue, or 'ctrl+c' to exit."; fi

./newton.sh ${re_sec[1]} ${im_sec[1]} $step $prec 2> /dev/null &
if [[ $th -eq 1 ]]; then
	wait; fi
./newton.sh ${re_sec[0]} ${im_sec[1]} $step $prec 2> /dev/null &
if [[ $th -eq 1 ]] || [[ $th -eq 2 ]]; then
	wait; fi
./newton.sh ${re_sec[0]} ${im_sec[0]} $step $prec 2> /dev/null &
if [[ $th -eq 1 ]] || [[ $th -eq 3 ]]; then
	wait; fi
./newton.sh ${re_sec[1]} ${im_sec[0]} $step $prec 2> /dev/null &
wait

if [[ -e $out ]]; then
	rm $out; fi
for i in `seq 1 4`; do
	cat .${i}q.dat >> ".$out"
	rm .${i}q.dat
done

./parser.sh $out
if [[ `which "gnuplot"` != "" ]]; then
	./ploter.sh ${re_sec[0]} ${re_sec[1]} ${im_sec[0]} ${im_sec[1]} "$out"; fi
rm ".$out"

cd $path