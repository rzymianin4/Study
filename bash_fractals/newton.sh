#!/bin/bash
#Dawid Tracz

source fun.sh

#-------------------------------------------------------

function newton_step {
	local _z="$1 $2"
	local _f=`f $_z`
	local _df=`d_f $_z`
	local _f=`div $_f $_df`
	echo `sub $_z $_f`; }

function newton {
	# $3 -- prec
	# $4 -- file
	local _z1="$1 $2"
	local _z2=`add $_z1 1 1`
	while [[ $(echo "$(abs `sub $_z1 $_z2`) > $3" | bc -l) = 1 ]]; do
	#	echo -ne "${_z1},\t" >> $4
		echo -ne "`printf "%.19f %.19f" ${_z1}`\t" >> $4
		_z2="$_z1"
		_z1=`newton_step $_z2`
	done
	echo -ne "|" >> $4
	echo $_z1; }

#-------------------------------------------------------

function to_file {
	# $1 $2 root
	# $3 -- prec
	# $4 -- file
	p=$((`printf %.0f $(echo "-l($3)/l(10)+0.5" | bc -l)`-1))
	local _a=`printf "%.${p}f" $1`
	local _b=`printf "%.${p}f" $2`
	echo -ne "$_a $_b" >> $4; }

function run {
	# $1 -- re_max
	# $2 -- im_max
	# $3 -- step
	# $4 -- prec
	# $5 -- file
	local sr="1"
	local si="1"
	if [[ $1 -lt 0 ]]; then
		local sr="-1"; fi
	if [[ $2 -lt 0 ]]; then
		local si="-1"; fi
	for r in `seq 0 $sr $(echo "$1/$3" | bc)`; do
		for i in `seq 0 $si $(echo "$2/$3" | bc)`; do
			local _z="`echo "$r*$3" | bc -l` `echo "$i*$3" | bc -l`"
			local _root=`newton $_z $4 .${5}q.dat`
			to_file $_root $4 ".${5}q.dat"
			echo >> ".${5}q.dat"
		done
	done; }

#=======================================================

# $1 -- re_max
# $2 -- im_max
# $3 -- step
# $4 -- prec

if [[ $1 -gt 0 ]]; then
	if [[ $2 -gt 0 ]]; then
		file="1"
	else
		file="4"
	fi
else
	if [[ $2 -gt 0 ]]; then
		file="2"
	else
		file="3"
	fi
fi

rm .${file}q.dat
run $1 $2 $3 $4 $file