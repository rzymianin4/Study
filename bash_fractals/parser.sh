#!/bin/bash
#Dawid Tracz

function unsigner {
	local _a=$1
	local _b=$2
	if [[ `echo "${_a}==0" | bc` = 1 ]]; then
		local _a="0"; fi
	if [[ `echo "${_b}==0" | bc` = 1 ]]; then
		local _b="0"; fi
	echo "$_a $_b"; }

function app_to {
	f=$1
	shift
	while [[ $1 ]]; do
		echo "$1 $2" >> $f
		shift 2
	done; }

#-------------------------------------------------------

declare -A files
nor=0

l=0

while IFS='' read -r line || [[ -n "$line" ]]; do
	points=`echo $line | cut -d "|" -"f1"`
	root=`echo $line | cut -d "|" -"f2"`
	if [[ $root = "" ]]; then
		continue; fi
	root=`unsigner $root`

	if [[ ${files[$root]} == "" ]]; then
		((nor++))
		files[$root]=".${nor}r.dat"
		echo "# $root (index $((${nor}-1)))" > ${files[$root]}; fi

	app_to ${files[$root]} $points
done < ".$1"

echo "# data set" > $1
for i in `seq 1 $nor`; do
	cat ".${i}r.dat" >> $1
	echo >> "$1"
	echo >> "$1"
	rm ".${i}r.dat"
done