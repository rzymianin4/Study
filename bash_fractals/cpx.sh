#!/bin/bash
#Dawid Tracz

function con {
	local _b=`echo "(-1.0)*$2" | bc -l`
	echo "$1 $_b"; }

function mult {
	local _a=`echo "$1*$3 - $2*$4" | bc -l`
	local _b=`echo "$1*$4 + $2*$3" | bc -l`
	echo "$_a $_b"; }

function div_by_R {
	local _a=`echo "$1/$3" | bc -l`
	local _b=`echo "$2/$3" | bc -l`
	echo "$_a $_b"; }

function div {
	local _z1="$1 $2"
	local _z2=`con $3 $4`
	local _z1=`mult $_z1 $_z2`
	local _denom=`echo "$3*$3 + $4*$4" | bc -l`
	echo `div_by_R $_z1 $_denom`; }

function add {
	local _a=`echo "$1+$3" | bc -l`
	local _b=`echo "$2+$4" | bc -l`
	echo "$_a $_b"; }

function sub {
	local _b1=`echo "(-1.0)*$3" | bc -l`
	local _b2=`echo "(-1.0)*$4" | bc -l`
	echo `add $1 $2 $_b1 $_b2`; }

function abs {
	echo `echo "sqrt($1*$1 + $2*$2)" | bc -l`; }