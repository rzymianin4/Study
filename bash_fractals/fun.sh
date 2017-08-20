#!/bin/bash
#Dawid Tracz

source cpx.sh

function f {
	local _z="$1 $2"
	local _z2=`mult $_z $_z`
	local _z3=`mult $_z2 $_z`
	echo `sub $_z3 1 0`; }

function d_f {
	local _z="$1 $2"
	local _z2=`mult $_z $_z`
	echo `mult 3 0 $_z2`; }