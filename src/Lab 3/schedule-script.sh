#!/bin/bash

FIRSTTIMEOUT=2
TIMEOUT=3

if [[ $# -lt 3 ]]; then
    echo "Usage: <executable> <runid> <profiling;yes|no> <arguments>"
	exit 1
fi
arguments=""
if [[ $# -eq 4 ]]; then
arguments=$4
fi

rm -f ~/run/*
echo Cleaned run directory
mkdir -p run_output
echo Created output directory
cat <<EOT >> ~/run/run_settings.ini
[general]
exec_args = $arguments
working_dir = /data/home/group27/projects/Lab 3/bin/
[profiling]
enable = $3
all_metrics = no
all_events = no
custom_options = --metrics all --csv --normalized-time-unit ns
EOT
#cp run_settings.ini ~/run/run_settings.ini
echo Written run settings for $1
cp $1 ~/run/gpu_program
echo Copied files
echo Sleeping $FIRSTTIMEOUT seconds
sleep $TIMEOUT
while [ ! -f ~/run/gpu_program.old ]
do
	#PID=`ps aux | grep -i gpu_program | awk {'print $2'}`
	echo Waiting for `ps aux | grep [a]cs.*gpu_program | awk {'print $11'}`
	echo Sleeping $TIMEOUT seconds
	sleep $TIMEOUT
done
echo Copy back output
echo Run with run ID $2
mkdir -p run_output/$2
cp -f ~/run/* run_output/$2
rm run_output/$2/gpu_program.old
if [ -f trace.nvprof ]; then
	mv -f trace.nvprof run_output/$2
fi
echo Done