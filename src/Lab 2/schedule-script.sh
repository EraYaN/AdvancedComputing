#!/bin/bash

FIRSTTIMEOUT=2
TIMEOUT=3

if [[ $# -lt 4 ]]; then
    echo "Usage: <executable> <image> <runid> <profiling;yes|no> <arguments>"
	exit 1
fi
arguments=""
if [[ $# -eq 5 ]]; then
arguments=$5
fi

rm -f ~/run/*
echo Cleaned run directory
mkdir -p run_output
echo Created output directory
cat <<EOT >> ~/run/run_settings.ini
[general]
exec_args = images/$2 run_output/benchmark-$2 $arguments
working_dir = /data/home/group27/projects/Lab 2/
[profiling]
enable = $4
all_metrics = no
all_events = no
custom_options = --metrics all --csv --normalized-time-unit ns
EOT
#cp run_settings.ini ~/run/run_settings.ini
echo Written run settings for image: $2
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
DATE=`date +%Y-%m-%dT%H-%M-%S`
echo Run at $3
mkdir -p run_output/$3
cp -f ~/run/* run_output/$3
rm run_output/$3/gpu_program.old
if [ -f trace.nvprof ]; then
	mv -f trace.nvprof run_output/$3
fi
echo Done