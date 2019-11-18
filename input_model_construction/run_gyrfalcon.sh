#!/bin/bash
echo gyrfalcON in="$1_magalie" out="$1_gyrfalcon" tstop=1000 step=50 logfile="$1_gyrfalcon.log" stopfile=stop logstep=50 kmax=6 eps=0.05 give=mxv startout=t lastout=t > "$1_gyrfalcon.stdout" 2>&1
