#! /usr/bin/env bash

MYFILE=${1:-"./ddai-gin-perf.log"}
IP_ADDR=${2:-"ctr2-alola-login-01.adc.amd.com"}

rsync -avzL $MYFILE dondai@${IP_ADDR}:/home/AMD/dondai/rocm-systems.git/

