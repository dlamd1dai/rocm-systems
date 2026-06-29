#! /usr/bin/env bash

MYFILE=${1:-"./ddai-gin-perf.log"}
# IP_ADDR=${2:-"ctr2-alola-login-01.adc.amd.com"}
IP_ADDR=${2:-"ctr-cx63-mi300x-12.adc.amd.com"}

rsync -avzL $MYFILE dondai@${IP_ADDR}:/home/AMD/dondai/rocm-systems.git/

