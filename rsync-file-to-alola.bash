#! /usr/bin/env bash

MYFILE=${1:-"./ddai-gin-perf.log"}

rsync -avzL $MYFILE dondai@ctr2-alola-login-01.adc.amd.com:/home/AMD/dondai/rocm-systems.git/

