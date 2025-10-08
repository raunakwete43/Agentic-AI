#!/usr/bin/env bash

set -euo pipefail

MAX_CPU_USAGE=1
CHECK_INTERVAL=5
DURATION=30

check_cpu_usage() {
  local cpu_usage
  cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')

  if (( $(echo "$cpu_usage > $MAX_CPU_USAGE" | bc -l) )); then
    echo "WARNING: CPU usage is at ${cpu_usage}% which is above the threshold of ${MAX_CPU_USAGE}%."
  fi
}

run_monitor() {
  local start_time
  start_time=$SECONDS
  local end_time
  end_time=$((start_time + DURATION))

  while (( SECONDS < end_time )); do
    check_cpu_usage
    local remaining_time=$((end_time - SECONDS))
    if (( remaining_time < CHECK_INTERVAL )); then
      sleep "$remaining_time"
    else
      sleep "$CHECK_INTERVAL"
    fi
  done
}

main() {
  run_monitor
}

main