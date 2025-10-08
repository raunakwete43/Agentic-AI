#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# --- Configuration ---
# Default sort order: descending (largest first)
DEFAULT_SORT_ORDER="desc"
# Supported sort orders
SUPPORTED_SORT_ORDERS=("asc" "desc")

# --- Functions ---

# Function to display usage information
usage() {
  echo "Usage: $(basename "$0") [-s <asc|desc>]"
  echo "  Displays disk usage sorted by size."
  echo ""
  echo "Options:"
  echo "  -s <asc|desc>  Sort order: 'asc' for ascending, 'desc' for descending (default: ${DEFAULT_SORT_ORDER})."
  echo "  -h             Display this help message."
  echo ""
  echo "Example:"
  echo "  $(basename "$0")"
  echo "  $(basename "$0") -s asc"
}

# Function to validate sort order
validate_sort_order() {
  local order="$1"
  for supported_order in "${SUPPORTED_SORT_ORDERS[@]}"; do
    if [[ "${order}" == "${supported_order}" ]]; then
      return 0 # Valid
    fi
  done
  return 1 # Invalid
}

# Function to display disk usage
display_disk_usage() {
  local sort_order="$1"
  local sort_flags=""

  # Set sort flags based on the desired order
  if [[ "${sort_order}" == "desc" ]]; then
    sort_flags="-r" # Use -r for descending order
  fi

  # Use 'df -hP' for POSIX-compliant output, '-P' prevents line wrapping
  # Pipe to 'sort -k4 -h' to sort by the size column (column 4) in human-readable format
  # '-h' handles human-readable numbers (e.g., 1K, 2M, 3G)
  # The 'tail -n +2' removes the header line from 'df' output before sorting
  # The 'head -n -1' removes the last line if it's a summary line (e.g., 'total')
  if ! df -hP | tail -n +2 | head -n -1 | sort -k4 -h ${sort_flags}; then
    echo "Error: Failed to retrieve or sort disk usage information." >&2
    return 1
  fi
}

# --- Main Script Logic ---

# Initialize variables
sort_order="${DEFAULT_SORT_ORDER}"

# Parse command-line options
while getopts ":s:h" opt; do
  case ${opt} in
    s )
      sort_order="${OPTARG}"
      if ! validate_sort_order "${sort_order}"; then
        echo "Error: Invalid sort order '${sort_order}'. Supported orders are: ${SUPPORTED_SORT_ORDERS[*]}" >&2
        usage >&2
        exit 1
      fi
      ;;
    h )
      usage
      exit 0
      ;;
    \? )
      echo "Error: Invalid option: -${OPTARG}" >&2
      usage >&2
      exit 1
      ;;
    : )
      echo "Error: Option -${OPTARG} requires an argument." >&2
      usage >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# Check if any non-option arguments remain
if [[ $# -gt 0 ]]; then
  echo "Error: Unexpected arguments: $*" >&2
  usage >&2
  exit 1
fi

# Display disk usage with the determined sort order
if ! display_disk_usage "${sort_order}"; then
  exit 1
fi

exit 0