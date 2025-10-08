#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 <file_path> <word_to_count>"
  echo "Counts the number of occurrences of a specific word in a given file."
  echo ""
  echo "Arguments:"
  echo "  file_path       The path to the text file to analyze."
  echo "  word_to_count   The word to search for and count."
  exit 1
}

count_word_in_file() {
  local file_path="$1"
  local word_to_count="$2"

  if [[ ! -f "$file_path" ]]; then
    echo "Error: File not found at '$file_path'." >&2
    exit 2
  fi

  if [[ -z "$word_to_count" ]]; then
    echo "Error: Word to count cannot be empty." >&2
    exit 3
  fi

  local count
  count=$(grep -o -w "$word_to_count" "$file_path" | wc -l)

  echo "The word '$word_to_count' appears $count times in '$file_path'."
}

main() {
  if [[ $# -ne 2 ]]; then
    usage
  fi

  local file_path="$1"
  local word_to_count="$2"

  count_word_in_file "$file_path" "$word_to_count"
}

main "$@"