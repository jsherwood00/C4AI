#!/bin/bash
# Usage: ./clean_gemini_log.sh input.txt > clean_output.txt

if [ -z "$1" ]; then
    echo "Usage: $0 <input_file> > <output_file>"
    exit 1
fi

sed 's/\r//g' "$1" | \
sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | \
sed 's/\x1b\][^\x07]*\x07//g' | \
sed 's/\x1b\[?[0-9]*[a-z]//g' | \
grep -v "screen reader" | \
grep -v "settings.json" | \
grep -v "disappear on next run" | \
grep -v "256-color" | \
grep -v "^workspace (/directory)" | \
grep -v "^/workspace sandbox" | \
grep -v "^no sandbox /model" | \
grep -v "^gemini-3.1-pro-preview$" | \
grep -v "trust the files" | \
grep -v "Trusting a folder" | \
grep -v "custom commands, hooks" | \
grep -v "configurations could execute" | \
grep -v "behavior of the" | \
grep -v "Trust folder" | \
grep -v "Trust parent" | \
grep -v "Don't trust" | \
grep -v "^CLI\.$" | \
grep -v "Waiting for authentication" | \
grep -v "Ready (workspace)" | \
grep -v "(Tab to focus)" | \
grep -v "^⊶\|^⊷" | \
grep -v "^$" | \
awk '!seen[$0]++' | \
cat -s
