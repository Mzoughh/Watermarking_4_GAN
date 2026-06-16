#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_folder_or_tfevents_file>"
  exit 1
fi

PATH_INPUT="$1"

if [[ -f "$PATH_INPUT" && "$PATH_INPUT" == *.tfevents* ]]; then
  LOGDIR=$(dirname "$PATH_INPUT")
else
  LOGDIR="$PATH_INPUT"
fi

if [ ! -d "$LOGDIR" ]; then
  echo "Error: directory $LOGDIR does not exist."
  exit 1
fi

echo "Launching TensorBoard with logdir: $LOGDIR"
echo "Open http://localhost:6006 in your browser."
tensorboard --logdir="$LOGDIR" --port=6006 --host=localhost
