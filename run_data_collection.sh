#!/bin/bash

# Data Collection Workflow Script for PhaseScale
# This script automates the entire data collection process

set -e

# Configuration
OUTPUT_DIR="./data"
METRICS_DIR="$OUTPUT_DIR/metrics"
PROCESSED_DIR="$OUTPUT_DIR/processed"
VISUALIZATIONS_DIR="$OUTPUT_DIR/visualizations"
WORKLOAD_DATA_DIR="$OUTPUT_DIR/workloads"

# Create directory structure
mkdir -p "$METRICS_DIR" "$PROCESSED_DIR" "$VISUALIZATIONS_DIR" "$WORKLOAD_DATA_DIR"

# Parse command line arguments
WORKLOAD_TYPE=""
DURATION=600  # Default duration: 10 minutes
COLLECT_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --workload)
      WORKLOAD_TYPE="$2"
      shift 2
      ;;
    --sequence)
      SEQUENCE_NAME="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --collect-only)
      COLLECT_ONLY=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--workload <workload_path>] [--sequence <sequence_name>] [--duration <seconds>] [--collect-only]"
      exit 1
      ;;
  esac
done

# Start timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="run_${TIMESTAMP}"
RUN_DIR="$OUTPUT_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "=== PhaseScale Data Collection ==="
echo "Run ID: $RUN_ID"
echo "Output directory: $RUN_DIR"

# Create a metadata file
cat > "$RUN_DIR/metadata.txt" << EOL
Run ID: $RUN_ID
Timestamp: $TIMESTAMP
Duration: $DURATION seconds
EOL

if [ -n "$WORKLOAD_TYPE" ]; then
  echo "Workload: $WORKLOAD_TYPE" >> "$RUN_DIR/metadata.txt"
fi

if [ -n "$SEQUENCE_NAME" ]; then
  echo "Sequence: $SEQUENCE_NAME" >> "$RUN_DIR/metadata.txt"
fi

# Start metrics collection
echo "Starting metrics collection..."
METRICS_SESSION_DIR="$METRICS_DIR/session_${TIMESTAMP}"
mkdir -p "$METRICS_SESSION_DIR"

# Start the metrics collector in the background
./metrics_collector.py --output-dir="$METRICS_DIR" --system-interval=1.0 --detailed-interval=5.0 &
METRICS_PID=$!

echo "Metrics collector started with PID $METRICS_PID"
echo "Metrics collector PID: $METRICS_PID" >> "$RUN_DIR/metadata.txt"

# If collect-only flag is set, just run the metrics collector
if [ "$COLLECT_ONLY" = true ]; then
  echo "Running in collect-only mode. Press Ctrl+C to stop data collection."
  
  # Wait for the user to press Ctrl+C
  trap "echo 'Stopping metrics collection...'; kill $METRICS_PID; wait $METRICS_PID 2>/dev/null; exit 0" INT TERM
  wait $METRICS_PID
  exit 0
fi

# Run the workload if specified
if [ -n "$WORKLOAD_TYPE" ]; then
  echo "Running workload: $WORKLOAD_TYPE"
  ./workload_runner.py --config=workload_config.yaml --output-dir="$WORKLOAD_DATA_DIR" workload "$WORKLOAD_TYPE" &
  WORKLOAD_PID=$!
  echo "Workload started with PID $WORKLOAD_PID"
  echo "Workload PID: $WORKLOAD_PID" >> "$RUN_DIR/metadata.txt"
elif [ -n "$SEQUENCE_NAME" ]; then
  echo "Running sequence: $SEQUENCE_NAME"
  ./workload_runner.py --config=workload_config.yaml --output-dir="$WORKLOAD_DATA_DIR" sequence "$SEQUENCE_NAME" &
  WORKLOAD_PID=$!
  echo "Workload sequence started with PID $WORKLOAD_PID"
  echo "Workload sequence PID: $WORKLOAD_PID" >> "$RUN_DIR/metadata.txt"
else
  echo "No workload or sequence specified. Collecting system metrics only."
  WORKLOAD_PID=""
fi

# Wait for the specified duration
echo "Collecting data for $DURATION seconds..."
sleep $DURATION

# Stop the workload if it's still running
if [ -n "$WORKLOAD_PID" ]; then
  if kill -0 $WORKLOAD_PID 2>/dev/null; then
    echo "Stopping workload..."
    kill $WORKLOAD_PID
    wait $WORKLOAD_PID 2>/dev/null
  fi
fi

# Stop the metrics collector
echo "Stopping metrics collection..."
kill $METRICS_PID
wait $METRICS_PID 2>/dev/null

echo "Data collection completed."

# Process the collected metrics
echo "Processing collected metrics..."
PROCESSED_OUTPUT="$RUN_DIR/processed"
mkdir -p "$PROCESSED_OUTPUT"

./process_metrics.py --metrics-dir="$METRICS_SESSION_DIR" --output-dir="$PROCESSED_OUTPUT" --output-prefix="$RUN_ID"

# Generate visualizations
echo "Generating visualizations..."
VISUALIZATIONS_OUTPUT="$RUN_DIR/visualizations"
mkdir -p "$VISUALIZATIONS_OUTPUT"

if [ -f "$PROCESSED_OUTPUT/${RUN_ID}_merged.csv" ]; then
  ./visualize_metrics.py --data-file="$PROCESSED_OUTPUT/${RUN_ID}_merged.csv" --output-dir="$VISUALIZATIONS_OUTPUT"
else
  echo "No merged metrics file found. Skipping visualizations."
fi

echo "=== Data collection workflow completed ==="
echo "Results saved to: $RUN_DIR"
echo "Raw metrics: $METRICS_SESSION_DIR"
echo "Processed data: $PROCESSED_OUTPUT"
echo "Visualizations: $VISUALIZATIONS_OUTPUT"
