#!/bin/bash

# Remove logs directory
echo "Removing /home/ubuntu/oml-exploration/logs..."
rm -rf /home/ubuntu/oml-exploration/logs

# Remove saved_models and all_run_logs.txt from results directory
echo "Removing saved_models and all_run_logs.txt from /ephemeral/oml-exploration-results..."
rm -rf /ephemeral/oml-exploration-results/saved_models
rm -rf /ephemeral/oml-exploration-results/logs
rm -rf /ephemeral/oml-exploration-results/all_run_logs.txt

echo "Cleanup completed!" 