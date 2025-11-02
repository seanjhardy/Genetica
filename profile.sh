#!/bin/bash
# Quick profiling script for Genetica

cd "$(dirname "$0")/build" || exit 1

echo "Starting Genetica from build directory..."
./Genetica.app/Contents/MacOS/Genetica &
APP_PID=$!

echo "Genetica PID: $APP_PID"
echo "Waiting 3 seconds for app to start..."
sleep 3

# Check if process is still running
if ! ps -p $APP_PID > /dev/null; then
    echo "Error: App crashed during startup"
    exit 1
fi

echo "Profiling for 20 seconds..."
sample $APP_PID 20 -f profile_output.txt

echo ""
echo "Profile complete! Results saved to build/profile_output.txt"
echo ""
echo "Top functions by CPU time:"
grep -A 20 "Call graph:" profile_output.txt 2>/dev/null | head -30 || echo "Profile data not available"

cd ..

# Keep app running so you can interact
echo ""
echo "App is still running (PID: $APP_PID)"
echo "Press Ctrl+C when done, or run: kill $APP_PID"
wait $APP_PID

