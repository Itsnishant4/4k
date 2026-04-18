#!/bin/bash

# Start the Backend
echo "Starting Backend Server..."
cd backend
source venv/bin/activate
export PATH="/opt/homebrew/bin:$PATH"
python main.py &
BACKEND_PID=$!

# Start the Frontend
echo "Starting Frontend Server..."
cd ../frontend
export PATH="/opt/homebrew/bin:$PATH"
npm run dev &
FRONTEND_PID=$!

echo "Servers are running!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Open your browser at http://localhost:3000"

# Keep script running
wait
