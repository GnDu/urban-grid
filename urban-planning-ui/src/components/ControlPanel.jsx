import React from "react";

export default function ControlPanel({ running, onStart, onStop, onReset, speed, setSpeed }) {
  return (
    <div className="flex items-center gap-4 bg-gray-900 p-4 rounded-xl shadow-md">
      {running ? (
        <button onClick={onStop} className="bg-red-600 px-4 py-2 rounded-lg hover:bg-red-700">⏸ Pause</button>
      ) : (
        <button onClick={onStart} className="bg-green-600 px-4 py-2 rounded-lg hover:bg-green-700">▶️ Start</button>
      )}
      <button onClick={onReset} className="bg-yellow-600 px-4 py-2 rounded-lg hover:bg-yellow-700">🔄 Reset</button>
      <div className="flex items-center gap-2">
        <span>Speed:</span>
        <input type="range" min="100" max="1000" step="100" value={speed}
          onChange={(e) => setSpeed(e.target.value)} />
      </div>
    </div>
  );
}
