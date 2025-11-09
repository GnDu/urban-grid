import React from "react";

const LEGEND = [
  ["barren", "#8B4513"],
  ["residence", "#4A90E2"],
  ["greenery", "#4CAF50"],
  ["industry", "#FFA500"],
  ["service", "#FFD700"],
  ["road", "#333333"],
];

export default function Legend() {
  return (
    <div className="flex flex-wrap justify-center gap-4 bg-gray-900 text-white p-3 rounded-xl shadow-md">
      {LEGEND.map(([label, color]) => (
        <div key={label} className="flex items-center gap-1">
          <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: color }} />
          <span className="text-sm">{label}</span>
        </div>
      ))}
    </div>
  );
}
