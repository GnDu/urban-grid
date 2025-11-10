import React from "react";

// Define colors for each type of land cell
const COLORS = {
  BARREN: "#8B4513",    // brown
  RESIDENCE: "#4A90E2", // blue
  GREENERY: "#4CAF50",  // green
  INDUSTRY: "#FFA500",  // orange
  SERVICE: "#FFD700",   // yellow
  ROAD: "#333333",      // dark gray
};

export default function GridWorld({ grid }) {
  // Handle null or undefined grid
  if (!grid || !Array.isArray(grid)) return null;

  return (
    <div
      className="grid gap-[2px] bg-gray-800 p-3 rounded-xl shadow-lg"
      // ✅ Dynamic grid layout (auto adjusts column count)
      style={{
        gridTemplateColumns: `repeat(${grid.length}, minmax(0, 1fr))`,
      }}
    >
      {grid.map((row, y) =>
        row.map((cell, x) => (
          <div
            key={`${x}-${y}`}
            className="w-5 h-5 rounded-sm"
            style={{ backgroundColor: COLORS[cell] || "#555" }}
          />
        ))
      )}
    </div>
  );
}
