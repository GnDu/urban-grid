import React from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend } from "recharts";

export default function MetricsPanel({ history, tick, population, pollution }) {
  return (
    <div className="bg-gray-900 text-white p-4 rounded-xl shadow-md w-full max-w-[480px]">
      <div className="flex justify-between mb-3 text-sm">
        <span>Tick: <b>{tick}</b></span>
        <span>Population: <b className="text-blue-400">{population}</b></span>
        <span>Pollution: <b className="text-red-400">{pollution}</b></span>
      </div>
      <LineChart width={440} height={200} data={history}>
        <XAxis dataKey="tick" stroke="#ccc" />
        <YAxis stroke="#ccc" />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="population" stroke="#4A90E2" />
        <Line type="monotone" dataKey="pollution" stroke="#E94E77" />
      </LineChart>
    </div>
  );
}
