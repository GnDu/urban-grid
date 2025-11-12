import React, { useState, useEffect, useRef } from "react";
import GridWorld from "./components/GridWorld";
import MetricsPanel from "./components/MetricsPanel";
import ControlPanel from "./components/ControlPanel";
import Legend from "./components/Legend";
import { getState, stepSim, resetSim } from "./api";

export default function App() {
  const size = 20;
  const types = ["BARREN", "RESIDENCE", "GREENERY", "INDUSTRY", "SERVICE", "ROAD"];
  const models = [
  { id: 1, name: "Model 1" },
  { id: 2, name: "Model 2" },
  { id: 3, name: "Model 3" },
  ];

  const [grid, setGrid] = useState(createGrid(size, types));
  const [tick, setTick] = useState(0);
  const [population, setPopulation] = useState(0);
  const [pollution, setPollution] = useState(0);
  const [history, setHistory] = useState([]);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(500);
  const intervalRef = useRef(null);
  const chartUpdateRef = useRef(null);
  const historyRef = useRef([]);
  const [selectedModel, setSelectedModel] = useState(models[0].id);

  function createGrid(size, types) {
    return Array.from({ length: size }, () =>
      Array.from({ length: size }, () => types[Math.floor(Math.random() * types.length)])
    );
  }

  async function fetchFromBackend() {
    const data = await getState();
    if (data) {
      setGrid(data.grid || createGrid(size, types));
      setTick(data.tick);
      setPopulation(data.population);
      setPollution(data.pollution);
      historyRef.current = [];
      return true;
    }
    return false;
  }

  async function step() {
    const data = await stepSim();
    if (data) {
      if (data.message === "Simulation complete") {
        handleStop();
        return;
      }

      setGrid(data.grid);
      setTick(data.tick);
      setPopulation(data.population);
      setPollution(data.pollution);

      // accumulate history in ref
      historyRef.current.push({
        tick: data.tick,
        population: data.population,
        pollution: data.pollution,
      });
    } else {
      // fallback simulation
      setGrid(createGrid(size, types));
      setTick((t) => t + 1);
      setPopulation((p) => p + 20);
      setPollution((p) => p + 10);

      historyRef.current.push({
        tick: tick + 1,
        population: population + 20,
        pollution: pollution + 10,
      });
    }
  }

  const handleStart = () => {
    setRunning(true);
    intervalRef.current = setInterval(step, speed);

    // Update chart every 10 seconds
    chartUpdateRef.current = setInterval(() => {
      setHistory([...historyRef.current]);
    }, 10000);
  };

  const handleStop = () => {
    setRunning(false);
    clearInterval(intervalRef.current);
    clearInterval(chartUpdateRef.current);
  };

  const handleReset = async () => {
    handleStop();
    const data = await resetSim(selectedModel);
    setGrid(data?.grid || createGrid(size, types));
    setTick(0);
    setPopulation(0);
    setPollution(0);
    historyRef.current = [];
    setHistory([]);
  };

  useEffect(() => {
    fetchFromBackend();
  }, []);

  useEffect(() => {
    if (running) {
      clearInterval(intervalRef.current);
      intervalRef.current = setInterval(step, speed);
    }
  }, [speed]);

  return (
    <div className="min-h-screen flex flex-col items-center gap-6 p-6">
      <h1 className="text-4xl font-bold">🌆 Urban Planning Simulation</h1>

      <div className="flex items-center gap-4 bg-gray-900 p-4 rounded-xl shadow-m">
        <span className="text-lg">Model:</span>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(Number(e.target.value))}
          disabled={running}
          className="border rounded-md px-3 py-1.5 text-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-900">
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
      </div>

      <ControlPanel running={running} onStart={handleStart} onStop={handleStop} onReset={handleReset} speed={speed} setSpeed={setSpeed} />
      <Legend />
      <GridWorld grid={grid} />
      <MetricsPanel history={history} tick={tick} population={population} pollution={pollution} />
    </div>
  );
}
