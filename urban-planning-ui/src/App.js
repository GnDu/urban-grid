import React, { useState, useEffect, useRef } from "react";
import GridWorld from "./components/GridWorld";
import MetricsPanel from "./components/MetricsPanel";
import ControlPanel from "./components/ControlPanel";
import Legend from "./components/Legend";
import { getState, stepSim, resetSim } from "./api";

export default function App() {
  const size = 20;
  const types = ["barren", "residence", "greenery", "industry", "service", "road"];

  const [grid, setGrid] = useState(createGrid(size, types));
  const [tick, setTick] = useState(0);
  const [population, setPopulation] = useState(0);
  const [pollution, setPollution] = useState(0);
  const [history, setHistory] = useState([]);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(500);
  const intervalRef = useRef(null);

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
      setHistory(data.history || []);
      return true;
    }
    return false;
  }

  async function step() {
    const data = await stepSim();
    if (data) {
      setGrid(data.grid);
      setTick(data.tick);
      setPopulation(data.population);
      setPollution(data.pollution);
      setHistory(data.history);
    } else {
      // fallback local sim
      setGrid(createGrid(size, types));
      setTick((t) => t + 1);
      setPopulation((p) => p + 20);
      setPollution((p) => p + 10);
      setHistory((h) => [...h, { tick: tick + 1, population, pollution }]);
    }
  }

  const handleStart = () => {
    setRunning(true);
    intervalRef.current = setInterval(step, speed);
  };

  const handleStop = () => {
    setRunning(false);
    clearInterval(intervalRef.current);
  };

  const handleReset = async () => {
    handleStop();
    const data = await resetSim();
    if (data) {
      setGrid(data.grid);
      setTick(0);
      setPopulation(0);
      setPollution(0);
      setHistory([]);
    } else {
      setGrid(createGrid(size, types));
      setTick(0);
      setPopulation(0);
      setPollution(0);
      setHistory([]);
    }
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
      <ControlPanel running={running} onStart={handleStart} onStop={handleStop} onReset={handleReset} speed={speed} setSpeed={setSpeed} />
      <Legend />
      <GridWorld grid={grid} />
      <MetricsPanel history={history} tick={tick} population={population} pollution={pollution} />
    </div>
  );
}
