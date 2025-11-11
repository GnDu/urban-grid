import axios from "axios";

const API_BASE = "http://127.0.0.1:5000"; // Change if backend is hosted elsewhere

export const getState = async () => {
  try {
    const res = await axios.get(`${API_BASE}/state`);
    return res.data;
  } catch {
    return null;
  }
};

export const stepSim = async () => {
  try {
    const res = await axios.post(`${API_BASE}/step`);
    return res.data;
  } catch {
    return null;
  }
};

export const resetSim = async () => {
  try {
    const res = await axios.post(`${API_BASE}/reset`);
    return res.data;
  } catch {
    return null;
  }
};
