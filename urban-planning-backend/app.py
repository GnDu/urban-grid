from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import logging

app = Flask(__name__)
CORS(app)

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# TO CHANGE
MODEL = {
    1: "dqn_selva.json",
    2: "BK_SequencedBuildOrderAgent.json",
    3: "dqn_charles.json"
}

DEFAULT_MODEL_ID = 1

with open("steps_with_totals.json") as f:
    steps_data = json.load(f)

# Keep track of current step index in memory
current_step = 0
types = ["BARREN", "RESIDENCE", "GREENERY", "INDUSTRY", "SERVICE", "ROAD"]
history = []

# TO CHANGE
size = 10

def create_grid(size, types):
    return [["BARREN" for _ in range(size)] for _ in range(size)]
    
# Store the current grid and tick
current_grid = create_grid(size, types)

@app.route("/state", methods=["GET"])
def get_state():
    return jsonify({
        "grid": current_grid,
        "tick": current_step,
        "population": 0,
        "pollution": 0
    })

@app.route("/step", methods=["POST"])
def step():
    """Return the next step from JSON file."""
    global current_step, current_grid
    logging.info(f"Received /step request | Current step index: {current_step}")

    if current_step >= len(steps_data):
        logging.info("Simulation complete — no more steps.")
        return jsonify({"message": "Simulation complete"}), 200

    # Get the next action
    action = steps_data[current_step]
    current_step += 1

    logging.info(f"Applying action: {action}")

    # Apply the action to the grid
    row = action["row"]
    col = action["col"]
    tile_type = action["tile_type"].upper()  # ensure uppercase
    current_grid[row][col] = tile_type
    logging.info(f"Returning action: {action}")

    response = {
        "grid": current_grid,
        "tick": action["step"],
        "population": action["total_population"],
        "pollution": action["total_pollution"],    
        "action": action
    }

    logging.info(f"Response data: {response}")

    return jsonify(response)

@app.route("/reset", methods=["POST"])
def reset():
    global current_step, current_grid, steps_data
    current_step = 0

    req = request.get_json() or {}
    model_id = req.get("model_id", DEFAULT_MODEL_ID)

    if model_id == 3:
        size = 20
    else:
        size = 10

    current_grid = create_grid(size, types)

    file_name = MODEL.get(model_id, MODEL[DEFAULT_MODEL_ID])
    logging.info(f"Resetting simulation for model {model_id}, loading {file_name}")
    with open(file_name) as f:
        steps_data = json.load(f)


    return jsonify({"message": "Simulation reset", "grid": current_grid, "tick": 0, "population": 0, "pollution": 0})

if __name__ == "__main__":
    app.run(debug=True)