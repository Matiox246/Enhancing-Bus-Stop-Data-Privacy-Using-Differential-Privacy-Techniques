import pandas as pd
import numpy as np
from graphviz import Digraph

# Data with Station Names and Monthly Average Passengers
data = {
    "Station": ["L1", "L21", "L22", "L23", "L31", "L32", "L33", "L34", "L35"],
    "Average Passengers": [35, 20, 15, 25, 12, 18, 8, 10, 14],
}

df = pd.DataFrame(data)

# Define Station Hierarchy (Parent-Child Relationships)
routes = {
    "L1": ["L21", "L22", "L23"],
    "L21": ["L31", "L32", "L33"],
    "L22": ["L33", "L34"],
    "L23": ["L35"],
}

# Function for Dynamic Epsilon Assignment
def calculate_epsilon(data, base_epsilon=0.5, decay_rate=0.1):
    """
    Assigns dynamic epsilon values with decreasing noise as we move down the hierarchy.
    Epsilon values reduce based on the depth level of each station.
    """
    epsilon_values = {}
    depth = {station: 0 for station in data["Station"]}

    # Calculate depth for each station in the hierarchy
    for parent, children in routes.items():
        for child in children:
            depth[child] = depth[parent] + 1

    # Assign epsilon dynamically
    max_depth = max(depth.values())
    for station, station_depth in depth.items():
        epsilon_values[station] = base_epsilon - (station_depth * decay_rate)
        epsilon_values[station] = max(0.1, epsilon_values[station])  # Ensure epsilon >= 0.1

    return epsilon_values

# Assign Epsilon to Stations
epsilon_values = calculate_epsilon(df)
df["Epsilon"] = df["Station"].map(epsilon_values)

# Differential Privacy with Laplace Noise
def add_laplace_noise(data, sensitivity, epsilon):
    """
    Adds Laplace noise to passenger data while ensuring non-negative counts.
    """
    noise = np.random.laplace(0, sensitivity / epsilon, size=len(data))
    return np.maximum(0, np.round(data + noise))  # Ensure non-negative values

# Add Laplace Noise to Passengers
sensitivity = 1  # Sensitivity of one passenger
df["Noisy Passengers"] = add_laplace_noise(df["Average Passengers"], sensitivity, df["Epsilon"])

# Enforce Parent-Child Constraints
def enforce_parent_child_constraints(routes, station_data):
    """
    Ensures the sum of children values does not exceed the parent value.
    Adjusts children values dynamically while preserving relative proportions.
    """
    for parent, children in routes.items():
        parent_noisy = station_data.loc[station_data["Station"] == parent, "Noisy Passengers"].iloc[0]
        child_values = station_data[station_data["Station"].isin(children)]["Noisy Passengers"]

        if child_values.sum() > parent_noisy:
            # Scale children values to respect the parent constraint
            scale = parent_noisy / child_values.sum()
            station_data.loc[station_data["Station"].isin(children), "Noisy Passengers"] = np.floor(
                child_values * scale
            )
    return station_data

df = enforce_parent_child_constraints(routes, df)

# Generate Tree Visualization
def generate_tree(data, routes):
    """
    Generates a tree diagram for Vancouver's bus network.
    """
    dot = Digraph(format='svg')
    dot.attr(rankdir="TB", size="12,8")
    dot.attr("node", shape="ellipse", style="filled", fillcolor="lightblue")

    # Add nodes with both original and noisy data
    for _, row in data.iterrows():
        station = row["Station"]
        orig_label = f"Original: {row['Average Passengers']}"
        noisy_label = f"Noisy: {row['Noisy Passengers']}"
        dot.node(station, f"{station}\\n{orig_label}\\n{noisy_label}")

    # Add edges for each route
    for parent, children in routes.items():
        for child in children:
            dot.edge(parent, child)

    return dot

# Generate and Save Tree Diagram
tree = generate_tree(df, routes)
output_path = "vancouver_bus_hierarchy"
tree.render(output_path, cleanup=True)

print(f"Tree diagram saved as {output_path}.svg")
