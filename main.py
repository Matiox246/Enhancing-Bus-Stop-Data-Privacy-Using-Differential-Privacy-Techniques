import pandas as pd
import numpy as np
from graphviz import Digraph


# Sample Data for Vancouver Bus Network
data = {
    "Station": ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"],
    "Route": [
        "L1 > L2 > L3", "L1 > L2 > L4", 
        "L2 > L5 > L6", "L2 > L7 > L8",
        "L3 > L9", "L4 > L9","L3 > L5 > L6 > L7",
        "L2 > L6 > L8 > L9", "L1 > L5 > L6 > L7",
    ],
    "Total Passengers": [30, 25, 20, 18, 15, 12, 8, 10, 5],
    "Alighting": [5, 4, 6, 3, 2, 4, 3, 2, 5],
    "To Next": [15, 20, 12, 10, 8, 7, 5, 3, 0],
    "Transfer": [10, 1, 2, 5, 3, 1, 0, 1, 0],
    "Epsilon": [1.0, 0.8, 0.6, 0.7, 0.9, 0.8, 0.5, 0.7, 0.6],
}

df = pd.DataFrame(data)

# Add original records for comparison
df["Original Total"] = df["Total Passengers"].copy()
df["Original Alighting"] = df["Alighting"].copy()
df["Original To Next"] = df["To Next"].copy()

# Differential Privacy Function
def add_laplace_noise(counts, sensitivity, epsilon):
    """
    Adds Laplace noise to counts with adjustable epsilon.
    """
    noise = np.random.laplace(0, sensitivity / epsilon, size=len(counts))
    return np.maximum(0, np.round(counts + noise))  # Ensure non-negative counts

# Add noise to the dataset
df["Total Passengers"] = add_laplace_noise(df["Total Passengers"], sensitivity=1, epsilon=df["Epsilon"])
df["Alighting"] = add_laplace_noise(df["Alighting"], sensitivity=1, epsilon=df["Epsilon"])
df["To Next"] = add_laplace_noise(df["To Next"], sensitivity=1, epsilon=df["Epsilon"])

# Prune stations with no passengers remaining
df = df[df["Total Passengers"] > 0]

# Generate Tree Visualization
def generate_tree(data):
    """
    Generates a realistic branching tree diagram for Vancouver's bus network.
    """
    dot = Digraph(format='JPEG')
    dot.attr(rankdir="TB", size="12,8")  # Top-to-Bottom
    dot.attr("node", shape="ellipse", style="filled", fillcolor="lightblue")

    # Add nodes with both original and noisy data
    for _, row in data.iterrows():
        station = row["Station"]
        orig_label = (
            f"Original:\n"
            f"Total: {row['Original Total']}, "
        )
        noisy_label = (
            f"Noisy:\n"
            f"Total: {row['Total Passengers']}, "
        )
        dot.node(station, f"{station}\n\n{orig_label}\n\n{noisy_label}")

    # Add edges for each route
    for route in data["Route"].unique():
        stops = route.split(" > ")
        for i in range(len(stops) - 1):
            dot.edge(stops[i], stops[i + 1])

    return dot

# Generate the tree diagram
tree = generate_tree(df)
output_path = "vancouver_bus_tree"
tree.render(output_path, cleanup=True)

print(f"Tree diagram saved as {output_path}.JPEG")


