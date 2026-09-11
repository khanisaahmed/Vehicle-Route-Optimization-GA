import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# --------------------------------------------------
# Page Settings
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Route Optimization",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Vehicle Route Optimization Using Genetic Algorithm")

st.write(
    "A small-scale vehicle route optimization system using "
    "Genetic Algorithm and a coordinate-based dataset."
)

# --------------------------------------------------
# Module 1: Data Generation / Input
# --------------------------------------------------

st.header("1. Data Generation / Input")

st.write("Upload vehicle location CSV file")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

# --------------------------------------------------
# Stop application until file is uploaded
# --------------------------------------------------

if uploaded_file is None:

    st.info(
        "Please upload a CSV file containing Location, X and Y coordinates."
    )

    st.stop()

# --------------------------------------------------
# Load Uploaded Dataset
# --------------------------------------------------

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("Unable to read the CSV file.")
    st.stop()

# Check required columns

required_columns = ["Location", "X", "Y"]

if not all(column in df.columns for column in required_columns):

    st.error(
        "Invalid CSV file. The file must contain these columns: "
        "Location, X, Y"
    )

    st.stop()

# Check for empty dataset

if len(df) < 3:

    st.error(
        "Please upload a CSV file containing at least "
        "one Depot and two delivery locations."
    )

    st.stop()

# Check Depot

if "Depot" not in df["Location"].values:

    st.error(
        "The CSV file must contain one location named 'Depot'."
    )

    st.stop()

# Convert coordinates to numbers

try:

    df["X"] = pd.to_numeric(df["X"])
    df["Y"] = pd.to_numeric(df["Y"])

except Exception:

    st.error(
        "X and Y columns must contain numeric coordinates."
    )

    st.stop()

# --------------------------------------------------
# Display Dataset
# --------------------------------------------------

st.subheader("Dataset")

st.dataframe(
    df,
    use_container_width=True
)

# --------------------------------------------------
# Location Dictionary
# --------------------------------------------------

locations = {}

for _, row in df.iterrows():

    locations[row["Location"]] = (
        float(row["X"]),
        float(row["Y"])
    )

location_names = list(locations.keys())

delivery_locations = [
    location
    for location in location_names
    if location != "Depot"
]

# --------------------------------------------------
# Module 2: Route Representation and Population
# --------------------------------------------------

st.header(
    "2. Route Representation and Population Initialization"
)

population_size = 50
number_of_generations = 200
crossover_rate = 0.90
mutation_rate = 0.10
elite_size = 2

st.write(
    "Each chromosome represents one possible delivery route."
)

st.write(
    f"Number of delivery locations: {len(delivery_locations)}"
)

# --------------------------------------------------
# Distance Matrix
# --------------------------------------------------

num_locations = len(location_names)

distance_matrix = np.zeros(
    (num_locations, num_locations)
)

for i in range(num_locations):

    for j in range(num_locations):

        x1, y1 = locations[location_names[i]]
        x2, y2 = locations[location_names[j]]

        distance_matrix[i][j] = np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

# --------------------------------------------------
# Create Random Route
# --------------------------------------------------

def create_random_route():

    route = delivery_locations.copy()

    random.shuffle(route)

    return route


# --------------------------------------------------
# Create Initial Population
# --------------------------------------------------

def create_initial_population():

    population = []

    for _ in range(population_size):

        population.append(
            create_random_route()
        )

    return population


# --------------------------------------------------
# Module 3: Fitness Evaluation
# --------------------------------------------------

st.header("3. Fitness Evaluation")


def calculate_route_distance(route):

    complete_route = (
        ["Depot"] +
        route +
        ["Depot"]
    )

    total_distance = 0

    for i in range(
        len(complete_route) - 1
    ):

        current_location = complete_route[i]

        next_location = complete_route[i + 1]

        current_index = location_names.index(
            current_location
        )

        next_index = location_names.index(
            next_location
        )

        total_distance += distance_matrix[
            current_index
        ][
            next_index
        ]

    return total_distance


def calculate_fitness(route):

    distance = calculate_route_distance(route)

    return 1 / distance


# --------------------------------------------------
# Module 4: Genetic Algorithm Operations
# --------------------------------------------------

st.header(
    "4. Genetic Algorithm Operations"
)

# --------------------------------------------------
# Selection
# --------------------------------------------------

def tournament_selection(
    population,
    tournament_size=3
):

    tournament = random.sample(
        population,
        tournament_size
    )

    winner = min(
        tournament,
        key=calculate_route_distance
    )

    return winner.copy()


# --------------------------------------------------
# Order Crossover
# --------------------------------------------------

def order_crossover(parent1, parent2):

    length = len(parent1)

    start, end = sorted(
        random.sample(
            range(length),
            2
        )
    )

    child = [None] * length

    child[start:end + 1] = (
        parent1[start:end + 1]
    )

    remaining_genes = [
        gene
        for gene in parent2
        if gene not in child
    ]

    remaining_index = 0

    for i in range(length):

        if child[i] is None:

            child[i] = remaining_genes[
                remaining_index
            ]

            remaining_index += 1

    return child


# --------------------------------------------------
# Swap Mutation
# --------------------------------------------------

def swap_mutation(route):

    mutated_route = route.copy()

    index1, index2 = random.sample(
        range(len(mutated_route)),
        2
    )

    mutated_route[index1], mutated_route[index2] = (
        mutated_route[index2],
        mutated_route[index1]
    )

    return mutated_route


# --------------------------------------------------
# Genetic Algorithm
# --------------------------------------------------

def genetic_algorithm():

    current_population = (
        create_initial_population()
    )

    best_route = None

    best_distance = float("inf")

    best_distance_history = []

    average_distance_history = []

    for generation in range(
        number_of_generations
    ):

        current_population = sorted(
            current_population,
            key=calculate_route_distance
        )

        current_best_route = (
            current_population[0]
        )

        current_best_distance = (
            calculate_route_distance(
                current_best_route
            )
        )

        if current_best_distance < best_distance:

            best_distance = (
                current_best_distance
            )

            best_route = (
                current_best_route.copy()
            )

        distances = [
            calculate_route_distance(route)
            for route in current_population
        ]

        best_distance_history.append(
            min(distances)
        )

        average_distance_history.append(
            np.mean(distances)
        )

        # Elitism

        new_population = [
            route.copy()
            for route in current_population[
                :elite_size
            ]
        ]

        # Generate new population

        while len(new_population) < population_size:

            parent1 = tournament_selection(
                current_population
            )

            parent2 = tournament_selection(
                current_population
            )

            # Crossover

            if random.random() < crossover_rate:

                child1 = order_crossover(
                    parent1,
                    parent2
                )

                child2 = order_crossover(
                    parent2,
                    parent1
                )

            else:

                child1 = parent1.copy()

                child2 = parent2.copy()

            # Mutation

            if random.random() < mutation_rate:

                child1 = swap_mutation(
                    child1
                )

            if random.random() < mutation_rate:

                child2 = swap_mutation(
                    child2
                )

            new_population.append(
                child1
            )

            if len(new_population) < population_size:

                new_population.append(
                    child2
                )

        current_population = new_population

    return (
        best_route,
        best_distance,
        best_distance_history,
        average_distance_history
    )


# --------------------------------------------------
# Run Button
# --------------------------------------------------

st.divider()

run_button = st.button(
    "🚀 Run Genetic Algorithm",
    type="primary"
)

# --------------------------------------------------
# Run Optimization
# --------------------------------------------------

if run_button:

    random.seed(42)

    np.random.seed(42)

    # Initial Population

    initial_population = (
        create_initial_population()
    )

    initial_distances = [
        calculate_route_distance(route)
        for route in initial_population
    ]

    best_initial_index = np.argmin(
        initial_distances
    )

    best_initial_distance = (
        initial_distances[
            best_initial_index
        ]
    )

    best_initial_route = (
        initial_population[
            best_initial_index
        ]
    )

    # Run Genetic Algorithm

    (
        best_route,
        optimized_distance,
        best_distance_history,
        average_distance_history
    ) = genetic_algorithm()

    # --------------------------------------------------
    # Module 5: Result Analysis
    # --------------------------------------------------

    st.header("5. Result Analysis")

    distance_reduction = (
        best_initial_distance -
        optimized_distance
    )

    improvement_percentage = (
        distance_reduction /
        best_initial_distance
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Initial Distance",
        f"{best_initial_distance:.2f}"
    )

    col2.metric(
        "Optimized Distance",
        f"{optimized_distance:.2f}"
    )

    col3.metric(
        "Improvement",
        f"{improvement_percentage:.2f}%"
    )

    st.subheader("Initial Best Route")

    st.code(
        "Depot → " +
        " → ".join(
            best_initial_route
        ) +
        " → Depot"
    )

    st.subheader("Optimized Route")

    st.code(
        "Depot → " +
        " → ".join(
            best_route
        ) +
        " → Depot"
    )

    st.write(
        "**Distance Reduced:**",
        round(distance_reduction, 2)
    )

    # Results Table

    results = pd.DataFrame({

        "Metric": [

            "Initial Route Distance",

            "Optimized Route Distance",

            "Distance Reduction",

            "Improvement Percentage",

            "Delivery Locations",

            "Population Size",

            "Generations",

            "Crossover Rate",

            "Mutation Rate",

            "Elite Size"

        ],

        "Value": [

            round(
                best_initial_distance,
                2
            ),

            round(
                optimized_distance,
                2
            ),

            round(
                distance_reduction,
                2
            ),

            round(
                improvement_percentage,
                2
            ),

            len(
                delivery_locations
            ),

            population_size,

            number_of_generations,

            crossover_rate,

            mutation_rate,

            elite_size

        ]
    })

    st.subheader(
        "Final Project Results"
    )

    st.dataframe(
        results,
        use_container_width=True
    )

    # --------------------------------------------------
    # Module 6: Visualization
    # --------------------------------------------------

    st.header("6. Visualization")

    # Optimized Route

    st.subheader(
        "Optimized Vehicle Route"
    )

    route_for_plot = (
        ["Depot"] +
        best_route +
        ["Depot"]
    )

    x_values = [
        locations[location][0]
        for location in route_for_plot
    ]

    y_values = [
        locations[location][1]
        for location in route_for_plot
    ]

    fig1, ax1 = plt.subplots(
        figsize=(10, 6)
    )

    ax1.plot(
        x_values,
        y_values,
        marker="o"
    )

    for name, (x, y) in locations.items():

        ax1.annotate(
            name,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax1.set_xlabel(
        "X Coordinate"
    )

    ax1.set_ylabel(
        "Y Coordinate"
    )

    ax1.set_title(
        "Optimized Vehicle Route Using Genetic Algorithm"
    )

    ax1.grid(True)

    st.pyplot(fig1)

    # GA Progress

    st.subheader(
        "Genetic Algorithm Optimization Progress"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 6)
    )

    ax2.plot(
        best_distance_history,
        label="Best Distance"
    )

    ax2.plot(
        average_distance_history,
        label="Average Distance"
    )

    ax2.set_xlabel(
        "Generation"
    )

    ax2.set_ylabel(
        "Route Distance"
    )

    ax2.set_title(
        "Genetic Algorithm Optimization Progress"
    )

    ax2.legend()

    ax2.grid(True)

    st.pyplot(fig2)

    # Initial vs Optimized

    st.subheader(
        "Initial vs Optimized Route Distance"
    )

    comparison = pd.DataFrame({

        "Route": [
            "Initial",
            "Optimized"
        ],

        "Distance": [
            best_initial_distance,
            optimized_distance
        ]
    })

    fig3, ax3 = plt.subplots(
        figsize=(8, 5)
    )

    ax3.bar(
        comparison["Route"],
        comparison["Distance"]
    )

    ax3.set_xlabel(
        "Route"
    )

    ax3.set_ylabel(
        "Distance"
    )

    ax3.set_title(
        "Initial vs Optimized Route Distance"
    )

    ax3.grid(
        axis="y"
    )

    st.pyplot(fig3)

    # Final Message

    st.success(
        "Genetic Algorithm successfully optimized the vehicle route!"
    )
