import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Vehicle Route Optimization",
    page_icon="🚚",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🚚 Vehicle Route Optimization Using Genetic Algorithm")

st.write(
    "A small-scale vehicle route optimization system using "
    "Genetic Algorithm and a simulated coordinate-based dataset."
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

st.header("1. Data Generation / Input")

uploaded_file = st.file_uploader(
    "Upload vehicle location CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("vehicle_locations.csv")
        st.info("Default dataset loaded: vehicle_locations.csv")
    except:
        st.error("vehicle_locations.csv was not found.")
        st.stop()


# Check required columns
required_columns = ["Location", "X", "Y"]

if not all(column in df.columns for column in required_columns):
    st.error(
        "CSV must contain these columns: Location, X, Y"
    )
    st.stop()


st.subheader("Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# ---------------------------------------------------------
# CONVERT DATA INTO LOCATIONS
# ---------------------------------------------------------

locations = {}

for _, row in df.iterrows():

    locations[row["Location"]] = (
        float(row["X"]),
        float(row["Y"])
    )


location_names = list(locations.keys())

if "Depot" not in location_names:
    st.error("The dataset must contain a location named 'Depot'.")
    st.stop()


delivery_locations = [
    location
    for location in location_names
    if location != "Depot"
]


st.write(
    "Number of delivery locations:",
    len(delivery_locations)
)


# ---------------------------------------------------------
# DISTANCE MATRIX
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# GENETIC ALGORITHM PARAMETERS
# ---------------------------------------------------------

st.header("2. Genetic Algorithm Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    population_size = st.number_input(
        "Population Size",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

with col2:
    number_of_generations = st.number_input(
        "Generations",
        min_value=50,
        max_value=1000,
        value=200,
        step=50
    )

with col3:
    crossover_rate = st.slider(
        "Crossover Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.05
    )

with col4:
    mutation_rate = st.slider(
        "Mutation Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.05
    )


elite_size = 2


# ---------------------------------------------------------
# ROUTE DISTANCE
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# FITNESS FUNCTION
# ---------------------------------------------------------

def calculate_fitness(route):

    distance = calculate_route_distance(route)

    return 1 / distance


# ---------------------------------------------------------
# CREATE RANDOM ROUTE
# ---------------------------------------------------------

def create_random_route():

    route = delivery_locations.copy()

    random.shuffle(route)

    return route


# ---------------------------------------------------------
# INITIAL POPULATION
# ---------------------------------------------------------

def create_initial_population():

    population = []

    for _ in range(population_size):

        population.append(
            create_random_route()
        )

    return population


# ---------------------------------------------------------
# TOURNAMENT SELECTION
# ---------------------------------------------------------

def tournament_selection(
    population,
    tournament_size=3
):

    tournament = random.sample(
        population,
        min(tournament_size, len(population))
    )

    winner = min(
        tournament,
        key=calculate_route_distance
    )

    return winner.copy()


# ---------------------------------------------------------
# ORDER CROSSOVER
# ---------------------------------------------------------

def order_crossover(parent1, parent2):

    length = len(parent1)

    if length < 2:
        return parent1.copy()

    start, end = sorted(
        random.sample(
            range(length),
            2
        )
    )

    child = [None] * length

    child[
        start:end + 1
    ] = parent1[
        start:end + 1
    ]

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


# ---------------------------------------------------------
# SWAP MUTATION
# ---------------------------------------------------------

def swap_mutation(route):

    mutated_route = route.copy()

    if len(mutated_route) < 2:
        return mutated_route

    index1, index2 = random.sample(
        range(len(mutated_route)),
        2
    )

    (
        mutated_route[index1],
        mutated_route[index2]
    ) = (
        mutated_route[index2],
        mutated_route[index1]
    )

    return mutated_route


# ---------------------------------------------------------
# GENETIC ALGORITHM
# ---------------------------------------------------------

def genetic_algorithm():

    random.seed(42)
    np.random.seed(42)

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
        while len(new
