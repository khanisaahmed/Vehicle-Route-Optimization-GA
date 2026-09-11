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


# ---------------------------------------------------------
# RUN OPTIMIZATION
# ---------------------------------------------------------

st.header("3. Route Optimization")

run_button = st.button(
    "🚀 Run Genetic Algorithm",
    type="primary"
)


if run_button:

    with st.spinner(
        "Running Genetic Algorithm..."
    ):

        # Generate initial population
        random.seed(42)
        np.random.seed(42)

        initial_population = (
            create_initial_population()
        )


        initial_population = sorted(
            initial_population,
            key=calculate_route_distance
        )


        initial_best_route = (
            initial_population[0]
        )


        initial_distance = (
            calculate_route_distance(
                initial_best_route
            )
        )


        # Run Genetic Algorithm
        (
            best_route,
            optimized_distance,
            best_distance_history,
            average_distance_history
        ) = genetic_algorithm()


        distance_reduction = (
            initial_distance -
            optimized_distance
        )


        improvement_percentage = (
            distance_reduction /
            initial_distance
        ) * 100


        # Save results
        st.session_state["best_route"] = best_route

        st.session_state["optimized_distance"] = optimized_distance

        st.session_state["initial_distance"] = initial_distance

        st.session_state["distance_reduction"] = distance_reduction

        st.session_state["improvement_percentage"] = improvement_percentage

        st.session_state["best_distance_history"] = best_distance_history

        st.session_state["average_distance_history"] = average_distance_history

        st.success(
            "Genetic Algorithm completed successfully!"
        )


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if "best_route" in st.session_state:

    st.header("4. Result Analysis")


    initial_distance = (
        st.session_state[
            "initial_distance"
        ]
    )

    optimized_distance = (
        st.session_state[
            "optimized_distance"
        ]
    )

    distance_reduction = (
        st.session_state[
            "distance_reduction"
        ]
    )

    improvement_percentage = (
        st.session_state[
            "improvement_percentage"
        ]
    )

    best_route = (
        st.session_state[
            "best_route"
        ]
    )


    # Metrics
    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Initial Distance",
            f"{initial_distance:.2f}"
        )


    with col2:

        st.metric(
            "Optimized Distance",
            f"{optimized_distance:.2f}"
        )


    with col3:

        st.metric(
            "Distance Reduction",
            f"{distance_reduction:.2f}"
        )


    with col4:

        st.metric(
            "Improvement",
            f"{improvement_percentage:.2f}%"
        )


    # Optimized route
    st.subheader(
        "Optimized Vehicle Route"
    )


    complete_best_route = (
        ["Depot"] +
        best_route +
        ["Depot"]
    )


    st.success(
        " → ".join(
            complete_best_route
        )
    )


    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    st.header("5. Visualization")


    # Location visualization
    st.subheader(
        "Simulated Vehicle Delivery Locations"
    )


    fig1, ax1 = plt.subplots(
        figsize=(10, 7)
    )


    for name, (x, y) in locations.items():

        if name == "Depot":

            ax1.scatter(
                x,
                y,
                s=250,
                marker="s",
                label="Depot"
            )

        else:

            ax1.scatter(
                x,
                y,
                s=100
            )


        ax1.text(
            x + 1.5,
            y + 1.5,
            name,
            fontsize=10
        )


    ax1.set_title(
        "Simulated Vehicle Delivery Locations"
    )

    ax1.set_xlabel(
        "X Coordinate"
    )

    ax1.set_ylabel(
        "Y Coordinate"
    )

    ax1.grid(True)

    ax1.legend()

    st.pyplot(fig1)


    # -----------------------------------------------------
    # OPTIMIZED ROUTE GRAPH
    # -----------------------------------------------------

    st.subheader(
        "Optimized Vehicle Route"
    )


    x_coordinates = [
        locations[location][0]
        for location in complete_best_route
    ]


    y_coordinates = [
        locations[location][1]
        for location in complete_best_route
    ]


    fig2, ax2 = plt.subplots(
        figsize=(11, 8)
    )


    ax2.plot(
        x_coordinates,
        y_coordinates,
        marker="o",
        linewidth=2
    )


    for name, (x, y) in locations.items():

        if name == "Depot":

            ax2.scatter(
                x,
                y,
                s=250,
                marker="s",
                label="Depot"
            )

        else:

            ax2.scatter(
                x,
                y,
                s=100
            )


        ax2.text(
            x + 1.5,
            y + 1.5,
            name,
            fontsize=10
        )


    ax2.set_title(
        "Optimized Vehicle Route Using Genetic Algorithm"
    )

    ax2.set_xlabel(
        "X Coordinate"
    )

    ax2.set_ylabel(
        "Y Coordinate"
    )

    ax2.grid(True)

    ax2.legend()

    st.pyplot(fig2)


    # -----------------------------------------------------
    # GA PROGRESS GRAPH
    # -----------------------------------------------------

    st.subheader(
        "Genetic Algorithm Optimization Progress"
    )


    generation_numbers = range(
        1,
        number_of_generations + 1
    )


    fig3, ax3 = plt.subplots(
        figsize=(10, 6)
    )


    ax3.plot(
        generation_numbers,
        st.session_state[
            "best_distance_history"
        ],
        label="Best Route Distance",
        linewidth=2
    )


    ax3.plot(
        generation_numbers,
        st.session_state[
            "average_distance_history"
        ],
        label="Average Population Distance",
        linewidth=2
    )


    ax3.set_title(
        "Genetic Algorithm Optimization Progress"
    )

    ax3.set_xlabel(
        "Generation"
    )

    ax3.set_ylabel(
        "Route Distance"
    )

    ax3.grid(True)

    ax3.legend()

    st.pyplot(fig3)


    # -----------------------------------------------------
    # INITIAL VS OPTIMIZED GRAPH
    # -----------------------------------------------------

    st.subheader(
        "Initial vs Optimized Route Distance"
    )


    labels = [
        "Initial Route",
        "Optimized Route"
    ]


    distances = [
        initial_distance,
        optimized_distance
    ]


    fig4, ax4 = plt.subplots(
        figsize=(8, 6)
    )


    bars = ax4.bar(
        labels,
        distances
    )


    ax4.set_title(
        "Initial vs Optimized Route Distance"
    )

    ax4.set_ylabel(
        "Total Distance"
    )


    for bar, value in zip(
        bars,
        distances
    ):

        ax4.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 5,
            f"{value:.2f}",
            ha="center"
        )


    st.pyplot(fig4)


    # -----------------------------------------------------
    # PROJECT SUMMARY
    # -----------------------------------------------------

    st.header("6. Project Summary")


    summary = pd.DataFrame({

        "Metric": [

            "Number of Delivery Locations",

            "Population Size",

            "Number of Generations",

            "Crossover Rate",

            "Mutation Rate",

            "Selection Method",

            "Crossover Method",

            "Mutation Method",

            "Elitism",

            "Initial Distance",

            "Optimized Distance",

            "Distance Reduction",

            "Improvement Percentage"

        ],

        "Value": [

            len(delivery_locations),

            population_size,

            number_of_generations,

            crossover_rate,

            mutation_rate,

            "Tournament Selection",

            "Order Crossover (OX)",

            "Swap Mutation",

            "Yes",

            round(initial_distance, 2),

            round(optimized_distance, 2),

            round(distance_reduction, 2),

            f"{improvement_percentage:.2f}%"

        ]

    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    st.success(
        "Vehicle route optimization completed successfully."
    )
