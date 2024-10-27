import streamlit as st
import os
import pandas as pd
import tensorflow as tf
import json
import joblib
from datetime import datetime
from GA.model import *
from TrafficConditions.model import *
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI

# Load API key for OpenAI
# Replace with your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-3SSseKA8p57h-UfQ9wBbmcxiyYycRPPWRhcg5ICY6ZT3BlbkFJEQIy8niz2IMuVp8QBrgSJAAxJa6hOXzHfKNqOgDLQA"

# Load CSV for LangChain Chatbot
data_file_path = 'Telemetry_KJ/truck_chat_data.csv'

# Initialize the LangChain Agent for CSV


@st.cache_resource
def initialize_agent():
    return create_csv_agent(OpenAI(temperature=0), data_file_path, allow_dangerous_code=True, verbose=True)


# Initialize Network, Traffic, and GA only once
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    current_dir = os.path.dirname(__file__)
    district_fpath = os.path.join(
        current_dir, "GA", "district_and_planning_area.geojson")

    placeholder = st.empty()
    placeholder.write("Loading components...")

    # Load and initialize classes
    gdf_districts = load_districts(district_fpath)
    network = network_graph()
    network.load_graphml(os.path.join(
        current_dir, "GA", "singapore_road_network.graphml"))

    centroids = [[float(point.x), float(point.y)]
                 for point in gdf_districts['centroid']]
    network.set_collection_nodes(centroids)

    depot_locations = {
        "TWTE": (1.3001724, 103.6236303),
        "KSTP": (1.2976918, 103.6209247),
        "TSIP": (1.2962427, 103.620556),
        "SWTE": (1.4632878, 103.794695)
    }
    network.set_depot_nodes(depot_locations)

    location_mappings = {node: loc for node, loc in zip(
        network.collection_nodes, gdf_districts["planning_area"])}
    incineration_mappings = {node: loc for node, loc in zip(
        network.depot_nodes, depot_locations.keys())}
    all_mappings = {**location_mappings, **incineration_mappings}

    # Load LSTM and scaler model
    traffic_prediction_handler = TrafficPredictionModel(
        model_path=os.path.join(
            current_dir, "TrafficConditions", "traffic_prediction_model.keras"),
        scaler_path=os.path.join(
            current_dir, "TrafficConditions", "scaler.pkl"),
        traffic_data_path=os.path.join(
            current_dir, "TrafficConditions", "current_traffic.json")
    )

    predictions = traffic_prediction_handler.predict_traffic()
    current_date = datetime.now().strftime("%Y-%m-%d")
    traffic_mapping = {
        f'{current_date} {time:02}:00:00': traffic_prediction_handler.map_traffic(traffic)
        for time, traffic in enumerate(predictions) if (time > 9 and time < 20)
    }
    timeslot_location_time_matrix = network.get_time_location_traffic_matrix(
        traffic_mapping)

    # Prepare trucks and their capacities
    TRUCKS = [f'Truck {i+1}' for i in range(30)]
    CAPACITIES = {truck: 10000 + 2000 * (i % 5)
                  for i, truck in enumerate(TRUCKS)}
    WASTE = {location: gdf_districts.loc[gdf_districts['planning_area'] == all_mappings[location],
                                         'Waste Generated Per Day (kg)'].values[0] for location in network.collection_nodes}
    START_TIME = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)

    # Initialize GA
    ga = GeneticAlgorithm(WASTE, TRUCKS, CAPACITIES, network.depot_nodes,
                          network.collection_nodes, timeslot_location_time_matrix, START_TIME)
    best_solution, log = ga.run_ga()

    # Flatten solution data for display
    flattened_data = []
    working_trucks = set()
    working_times = set()
    for truck_idx, truck_trips in enumerate(best_solution):
        working_trucks.add(truck_idx + 1)
        for trip_time, locations in truck_trips:
            locations = [all_mappings[loc] for loc in locations]
            working_times.add(trip_time)
            flattened_data.append({
                "Truck": truck_idx + 1,
                "Departure Time": trip_time,
                "Locations": " -> ".join(map(str, locations))
            })

    # Save to session state
    st.session_state.update({
        'best_solution': best_solution,
        'working_trucks': sorted(list(working_trucks)),
        'working_times': sorted(list(working_times)),
        'flattened_data': flattened_data,
        'depot_locations': depot_locations,
        'network': network,
        'initialized': True
    })
    placeholder.empty()

# Layout
st.title("Logistics Optimization and Analysis Tool")

# Chatbot section
st.subheader("Chatbot: Vehicle Maintenance Data")
agent = initialize_agent()
user_query = st.text_input("Ask a question about vehicle maintenance data:")
if user_query:
    with st.spinner("Processing your question..."):
        response = agent.run(user_query)
    st.write("### Response")
    st.write(response)

# Display GA results with options
st.subheader("Optimal Routes for Waste Collection")
sorting_option = st.selectbox(
    "Filter By:", ['Show All', 'Truck', 'Departure Time', 'Depot Points'], index=0)

# Load and display flattened data
flattened_df = pd.DataFrame(st.session_state.flattened_data)
if sorting_option == 'Show All':
    st.table(flattened_df)
    placeholder = st.empty()
    placeholder.write('Loading Figure...')
    show_all_fig, ax = st.session_state.network.draw_show_all_fig(
        best_solution=st.session_state.best_solution)
    st.pyplot(show_all_fig)
    placeholder.empty()

elif sorting_option == 'Truck':
    truck_option = st.selectbox(
        "Choose Truck:", st.session_state.working_trucks, index=0)
    df = flattened_df[flattened_df['Truck'] == truck_option]
    st.table(df)
    placeholder = st.empty()
    placeholder.write('Loading Figure...')
    truck_fig, ax = st.session_state.network.draw_truck_fig(
        truck_option, st.session_state.best_solution)
    if truck_fig:
        st.pyplot(truck_fig)
    else:
        st.write("No Routes Assigned for this Truck")
    placeholder.empty()

elif sorting_option == 'Departure Time':
    time_option = st.selectbox(
        "Choose Departure Time:", st.session_state.working_times, index=0)
    df = flattened_df[flattened_df['Departure Time'] == time_option]
    st.table(df)
    placeholder = st.empty()
    placeholder.write('Loading Figure...')
    time_fig, ax = st.session_state.network.draw_time_fig(
        time_option, st.session_state.best_solution)
    if time_fig:
        st.pyplot(time_fig)
    else:
        st.write("No Routes Starting From This Time")
    placeholder.empty()

elif sorting_option == 'Depot Points':
    depot_option = st.selectbox(
        "Choose Depot:", st.session_state.depot_locations.keys(), index=0)
    depot_dict = {"TWTE": 0, "KSTP": 1, "TSIP": 2, "SWTE": 3}
    df = flattened_df[flattened_df['Locations'].apply(
        lambda x: x.split(' ')[0]) == str(depot_option)]
    st.table(df)
    placeholder = st.empty()
    placeholder.write('Loading Figure...')
    depot_node = st.session_state.network.depot_nodes[depot_dict[depot_option]]
    depot_fig, ax = st.session_state.network.draw_depot_fig(
        depot_node, st.session_state.best_solution)
    if depot_fig:
        st.pyplot(depot_fig)
    else:
        st.write("No Routes Starting From This Depot")
    placeholder.empty()
