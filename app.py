import streamlit as st
from GA.model import *
from TrafficConditions.model import *
import os
import tensorflow as tf
import json
import joblib

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)
    district_fpath = os.path.join(
        current_dir, "GA", "district_and_planning_area.geojson")

    placeholder = st.empty()
    placeholder.write("Loading...")
    gdf_districts = load_districts(district_fpath)

    # Init classes
    network = network_graph()
    network.load_graphml(os.path.join(
        current_dir, "GA", "singapore_road_network.graphml"))

    # Set collection nodes
    centroids = [[float(point.x), float(point.y)]
                 for point in gdf_districts['centroid']]
    network.set_collection_nodes(centroids)

    # Set the depot nodes
    depot_locations = {"TWTE": (1.3001724, 103.6236303),
                       "KSTP": (1.2976918, 103.6209247),
                       "TSIP": (1.2962427, 103.620556),
                       "SWTE": (1.4632878, 103.794695)}

    # Find the nearest node to the depot
    network.set_depot_nodes(depot_locations)

    # Load LSTM and scaler model
    traffic_prediction_handler = TrafficPredictionModel(model_path=os.path.join(current_dir, "TrafficConditions", "traffic_prediction_model.keras"),
                                                        scaler_path=os.path.join(
                                                            current_dir, "TrafficConditions", "scaler.pkl"),
                                                        traffic_data_path=os.path.join(current_dir, "TrafficConditions", "current_traffic.json"))

    # Predict traffic
    predictions = traffic_prediction_handler.predict_traffic()

    # Map traffic for today starting from 00:00 am
    current_date = datetime.now().strftime("%Y-%m-%d")
    traffic_mapping = {f'{current_date} {time:02}:00:00': traffic_prediction_handler.map_traffic(
        # 10:00 am to 7:00 pm
        traffic) for time, traffic in enumerate(predictions) if (time > 9 and time < 20)}
    timeslot_location_time_matrix = network.get_time_location_traffic_matrix(
        traffic_mapping)

    # Predict trucks
    # Sample data to be replaced with actual data
    TRUCKS = ['Truck 1', 'Truck 2', 'Truck 3', 'Truck 4', 'Truck 5', 'Truck 6', 'Truck 7', 'Truck 8', 'Truck 9', 'Truck 10', "Truck 11", "Truck 12", "Truck 13", "Truck 14", "Truck 15",
              "Truck 16", "Truck 17", "Truck 18", "Truck 19", "Truck 20", "Truck 21", "Truck 22", "Truck 23", "Truck 24", "Truck 25", "Truck 26", "Truck 27", "Truck 28", "Truck 29", "Truck 30"]
    CAPACITIES = {'Truck 1': 1000, 'Truck 2': 1200, 'Truck 3': 900, 'Truck 4': 800, 'Truck 5': 1100, 'Truck 6': 1174, 'Truck 7': 1320, 'Truck 8': 1193, 'Truck 9': 768, 'Truck 10': 1340, 'Truck 11': 1000, 'Truck 12': 1200, 'Truck 13': 900, 'Truck 14': 800, 'Truck 15': 1100,
                  'Truck 16': 1174, 'Truck 17': 1320, 'Truck 18': 1193, 'Truck 19': 768, 'Truck 20': 1340, 'Truck 21': 1000, 'Truck 22': 1200, 'Truck 23': 900, 'Truck 24': 800, 'Truck 25': 1100, 'Truck 26': 1174, 'Truck 27': 1320, 'Truck 28': 1193, 'Truck 29': 768, 'Truck 30': 1340}
    WASTE = {location: random.randint(500, 1500)
             for location in network.collection_nodes}

    # Set start time as current date 10:00 am
    START_TIME = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)

    # Init GA
    ga = GeneticAlgorithm(WASTE, TRUCKS, CAPACITIES, network.depot_nodes,
                          network.collection_nodes, timeslot_location_time_matrix, START_TIME)

    # Run GA
    print("Start running GA")
    best_solution, log = ga.run_ga()

    # Display the best solution
    # Prepare a list to store the flattened data
    flattened_data = []

    working_trucks = set()
    working_times = set()
    # Loop through the best_individual and flatten it
    for truck_idx, truck_trips in enumerate(best_solution):
        working_trucks.add(truck_idx + 1)
        for trip_time, locations in truck_trips:
            working_times.add(trip_time)
            flattened_data.append({
                "Truck": truck_idx + 1,  # Truck number (1-based index)
                "Departure Time": trip_time,
                # Joining locations with arrows
                "Locations": " -> ".join(map(str, locations))
            })
    working_trucks = sorted(list(working_trucks))
    print("Working Trucks: ", working_trucks)
    working_times = sorted(list(working_times))
    print("Working Times:", working_times)

    placeholder.empty()
    st.session_state.best_solution = best_solution
    st.session_state.working_trucks = working_trucks
    st.session_state.working_times = working_times
    st.session_state.flattened_data = flattened_data
    st.session_state.depot_locations = depot_locations
    st.session_state.network = network
    st.session_state.initialized = True

st.title('Genetic Algorithm')

sorting_option = st.selectbox(
    "Sorting By: ", ['Show All', 'Truck', 'Departure Time', 'Depot Points'], index=0)

flattened_data = st.session_state.flattened_data
flattened_data = pd.DataFrame(flattened_data)

if sorting_option == 'Show All':
    # Display the table in Streamlit
    st.table(flattened_data)

    placeholder = st.empty()
    placeholder.write('Loading Figure...')
    show_all_fig, ax = st.session_state.network.draw_show_all_fig(
        best_solution=st.session_state.best_solution)
    st.pyplot(show_all_fig)
    placeholder.empty()

elif sorting_option == 'Truck':
    truck_option = st.selectbox(
        "Truck: ", st.session_state.working_trucks, index=0)

    df = flattened_data[flattened_data['Truck'] == truck_option]
    st.table(df)

    if truck_option:
        placeholder = st.empty()
        placeholder.write('Loading Figure...')
        truck_fig, ax = st.session_state.network.draw_truck_fig(
            truck_option, st.session_state.best_solution)
        if truck_fig:
            st.pyplot(truck_fig)
        else:
            st.write("No Routes Starting From This Depot")
        placeholder.empty()

elif sorting_option == 'Departure Time':
    time_option = st.selectbox(
        "Departure Time: ", st.session_state.working_times, index=0)

    df = flattened_data[flattened_data['Departure Time'] == time_option]
    st.table(df)

    if time_option:
        placeholder = st.empty()
        placeholder.write('Loading Figure...')
        time_fig, ax = st.session_state.network.draw_time_fig(
            time_option, st.session_state.best_solution)
        if time_fig:
            st.pyplot(time_fig)
        else:
            st.write("No Routes Starting From This Depot")
        placeholder.empty()

elif sorting_option == 'Depot Points':
    depot_option = st.selectbox(
        "Depot Points: ", st.session_state.depot_locations.keys(), index=0)

    depot_dict = {"TWTE": 0,
                  "KSTP": 1,
                  "TSIP": 2,
                  "SWTE": 3}

    df = flattened_data[flattened_data['Locations'].apply(
        lambda x: x.split(' ')[0]) == str(st.session_state.network.depot_nodes[depot_dict[depot_option]])]
    st.table(df)

    if depot_option:
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
