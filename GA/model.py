import geopandas as gpd
import geojson
from shapely.geometry import shape
import matplotlib.pyplot as plt
import fiona
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
from builtins import sum
from datetime import timedelta, datetime
from copy import deepcopy
import os


def load_districts(file_path: str) -> gpd.GeoDataFrame:
    """
    Load the districts from the geojson file
    """
    # Collect geometries and properties in lists
    geometries = []
    properties_list = []

    with fiona.open(file_path, 'r') as source:
        for feature in source:
            geometries.append(shape(feature['geometry']))
            properties_list.append(feature['properties'])

    # Create a GeoDataFrame
    gdf_districts = gpd.GeoDataFrame(properties_list, geometry=geometries)
    gdf_districts['centroid'] = gdf_districts['geometry'].centroid

    
    path = os.path.join("GA","Singapore population 2020 V2.xlsx")
    waste_df = pd.read_excel(path)
    waste_df = waste_df[waste_df['Subzone'] == 'Total']
    waste_df = waste_df.drop(index=0)
    waste_df = waste_df.iloc[:, [0, -1]]

    gdf_districts = gdf_districts.merge(
        waste_df, how='left', left_on='planning_area', right_on='Planning Area of Residence')
    gdf_districts = gdf_districts.drop(['Planning Area of Residence'], axis=1)

    gdf_districts = gdf_districts[
        gdf_districts['Waste Generated Per Day (kg)'] != 0.0]
    gdf_districts = gdf_districts.reset_index(drop=True)
    return gdf_districts


class network_graph:
    def __init__(self):
        self.G = None
        self.collection_nodes = None
        self.depot_nodes = None

    def get_graph(self) -> nx.Graph:
        """
        Get the graph
        """
        return self.G

    def load_graphml(self, file_path: str):
        """
        Load the graph from the graphml file
        """
        self.G = ox.load_graphml(file_path)

    def get_network_nodes(self) -> list:
        """
        Get the nodes from the graph
        """
        return list(self.G.nodes)

    def get_closest_nodes(self, points) -> list:
        """
        Get the closest nodes to the given points
        """
        if isinstance(points, dict):
            # Process points as a dictionary (Usecase: Incineration plants)
            nearest_node_ids = [ox.distance.nearest_nodes(
                self.G, X=location[1], Y=location[0]) for location in points.values()]
        else:
            # Process points as an np.array (General usecase)
            nearest_node_ids = [ox.distance.nearest_nodes(
                self.G, X=centroid[0], Y=centroid[1]) for centroid in points]

        return nearest_node_ids

    def set_collection_nodes(self, collection_points: np.array):
        """
        Set the collection nodes
        """
        self.collection_nodes = self.get_closest_nodes(collection_points)

    def set_depot_nodes(self, depot_points: dict):
        """
        Set the depot nodes
        """
        self.depot_nodes = self.get_closest_nodes(depot_points)

    def compute_distance_time_matrix(self):
        """
        Compute a distance/time matrix between all given locations and return as a NumPy matrix.
        """
        print("Start computing distance matrix")
        locations = self.collection_nodes + self.depot_nodes
        n = len(locations)
        matrix = np.zeros((n, n))  # Initialize a NumPy matrix with zeros

        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if loc1 == loc2:
                    matrix[i][j] = 0  # Distance from a location to itself is 0
                else:
                    # Compute the shortest path distance/time using networkx
                    shortest_path = nx.shortest_path(
                        self.G, loc1, loc2, weight='weight')
                    total_time = 0
                    for u, v in zip(shortest_path[:-1], shortest_path[1:]):
                        edge_data = self.G.get_edge_data(u, v)
                        # Edge length in meters
                        edge_length = edge_data[0]['length']

                        # If maxspeed is in a list, pick lowest value
                        # Default speed if not available (in km/h)
                        speed_kph = edge_data[0].get('maxspeed', 40)
                        speed_kph = min([int(speed) for speed in speed_kph]) if isinstance(
                            speed_kph, list) else int(speed_kph)
                        speed_mps = speed_kph * 1000 / 3600  # Convert speed to meters per second
                        if edge_length == 0:
                            travel_time = 0
                        else:
                            travel_time = edge_length / speed_mps  # Time in seconds for this edge
                        total_time += travel_time
                    matrix[i][j] = total_time / 60

        print("Stop computing distance matrix")
        return matrix

    def get_time_location_traffic_matrix(self, single_day_traffic: dict):
        """
        Get the time matrix with traffic
        """
        time_location_traffic_matrix = {}
        time_matrix = self.compute_distance_time_matrix()

        for date_time, traffic_modifier in single_day_traffic.items():
            time_location_traffic_matrix[date_time] = time_matrix * \
                traffic_modifier

        return time_location_traffic_matrix

    colors = ['#FF0000', '#00FF00', '#0000FF', '#F60009', '#09F600', '#0009F6',
              '#FFFF00', '#00FFFF', '#FF00FF', '#F6F609', '#09F6F6', '#F609F6',
              '#FFA500', '#800080', '#FFC0CB', '#F6A509', '#890089', '#F6C0D2',
              '#ADD8E6', '#8B0000', '#006400', '#A0D9E6', '#910009', '#096409',
              '#00008B', '#90EE90', '#FFD700', '#91F691', '#000091', '#F6D709',
              '#000080', '#808000', '#F5F5DC', '#C0C9C0', '#000089', '#898009',
              '#F6F5E2', '#87CEEB', '#C71585', '#228B22', '#B8860B', '#FF6347',
              '#FF0000', '#00FF00', '#0000FF', '#F60009', '#09F600', '#0009F6',
              '#FFFF00', '#00FFFF', '#FF00FF', '#F6F609', '#09F6F6', '#F609F6']*4

    def draw_show_all_fig(self, best_solution):
        all_routes = []
        for truck_idx, truck_trips in enumerate(best_solution):
            for trip_time, locations in truck_trips:
                full_route = []
                for i in range(0, len(locations) - 1):
                    route = nx.shortest_path(
                        self.G, locations[i], locations[i+1])
                    full_route.extend(route[1:])
                all_routes.append(full_route)
        if len(all_routes) > 1:
            fig, ax = ox.plot_graph_routes(
                self.G, all_routes, route_linewidths=1, node_size=0, route_colors=self.colors[0:len(all_routes)])
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        elif len(all_routes) == 1:
            fig, ax = ox.plot_graph_route(
                self.G, all_routes[0], route_linewidth=1, node_size=0)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        else:
            print("No Routes")
            return None

        return fig, ax

    def draw_truck_fig(self, truck_option, best_solution):
        all_routes = []
        for truck_idx, truck_trips in enumerate(best_solution):
            if truck_idx == truck_option - 1:
                for trip_time, locations in truck_trips:
                    full_route = []
                    for i in range(0, len(locations) - 1):
                        route = nx.shortest_path(
                            self.G, locations[i], locations[i+1])
                        full_route.extend(route[1:])
                    all_routes.append(full_route)
        if len(all_routes) > 1:
            fig, ax = ox.plot_graph_routes(
                self.G, all_routes, route_linewidths=1, node_size=0, route_colors=self.colors[0:len(all_routes)])
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        elif len(all_routes) == 1:
            fig, ax = ox.plot_graph_route(
                self.G, all_routes[0], route_linewidth=1, node_size=0)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        else:
            print("No Routes")
            return None

        return fig, ax

    def draw_time_fig(self, time_option, best_solution):
        all_routes = []
        for truck_idx, truck_trips in enumerate(best_solution):
            for trip_time, locations in truck_trips:
                if trip_time == time_option:
                    full_route = []
                    for i in range(0, len(locations) - 1):
                        route = nx.shortest_path(
                            self.G, locations[i], locations[i+1])
                        full_route.extend(route[1:])
                    all_routes.append(full_route)
        if len(all_routes) > 1:
            fig, ax = ox.plot_graph_routes(
                self.G, all_routes, route_linewidths=1, node_size=0, route_colors=self.colors[0:len(all_routes)])
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        elif len(all_routes) == 1:
            fig, ax = ox.plot_graph_route(
                self.G, all_routes[0], route_linewidth=1, node_size=0)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        else:
            print("No Routes")
            return None

        return fig, ax

    def draw_depot_fig(self, depot_option, best_solution):
        all_routes = []
        for truck_idx, truck_trips in enumerate(best_solution):
            for trip_time, locations in truck_trips:
                if locations[0] == depot_option:
                    full_route = []
                    for i in range(0, len(locations) - 1):
                        route = nx.shortest_path(
                            self.G, locations[i], locations[i+1])
                        full_route.extend(route[1:])
                    all_routes.append(full_route)
        if len(all_routes) > 1:
            fig, ax = ox.plot_graph_routes(
                self.G, all_routes, route_linewidths=1, node_size=0, route_colors=self.colors[0:len(all_routes)])
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        elif len(all_routes) == 1:
            fig, ax = ox.plot_graph_route(
                self.G, all_routes[0], route_linewidth=1, node_size=0)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        else:
            print("No Routes")
            return None

        return fig, ax

# Define the start time for the trips


class GeneticAlgorithm:
    def __init__(self, WASTE, TRUCKS, CAPACITIES, depot_nodes_index, collection_nodes, time_location_traffic_matrix, START_TIME) -> None:
        self.WASTE = WASTE
        self.TRUCKS = TRUCKS
        self.CAPACITIES = CAPACITIES
        self.depot_nodes_index = depot_nodes_index
        self.collection_nodes = collection_nodes
        self.locations = depot_nodes_index + collection_nodes
        self.time_location_traffic_matrix = time_location_traffic_matrix
        self.START_TIME = START_TIME
        self.MAX_LOCATIONS_PER_TRIP = 3  # You can parameterize this as needed

        # Define DEAP creators for fitness and individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.create_individual)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        self.toolbox.register("mate", self.mate_individual)
        self.toolbox.register("mutate", self.mut_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate_individual)

        # Define fitness log statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", np.min)
        self.stats.register("avg", np.mean)
        self.stats.register("max", np.max)

    # --- Function to create an individual ---
    def create_individual(self):
        individual = []
        remaining_waste = self.WASTE.copy()
        trips_per_truck = {truck: [] for truck in self.TRUCKS}
        earliest_available_time = {
            truck: self.START_TIME for truck in self.TRUCKS}

        while any(remaining_waste.values()):
            for truck in self.TRUCKS:
                truck_capacity = self.CAPACITIES[truck]
                truck_trips = []
                current_load = 0
                assigned_locations = []

                for location, waste in remaining_waste.items():
                    if remaining_waste[location] > 0:
                        collectable_waste = min(
                            truck_capacity - current_load, remaining_waste[location])
                        current_load += collectable_waste
                        remaining_waste[location] -= collectable_waste
                        assigned_locations.append(location)

                        if current_load >= truck_capacity:
                            break

                if assigned_locations:
                    trip_time = earliest_available_time[truck]
                    first_location_index = self.locations.index(
                        assigned_locations[0])
                    time_to_locations = self.time_location_traffic_matrix[trip_time.strftime(
                        '%Y-%m-%d %H:%M:%S')][first_location_index]

                    depot_index = [self.locations.index(
                        node) for node in self.depot_nodes_index]
                    closest_depot_index = min(
                        depot_index, key=lambda x: time_to_locations[x])
                    depot_start = self.locations[closest_depot_index]
                    assigned_locations = [depot_start] + \
                        assigned_locations + [depot_start]
                    truck_trips.append(
                        (trip_time.strftime('%Y-%m-%d %H:%M:%S'), assigned_locations))

                    earliest_available_time[truck] += timedelta(hours=1)
                    trips_per_truck[truck].extend(truck_trips)

        for truck, trips in trips_per_truck.items():
            individual.append(trips)

        for truck_trips in individual:
            available_time_slots = list(
                self.time_location_traffic_matrix.keys())
            random_indices = random.sample(
                range(len(available_time_slots)), len(truck_trips))
            random_indices.sort()

            for i, index in enumerate(random_indices):
                truck_trips[i] = (
                    available_time_slots[index], truck_trips[i][1])

        return individual

    # --- Function to evaluate an individual ---
    def evaluate_individual(self, individual):
        total_time = 0
        unused_trucks = 0

        for truck_trips in individual:
            if not truck_trips:
                unused_trucks += 1
                continue

            for departure_time, trip_route in truck_trips:
                distance_matrix = self.time_location_traffic_matrix[departure_time]
                node_id_to_index = {node_id: idx for idx,
                                    node_id in enumerate(self.locations)}
                trip_time = 0

                for j in range(len(trip_route) - 1):
                    from_node = trip_route[j]
                    to_node = trip_route[j + 1]
                    from_idx = node_id_to_index[from_node]
                    to_idx = node_id_to_index[to_node]

                    travel_time = distance_matrix[from_idx][to_idx]
                    trip_time += travel_time

                total_time += trip_time

        total_time += unused_trucks * 1000  # Adjust penalty factor
        return total_time,

    # --- Mutation function ---
    def mut_individual(self, individual):
        all_trips = []
        for truck_idx, truck_schedule in enumerate(individual):
            for trip in truck_schedule:
                trip_time, locations = trip
                all_trips.append(
                    (truck_idx, self.format_time_as_string(trip_time), locations))

        random.shuffle(all_trips)

        for i in range(len(all_trips)):
            if i < len(all_trips) - 1:
                truck_idx1, time1, locations1 = all_trips[i]
                truck_idx2, time2, locations2 = all_trips[i + 1]

                if not self.is_time_slot_taken(individual, truck_idx1, time2) and not self.is_time_slot_taken(individual, truck_idx2, time1):
                    self.swap_trips(individual, truck_idx1,
                                    time1, truck_idx2, time2)
                else:
                    self.reassign_trip(
                        individual, truck_idx1, time1, locations1)

        for truck_idx, truck_schedule in enumerate(individual):
            for trip_idx, (trip_time, locations) in enumerate(truck_schedule):
                individual[truck_idx][trip_idx] = (
                    self.format_time_as_string(trip_time), locations)
            individual[truck_idx] = sorted(
                individual[truck_idx], key=lambda x: x[0])

        return individual,

    # --- Helper Functions ---
    def is_time_slot_taken(self, individual, truck_idx, time_slot):
        for trip_time, _ in individual[truck_idx]:
            if trip_time == time_slot:
                return True
        return False

    def swap_trips(self, individual, truck_idx1, time1, truck_idx2, time2):
        for i, (trip_time1, _) in enumerate(individual[truck_idx1]):
            if trip_time1 == time1:
                for j, (trip_time2, _) in enumerate(individual[truck_idx2]):
                    if trip_time2 == time2:
                        individual[truck_idx1][i], individual[truck_idx2][j] = individual[truck_idx2][j], individual[truck_idx1][i]
                        return

    def reassign_trip(self, individual, truck_idx, current_time, locations):
        available_times = self.find_available_times(individual, truck_idx)
        if available_times:
            new_time = random.choice(available_times)
            for i, (trip_time, _) in enumerate(individual[truck_idx]):
                if trip_time == current_time:
                    individual[truck_idx][i] = (new_time, locations)
                    return

    def find_available_times(self, individual, truck_idx):
        all_times = [self.START_TIME + timedelta(hours=i) for i in range(10)]
        taken_times = [trip_time for trip_time, _ in individual[truck_idx]]
        available_times = [time.strftime(
            '%Y-%m-%d %H:%M:%S') for time in all_times if time.strftime('%Y-%m-%d %H:%M:%S') not in taken_times]
        return available_times

    def format_time_as_string(self, time):
        if isinstance(time, datetime):
            return time.strftime('%Y-%m-%d %H:%M:%S')
        return time

    # --- Mating function ---
    def mate_individual(self, parent1, parent2):
        offspring1 = deepcopy(parent1)
        offspring2 = deepcopy(parent2)

        num_trucks = len(parent1)
        crossover_point = random.randint(1, num_trucks - 1)
        for truck_idx in range(crossover_point, num_trucks):
            offspring1[truck_idx], offspring2[truck_idx] = offspring2[truck_idx], offspring1[truck_idx]

        offspring1 = self.resolve_conflicts_with_partial_collection(offspring1)
        offspring2 = self.resolve_conflicts_with_partial_collection(offspring2)

        return offspring1, offspring2

    # --- Conflict resolution function ---
    def resolve_conflicts_with_partial_collection(self, offspring):
        waste_collected = {loc: 0 for loc in self.WASTE.keys()}
        all_locations = set(self.WASTE.keys())

        for truck_idx, truck_trips in enumerate(offspring):
            for trip_idx, trip in enumerate(truck_trips):
                trip_time, locations = trip
                updated_locations = []
                current_load = 0
                truck_depot = locations[0]

                for location in locations:
                    if location == truck_depot:
                        continue

                    remaining_waste = self.WASTE[location] - \
                        waste_collected[location]
                    if remaining_waste > 0:
                        truck_capacity = self.CAPACITIES[self.TRUCKS[truck_idx]]
                        collectable_waste = min(
                            remaining_waste, truck_capacity - current_load)

                        current_load += collectable_waste
                        waste_collected[location] += collectable_waste
                        updated_locations.append(location)

                        if current_load >= truck_capacity or len(updated_locations) >= self.MAX_LOCATIONS_PER_TRIP:
                            break

                if updated_locations:
                    updated_locations.insert(0, truck_depot)
                    updated_locations.append(truck_depot)

                    offspring[truck_idx][trip_idx] = (
                        trip_time, updated_locations)
                else:
                    offspring[truck_idx].remove(trip)

        remaining_locations = all_locations - \
            set(
                loc for trips in offspring for trip in trips for loc in trip[1] if loc not in self.depot_nodes_index)

        for loc in remaining_locations:
            remaining_waste = self.WASTE[loc] - waste_collected[loc]
            if remaining_waste > 0:
                for truck_idx, truck_trips in enumerate(offspring):
                    if truck_trips:
                        last_trip_time = truck_trips[-1][0]
                        last_trip = truck_trips[-1][1]

                        if len(last_trip) - 2 < self.MAX_LOCATIONS_PER_TRIP:
                            last_trip.insert(-1, loc)
                            break

        return offspring

    # Run the Genetic Algorithm

    def run_ga(self):
        print("Starting Genetic Algorithm...")

        # Initialize population
        # Adjust population size as needed
        population = self.toolbox.population(n=20)
        print(f"Initial population generated with {len(population)} individuals.")

        # Run the GA using DEAP's eaSimple
        result_population, log = algorithms.eaSimple(
            population, self.toolbox, cxpb=0.3, mutpb=0.2, ngen=750, verbose=True, stats=self.stats
        )

        print("GA run complete. Extracting the best individual...")

        # Extract the best solution
        best_individual = tools.selBest(result_population, 1)[0]
        print("Best individual:", best_individual)

        return best_individual, log
