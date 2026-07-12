import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from IPython.display import HTML
from matplotlib import cm
from matplotlib.colors import Normalize

class TrafficSimulation:
    def __init__(self, road_length=500, num_lanes=2, dt=0.5):  # Reduced dt for smoother animation
        # Core parameters
        self.road_length = max(100, road_length)
        self.dt = max(0.1, min(dt, 1.0))  # Smaller timestep
        self.num_lanes = max(1, min(num_lanes, 4))
        self.dx = 20
        self.x_grid = np.arange(0, self.road_length + self.dx, self.dx)

        # Speed parameters (reduced for better visualization)
        self.rho_max = 0.15  # vehicles/meter/lane
        self.v_max = 5.0     # Reduced from 10 to 5 m/s (~18 km/h)
        self.v_min = 0.0

        # IDM parameters (adjusted for better stopping behavior)
        self.T = 1.8    # Increased time headway
        self.a = 0.5    # Reduced acceleration
        self.b = 2.0    # Increased braking
        self.s0 = 3.0   # Minimum gap

        # Transition parameters
        self.grad_threshold = 0.01

        # Traffic lights with longer cycles
        self.traffic_lights = {
            250: {
                'position': min(250, self.road_length-50),
                'state': 'red',  # Start with red light
                'timer': 25,      # Start in red phase
                'cycle': 40,      # Longer cycle
                'phases': [
                    (20, 'green'),
                    (30, 'yellow'),
                    (40, 'red')
                ],
                'stop_line': 5,
                'effective_distance': 50  # Distance at which cars react to light
            }
        }

        # Initialize vehicles
        self.vehicles = []
        self.initialize_vehicles()

        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(16, 6))
        self.scatter = None
        self.norm = Normalize(vmin=self.v_min, vmax=self.v_max)
        self.cmap = cm.RdYlGn
        plt.close()

    def initialize_vehicles(self):
        """Initialize vehicles with proper spacing"""
        num_vehicles = min(20, int(self.road_length * self.rho_max * self.num_lanes))  # Fewer vehicles
        for i in range(num_vehicles):
            lane = i % self.num_lanes
            spacing = self.road_length / num_vehicles
            pos = i * spacing
            speed = np.clip(3 + np.random.randn(), self.v_min, self.v_max)  # Lower initial speed
            self.add_vehicle(pos, speed, lane)

    def add_vehicle(self, pos, speed, lane):
        """Add a vehicle to the simulation"""
        self.vehicles.append({
            'id': len(self.vehicles),
            'position': float(pos % self.road_length),
            'speed': float(np.clip(speed, self.v_min, self.v_max)),
            'lane': int(lane % self.num_lanes),
            'length': 5.0,
            'use_micro': False,
            'leader': None,
            'gap': 20.0,
            'last_update': 0,
            'stopping_for_light': False
        })

    def update_traffic_lights(self):
        """Update traffic light states with proper timing"""
        for light in self.traffic_lights.values():
            light['timer'] = (light['timer'] + self.dt) % light['cycle']
            for threshold, state in light['phases']:
                if light['timer'] < threshold:
                    light['state'] = state
                    break

    def compute_density_field(self):
        """Calculate traffic density"""
        density = np.zeros((len(self.x_grid), self.num_lanes))
        for v in self.vehicles:
            pos_idx = v['position'] / self.dx
            lower_idx = min(int(np.floor(pos_idx)), len(self.x_grid)-1)
            upper_idx = min(int(np.ceil(pos_idx)), len(self.x_grid)-1)
            weight = pos_idx - lower_idx

            lane = min(v['lane'], self.num_lanes-1)
            if lower_idx >= 0:
                density[lower_idx, lane] += (1-weight) / self.dx
            if upper_idx >= 0:
                density[upper_idx, lane] += weight / self.dx

        return gaussian_filter(density, sigma=1.0, mode='wrap')

    def update_microscopic(self, v, current_time):
        """Improved IDM model with proper traffic light response"""
        # Find leader vehicle
        leaders = []
        for ov in self.vehicles:
            if (ov['lane'] == v['lane'] and
                (ov['position'] - v['position']) % self.road_length > 0 and
                ov['id'] != v['id']):
                leaders.append(ov)

        # Check traffic lights - improved logic
        stop_distance = float('inf')
        stop_required = False
        for light in self.traffic_lights.values():
            dist = (light['position'] - v['position']) % self.road_length
            if 0 < dist < light['effective_distance']:
                if light['state'] == 'red':
                    effective_dist = dist - light['stop_line']
                    if 0 < effective_dist < stop_distance:
                        stop_distance = effective_dist
                        stop_required = True
                elif light['state'] == 'yellow':
                    effective_dist = dist - light['stop_line']
                    if 0 < effective_dist < stop_distance:
                        stop_distance = effective_dist
                        stop_required = True

        # Create virtual leader if stopping for light
        if stop_required and stop_distance < light['effective_distance']:
            virtual_leader = {
                'position': (v['position'] + stop_distance) % self.road_length,
                'speed': 0,  # Stopped at light
                'length': 0
            }
            leaders.append(virtual_leader)
            v['stopping_for_light'] = True
        else:
            v['stopping_for_light'] = False

        # IDM calculation with improved parameters
        if leaders:
            leader = min(leaders, key=lambda x: (x['position'] - v['position']) % self.road_length)
            gap = max(0.1, (leader['position'] - v['position'] - v['length']) % self.road_length)
            dv = v['speed'] - leader['speed']
        else:
            gap = self.road_length
            dv = 0

        s_star = self.s0 + max(0, v['speed']*self.T + (v['speed']*dv)/(2*np.sqrt(self.a*self.b)))
        accel = self.a * (1 - (v['speed']/self.v_max)**4 - (s_star/gap)**2)

        # Apply acceleration limits
        v['speed'] = np.clip(v['speed'] + accel*self.dt, self.v_min, self.v_max)
        v['position'] = (v['position'] + v['speed']*self.dt) % self.road_length
        v['gap'] = gap
        v['leader'] = leaders[0]['id'] if leaders else None
        v['last_update'] = current_time

    def update_macroscopic(self, density):
        """Update using LWR model with reduced speeds"""
        flux = np.zeros_like(density)
        for lane in range(self.num_lanes):
            flux[:, lane] = np.where(
                density[:, lane] <= self.rho_max/2,
                density[:, lane] * self.v_max * 0.8,  # Reduced free-flow speed
                (self.rho_max - density[:, lane]) * self.v_max * 0.6
            )

        new_rho = np.zeros_like(density)
        for lane in range(self.num_lanes):
            for i in range(len(self.x_grid)):
                left = i-1 if i > 0 else -1
                new_rho[i, lane] = density[i, lane] - (self.dt/self.dx)*(flux[i, lane] - flux[left, lane])

        return np.clip(new_rho, 0, self.rho_max)

    def update_transitions(self, density, gradient):
        """Determine which model to use for each vehicle"""
        for v in self.vehicles:
            idx = min(int(v['position'] / self.dx), len(self.x_grid)-1)
            lane = min(v['lane'], self.num_lanes-1)

            near_light = any(
                abs(light['position'] - v['position']) < 50
                for light in self.traffic_lights.values()
            )

            v['use_micro'] = (
                gradient[idx, lane] > self.grad_threshold or
                density[idx, lane] > self.rho_max*0.6 or
                near_light or
                v['gap'] < 15 or
                v['stopping_for_light']
            )

    def update(self, frame):
        current_time = frame * self.dt

        try:
            self.update_traffic_lights()

            density = self.compute_density_field()
            gradient = np.abs(np.gradient(density, axis=0)) / self.dx
            self.update_transitions(density, gradient)

            # Update vehicles
            for v in self.vehicles:
                if v['use_micro']:
                    self.update_microscopic(v, current_time)
                else:
                    idx = min(int(v['position'] / self.dx), len(self.x_grid)-1)
                    lane = min(v['lane'], self.num_lanes-1)
                    macro_v = self.v_max * 0.8 * max(0, (1 - density[idx, lane]/self.rho_max))  # Reduced speed
                    v['speed'] = np.clip(0.9*macro_v + 0.1*v['speed'], self.v_min, self.v_max)
                    v['position'] = (v['position'] + v['speed']*self.dt) % self.road_length
        except Exception as e:
            print(f"Update error: {str(e)[:100]}")

    def animate(self, frames=200):
        def init():
            self.ax.clear()
            self.ax.set_xlim(0, self.road_length)
            self.ax.set_ylim(-0.5, self.num_lanes-0.5)
            self.ax.set_yticks(range(self.num_lanes))
            self.ax.set_yticklabels([f"Lane {i}" for i in range(self.num_lanes)])
            self.ax.set_xlabel("Position (m)")
            self.ax.grid(True, axis='x', alpha=0.3)

            # Create initial scatter plot
            pos = []
            lanes = []
            speeds = []
            for lane in range(self.num_lanes):
                for v in [v for v in self.vehicles if v['lane'] == lane]:
                    pos.append(v['position'])
                    lanes.append(lane)
                    speeds.append(v['speed'])

            self.scatter = self.ax.scatter(pos, lanes, c=speeds,
                                        cmap=self.cmap, norm=self.norm,
                                        s=60, zorder=4)

            # Add colorbar once
            if not hasattr(self, 'cbar'):
                self.cbar = self.fig.colorbar(self.scatter, ax=self.ax)
                self.cbar.set_label('Speed (m/s)')

            # Draw traffic lights
            for light in self.traffic_lights.values():
                # Stop line
                self.ax.axvline(light['position'] - light['stop_line'],
                               color='red' if light['state'] != 'green' else 'green',
                               linestyle='--', alpha=0.5)
                # Light box
                self.ax.add_patch(plt.Rectangle(
                    (light['position']-5, -0.5), 10, self.num_lanes,
                    color='black', alpha=0.1
                ))
                # Lights per lane
                for lane in range(self.num_lanes):
                    self.ax.scatter(
                        light['position'], lane,
                        color=light['state'],
                        s=100, marker='s', zorder=3
                    )

            return [self.scatter]

        def update(frame):
            try:
                self.update(frame)

                # Update vehicle positions and colors
                pos = []
                lanes = []
                speeds = []
                micro = []
                stopping = []

                for lane in range(self.num_lanes):
                    lane_vehs = [v for v in self.vehicles if v['lane'] == lane]
                    pos.extend([v['position'] for v in lane_vehs])
                    lanes.extend([lane] * len(lane_vehs))
                    speeds.extend([v['speed'] for v in lane_vehs])
                    micro.extend([v['use_micro'] for v in lane_vehs])
                    stopping.extend([v['stopping_for_light'] for v in lane_vehs])

                # Update scatter plot data
                self.scatter.set_offsets(np.column_stack([pos, lanes]))
                self.scatter.set_array(np.array(speeds))

                # Clear and redraw
                self.ax.clear()
                init()  # Reinitialize to maintain consistent appearance

                # Mark microscopic vehicles
                micro_pos = [p for p, m in zip(pos, micro) if m]
                micro_lanes = [l for l, m in zip(lanes, micro) if m]
                if micro_pos:
                    self.ax.scatter(micro_pos, micro_lanes, facecolors='none',
                                  edgecolors='black', s=80, linewidths=1.5, zorder=5)

                # Mark vehicles stopping for lights
                stop_pos = [p for p, s in zip(pos, stopping) if s]
                stop_lanes = [l for l, s in zip(lanes, stopping) if s]
                if stop_pos:
                    self.ax.scatter(stop_pos, stop_lanes, marker='x',
                                  color='red', s=50, zorder=6)

                self.ax.set_title(
                    f"Traffic Simulation | Time: {frame*self.dt:.1f}s | "
                    f"Vehicles: {len(self.vehicles)} | "
                    f"Micro: {sum(v['use_micro'] for v in self.vehicles)} | "
                    f"Stopping: {sum(v['stopping_for_light'] for v in self.vehicles)}"
                )

                return [self.scatter]

            except Exception as e:
                print(f"Rendering error: {str(e)[:100]}")
                return []

        ani = FuncAnimation(self.fig, update, frames=frames,
                          init_func=init, blit=False, interval=100)
        return ani

# Initialize and run
if __name__ == "__main__":
    print("=== Starting Traffic Simulation ===")
    sim = TrafficSimulation(road_length=500, num_lanes=2, dt=0.5)
    HTML(sim.animate(frames=200).to_jshtml())
