import sys
import os

class SimulationRunner:
    def __init__(self):
        self.setup_paths()
    
    def setup_paths(self):
        # Add the parent directory to the Python path
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if parent_path not in sys.path:
            sys.path.append(parent_path)

        # Add the simulation directory to the Python path
        simulation_path = os.path.abspath(os.path.join(parent_path, 'simulation'))
        if simulation_path not in sys.path:
            sys.path.append(simulation_path)

        # Add the utils directory to the Python path
        utils_path = os.path.abspath(os.path.join(simulation_path, 'utils'))
        if utils_path not in sys.path:
            sys.path.append(utils_path)

    def run(self, start, goal, camcoordinates,itemsize,wid,iterations):
        # Import run_simulation from the main module in the simulation directory
        from simulation.main import run_simulation
        return run_simulation(start, goal, camcoordinates,itemsize,wid,iterations)

if __name__ == "__main__":
    runner = SimulationRunner()
    start = (2, 1)  # Initial start position (row, col)
    goal = (13, 1)  # Goal position (row, col)
    camcoordinates = [(1, 0), (13, 0)]
    runner.run(start, goal, camcoordinates,"small",110,1)