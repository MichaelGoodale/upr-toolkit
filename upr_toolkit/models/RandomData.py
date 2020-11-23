import numpy as np
import subprocess 
from .model import ModelData

class RandomData(ModelData):
    def get_in_c_time(self, time):
        ratio = 0.006196484134864083
        return int(ratio*time)

    def calculate_c(self, filename):
        result = subprocess.run(['soxi', '-s', filename], capture_output=True)
        number_of_samples = int(result.stdout)
        return np.random.uniform(size=(1, self.d, self.get_in_c_time(number_of_samples)))

    def __init__(self, d=256, seed=1337, max_files=None):
        self.d = d
        np.random.seed(seed)
        super().__init__(None, max_files=max_files, multi_proc_c=True, calculate_formants=False)
