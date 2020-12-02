import numpy as np
import subprocess 
from .model import ModelData

class RandomData(ModelData):
    def get_in_c_time(self, time):
        ratio = 0.006196484134864083
        return int(ratio*time)

    def calculate_c(self, filename):
        result = subprocess.run(['soxi', '-s', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise ChildProcessError("Command soxi failed with errorcode {}, and stderr: {}".format(result.returncode, result.stderr))
        number_of_samples = int(result.stdout)
        return np.random.uniform(size=(1, self.d, self.get_in_c_time(number_of_samples) + 1))

    def __init__(self, d=256, seed=1337, max_files=None, **kwargs):
        self.d = d
        np.random.seed(seed)
        super().__init__(cache_file=None, max_files=max_files, multi_proc_c=True, calculate_formants=False, **kwargs)
