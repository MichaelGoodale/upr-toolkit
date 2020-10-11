import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy import signal
from scipy.io import wavfile

class TimitData:
    VOWELS = [ "iy",  "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah",
            "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "ax-h"]

    STOPS=["b", "d", "g", "p", "t", "k", "dx", "q"]

    AFFRICATES = ["jh", "ch"]
    FRICATIVES = [ "s" ,"sh" ,"z" ,"zh" ,"f" ,"th" ,"v" ,"dh"]
    NASALS = ["m", "n", "ng", "em", "en", "eng", "nx"]
    SEMIVOWELS = ["l", "r", "w", "y", "hh", "hv", "el"]

    PHONES = VOWELS + STOPS + AFFRICATES + FRICATIVES + NASALS + SEMIVOWELS

    def __init__(self, get_C_function, C_time_function, max_files=None, dropna=False, timit_dir='timit/TIMIT/train/'):
        self.TIMIT_DIR = timit_dir
        C = {}
        phones_df = []
        for filename in tqdm(self.get_timit_files(n=max_files)):
            C[filename["wav"]] = get_C_function(filename["wav"])
            phones_df.append(TimitData.get_phone_timing(filename["phn"]))
        phones_df = pd.concat(phones_df)
        phones_df["start_c"] = phones_df["start"].apply(C_time_function)
        phones_df["end_c"] = phones_df["end"].apply(C_time_function)
        phones_df["c"] = phones_df.apply(lambda x: C[x["wav"]][:, :, x["start_c"]:x["end_c"]].mean(axis=2), axis=1)
        if dropna:
            phones_df = phones_df.dropna()
            phones_df = phones_df[phones_df["c"].apply(lambda x: not np.isnan(x).any())]
        self.phones_df = phones_df

        self.spectograms = {}

    def get_timit_files(self, n=1):
        '''Search the directory for all TIMIT files'''

        ret_list = []
        for root, dirs, files in os.walk(self.TIMIT_DIR):
            for f in filter(lambda x: x.endswith(".wav"), files):
                f = os.path.join(root, f)
                ret_list.append({"wav": f,
                                 "txt": f.replace(".wav", ".txt"),
                                 "phn": f.replace(".wav", ".phn"),
                                 "wrd": f.replace(".wav", ".wrd")})
                if n is not None and n != 0 and len(ret_list) >= n:
                    return ret_list
        return ret_list

    def get_spectrogram(wav):
        '''Given a wave file, return its related spectogram.

        To plot: 
        >> times, frequencies, spectogram = get_spectrogram(wav)
        >> plt.pcolormesh(times, frequencies, spectrogram)
        >> plt.imshow(spectrogram)
        >> plt.ylabel('Frequency [Hz]')
        >> plt.xlabel('Time [sec]')
        >> plt.show()
        '''

        if wav not in self.spectograms:
            sample_rate, samples = wavfile.read('path-to-mono-audio-file.wav')
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
            self.spectrograms['wav'] = (frequencies, times, spectrogram)
        return self.spectograms['wav']


    def get_phone_timing(phones):
        '''Given a .PHN file, parse the phones in it and preprocess them to find basic phonological attributes
        Returns a pandas dataframe'''

        phone_times = []
        with open(phones) as f:
            prev_phone = None
            for l in f:
                start, end, phone = l.strip().split(" ")
                phone_times.append((int(start), int(end), phone, {}))

        words_times = []
        with open(phones.replace('.phn', '.wrd')) as f:
            for l in f:
                start, end, word = l.strip().split(" ")
                words_times.append((int(start), int(end), word))
        for i, (start, end, phone, info) in enumerate(phone_times):
            word = None
            word_s = None
            word_e = None
            for s, e, w in words_times:
                if start >= s and end <= e:
                    word = w
                    word_s = s
                    word_e = e

            info["word"] = word
            info["wav"] = phones.replace(".phn", ".wav")
            if word_s == start:
                info["word_pos"] = "initial"
            elif word_e == end:
                info["word_pos"] = "final"
            else:
                info["word_pos"] = "medial"

        columns = ["start", "end", "phone"]
        df = pd.DataFrame(columns = columns +sorted(info.keys()))
        df = df.append([{**{columns[i]:x[i] for i in range(3)},  **x[3]} for x in phone_times])
        return df
