import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy import signal
from scipy.io import wavfile

import parselmouth
from parselmouth.praat import call

import multiprocessing

class TimitData:
    VOWELS = [ "iy",  "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah",
            "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "ax-h"]

    STOPS=["b", "d", "g", "p", "t", "k", "dx", "q"]

    AFFRICATES = ["jh", "ch"]
    FRICATIVES = [ "s" ,"sh" ,"z" ,"zh" ,"f" ,"th" ,"v" ,"dh"]
    NASALS = ["m", "n", "ng", "em", "en", "eng", "nx"]
    SEMIVOWELS = ["l", "r", "w", "y", "hh", "hv", "el"]

    PHONES = VOWELS + STOPS + AFFRICATES + FRICATIVES + NASALS + SEMIVOWELS

    def __init__(self, get_C_function=None, C_time_function=None, max_files=None, dropna=False,
            timit_dir='timit/TIMIT/train/', load_file=None, save_file="timitdata.ft",
            n_proc=multiprocessing.cpu_count(), multi_proc_c=False,
            calculate_formants=True):
        if load_file is not None:
            self.phones_df = pd.read_feather(load_file)
            C = np.load(load_file + ".npy")
            self.phones_df["c"] = [c for c in np.split(C, self.phones_df["c_lengths"].cumsum()[:-1], axis=-1)]
        else:
            if get_C_function is None or C_time_function is None:
                raise Error("You must provide a function for get_C_function and C_time_function")
            self.TIMIT_DIR = timit_dir

            lexical_info = TimitData.get_phone_data()
            timit_files = self.get_timit_files(n=max_files)

            pool = multiprocessing.Pool(processes=n_proc)

            if multi_proc_c:
                C = list(tqdm(pool.imap(get_C_function, [f["wav"] for f in timit_files]), total=len(timit_files)))
                C = {f["wav"]: v for f, v in zip(timit_files, C)}
            else:
                C = {}
                for f in tqdm(timit_files):
                    C[f["wav"]] = get_C_function(f["wav"])

            load_phone_args = [(f['phn'], lexical_info[lexical_info['file'] == self.get_lex_file(f)])
                for f in timit_files]

            phones_df = pool.starmap(TimitData.get_phone_timing, load_phone_args)
            if calculate_formants:
                phones_df = list(tqdm(pool.imap(TimitData.get_formants, phones_df), total=len(timit_files)))
            pool.close()

            phones_df = pd.concat(phones_df)
            phones_df["start_c"] = phones_df["start"].apply(C_time_function)
            phones_df["end_c"] = phones_df["end"].apply(C_time_function)

            phones_df["c"] = phones_df.apply(lambda x: np.squeeze(C[x["wav"]][:, :, x["start_c"]:x["end_c"]], axis=0), axis=1)
            phones_df["c_lengths"] = phones_df["c"].apply(lambda x: x.shape[-1])

            if dropna:
                phones_df = phones_df[phones_df["c"].apply(lambda x: not np.isnan(x).any())]
                phones_df.reset_index(inplace=True)

            self.phones_df = phones_df
            if save_file is not None:
                self.phones_df.loc[:, self.phones_df.columns !=  "c"].to_feather(save_file)
                np.save(save_file + '.npy', np.hstack(self.phones_df["c"].values))

    def get_lex_file(self, filename):
        return filename['wav'].replace('.wav', '').replace(self.TIMIT_DIR, self.TIMIT_DIR.split('/')[-2]+'/')

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

    def get_formant_time(start, end):
        return ((0.5*(end - start)) + start) /  16000

    def get_formants(phones):
        wav_file = phones.loc[0, "wav"]
        snd = parselmouth.Sound(wav_file)
        formants = call(snd, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
        for i in range(1, 5):
            phones["f{}".format(i)] = phones.apply(lambda r: np.nan if r["phone"] not in TimitData.VOWELS
                    else call(formants, "Get value at time", i,
                              TimitData.get_formant_time(r['start'], r['end']), 'Hertz', 'Linear'),
                    axis=1)
        return phones

    def get_phone_timing(phones, df):
        '''Given a .PHN file, parse the phones in it and preprocess them to find basic phonological attributes
        Returns a pandas dataframe'''
        phone_times = []
        with open(phones) as f:
            prev_phone = 'null'
            for l, (_, phone_row) in zip(f, df[~df['phone'].isin(['+', '-'])].iterrows()):
                start, end, phone = l.strip().split(" ")
                if phone_row['phoneme'] == '+':
                    phone_times[-1][1] = int(end)
                    phone_times[-1][2] = phone_times[-1][2]+phone
                else: 
                    phone_times.append([int(start), int(end), phone] + list(phone_row[["phoneme", "stress", "boundary", "word"]]))
        df = pd.DataFrame(phone_times, columns=["start", "end", "phone", "phoneme", "stress", "boundary", "word"])
        df["wav"] = phones.replace('.phn', '.wav')
        return df

    def get_phone_data(word_align_file='/home/michael/Documents/Cogmaster/M1/S1/stage/timit/wrdalign.timit'):
        datas = []
        with open(word_align_file) as f:
            for l in f:
                if l[0] == '%':
                    continue
                elif l[0] == '#':
                    curr_file = l.strip().replace('#', '')
                else:
                    data = l.strip().split('\t')
                    if len(data) != 6:
                        data.append('-')
                    data.append(curr_file)
                    datas.append(data)
        df = pd.DataFrame(datas, columns=['phoneme', 'phone', 'distance', 'stress', 'boundary', 'word', 'file'])

        #Make sure every phone has its associated word listed, not just the first.
        df['word'] = df['word'].replace('-', method='ffill')


        #Ensures that if the phone that is the boundary is deleted, we still have the boundary labeled 
        #unless there is already a boundary there. 
        df.loc[(df['phone'].shift(1).isin(['+', '-'])) & (df['boundary'] == '0'), "boundary"] = pd.NA
        df['boundary'] = df['boundary'].fillna(method='ffill')
        return df

