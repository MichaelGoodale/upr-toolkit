import os
import pandas as pd

def get_timit_files(n=1):
    ret_list = []
    for root, dirs, files in os.walk("./timit/timit"):
        for f in filter(lambda x: x.endswith(".wav"), files):
            f = os.path.join(root, f)
            ret_list.append({"wav": f,
                             "txt": f.replace(".wav", ".txt"),
                             "phn": f.replace(".wav", ".phn"),
                             "wrd": f.replace(".wav", ".wrd")})
            if n != 0 and len(ret_list) >= n:
                return ret_list
    return ret_list

def get_phone_timing(phones, return_df=True):
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

    if return_df==True:
        columns = ["start", "end", "phone"]
        df = pd.DataFrame(columns = columns +sorted(info.keys()))
        df = df.append([{**{columns[i]:x[i] for i in range(3)},  **x[3]} for x in phone_times])
        return df
    return phone_times
