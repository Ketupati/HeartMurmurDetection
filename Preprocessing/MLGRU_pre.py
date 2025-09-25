from common import *


def butter_lowpass_filter(data, cutoff=400, fs=4000, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def normalize_min_max(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def parse_txt_metadata(txt_folder):
    metadata = []
    for fname in os.listdir(txt_folder):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(txt_folder, fname)
        try:
            with open(path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # header: patient_id, num_recordings, ?
            patient_id, num_recordings, _ = lines[0].split()
            recordings = []
            for line in lines[1:int(num_recordings)+1]:
                parts = line.split()
                # parts: location hea_path wav_path tsv_path
                if len(parts) >= 4:
                    recordings.append({
                        'location': parts[0],
                        'hea_path': parts[1],
                        'wav_path': parts[2],
                        'tsv_path': parts[3]
                    })

            info = {}
            for line in lines[int(num_recordings)+1:]:
                if line.startswith("#"):
                    kv = line[1:].split(":", 1)
                    if len(kv) == 2:
                        info[kv[0].strip()] = kv[1].strip()

            murmur_locs = []
            if "Murmur locations" in info and str(info["Murmur locations"]).lower() != "nan":
                murmur_locs = [loc.strip() for loc in info["Murmur locations"].split()]

            murmur_status = info.get("Murmur", "Unknown").strip()
            if murmur_status not in ["Present", "Absent", "Unknown"]:
                murmur_status = "Unknown"

            for rec in recordings:
                metadata.append({
                    "patient_id": patient_id,
                    "wav_path": rec['wav_path'],
                    "location": rec['location'],
                    "age": info.get("Age", ""),
                    "sex": info.get("Sex", ""),
                    "height": float(info.get("Height", "nan")) if info.get("Height") not in [None, ""] else np.nan,
                    "weight": float(info.get("Weight", "nan")) if info.get("Weight") not in [None, ""] else np.nan,
                    "pregnancy": str(info.get("Pregnancy status", "")).lower() == "true",
                    "murmur": murmur_status,
                    "murmur_locations": murmur_locs,
                    "outcome": str(info.get("Outcome", "")).lower() == "abnormal"
                })
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            continue
    return pd.DataFrame(metadata)


class PCGDataset(Dataset):
    def __init__(self, data_dir, txt_folder, seq_len=SEQ_LEN):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.metadata = parse_txt_metadata(txt_folder)
        # simple mapping for murmur classes
        self.murmur_map = {"Present": 0, "Absent": 1, "Unknown": 2}
        self.metadata['murmur_encoded'] = self.metadata['murmur'].map(self.murmur_map).fillna(2).astype(int)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        wav_path = os.path.join(self.data_dir, row['wav_path'])
        try:
            waveform, fs = torchaudio.load(wav_path)   # [channels, samples]
        except Exception as e:
            # return a dummy sample so collate_fn can filter it out
            print(f"Error loading {wav_path}: {e}")
            return None

        signal = waveform[0].numpy()                 # single-channel
        # apply lowpass
        try:
            signal = butter_lowpass_filter(signal)
        except Exception:
            pass
        # downsample by factor 4 (as your earlier code)
        signal = signal[::4]
        # truncate/pad to seq_len
        if len(signal) >= self.seq_len:
            sig = signal[:self.seq_len]
        else:
            sig = np.zeros(self.seq_len, dtype=np.float32)
            sig[:len(signal)] = signal
        sig = normalize_min_max(sig)
        pcg_tensor = torch.tensor(sig, dtype=torch.float32)   # [seq_len]

        murmur_label = int(row['murmur_encoded'])
        outcome_label = int(row['outcome'])

        return pcg_tensor, torch.tensor(murmur_label, dtype=torch.long), torch.tensor(outcome_label, dtype=torch.float32)
