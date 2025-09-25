from common import *

def butter_lowpass_filter(data, cutoff=400, fs=4000, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def extract_spectrogram(signal, fs=1000):
    f, t, Zxx = stft(signal, fs=fs, nperseg=40, noverlap=24, nfft=200)
    spectrogram = np.abs(Zxx)
    return spectrogram

def normalize_min_max(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

# ------------------------------
# Custom Dataset
# ------------------------------

def parse_txt_metadata(txt_folder):
    metadata = []
    for fname in os.listdir(txt_folder):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(txt_folder, fname)
        try:
            with open(path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Parse header line
            patient_id, num_recordings, _ = lines[0].split()

            # Parse recording lines
            recordings = []
            for line in lines[1:int(num_recordings)+1]:
                parts = line.split()
                recordings.append({
                    'location': parts[0],
                    'hea_path': parts[1],
                    'wav_path': parts[2],
                    'tsv_path': parts[3]
                })

            # Parse metadata lines (starting with #)
            info = {}
            for line in lines[int(num_recordings)+1:]:
                if line.startswith("#"):
                    key_value = line[1:].split(":", 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        info[key] = value

            # Get murmur locations
            murmur_locs = []
            if "Murmur locations" in info and info["Murmur locations"].lower() != "nan":
                murmur_locs = [loc.strip() for loc in info["Murmur locations"].split() if loc.strip()]

            # Handle murmur status (Present/Absent/Unknown)
            murmur_status = info.get("Murmur", "Unknown").strip()
            if murmur_status not in ["Present", "Absent", "Unknown"]:
                murmur_status = "Unknown"

            # Create metadata entry for each recording
            for rec in recordings:
                metadata.append({
                    "patient_id": patient_id,
                    "wav_path": rec['wav_path'],
                    "location": rec['location'],
                    "age": info.get("Age", ""),
                    "sex": info.get("Sex", ""),
                    "height": float(info.get("Height", "nan")),
                    "weight": float(info.get("Weight", "nan")),
                    "pregnancy": info.get("Pregnancy status", "").lower() == "true",
                    "murmur": murmur_status,  # Now one of "Present", "Absent", or "Unknown"
                    "murmur_locations": murmur_locs,
                    "outcome": info.get("Outcome", "").lower() == "abnormal"
                })

        except Exception as e:
            print(f"Error parsing {path}: {e}")
            continue

    return pd.DataFrame(metadata)



class PCGDataset(Dataset):
    def __init__(self, data_dir, txt_folder):
        self.data_dir = data_dir
        self.metadata = parse_txt_metadata(txt_folder)

        # Create binary indicators for murmur locations
        for loc in ['AV', 'PV', 'TV', 'MV']:
            self.metadata[loc] = self.metadata['murmur_locations'].apply(
                lambda x: int(loc in x) if isinstance(x, list) else 0)

        # Map murmur status to numerical values
        self.murmur_map = {"Present": 0, "Absent": 1, "Unknown": 2}
        self.metadata['murmur_encoded'] = self.metadata['murmur'].map(self.murmur_map)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        wav_path = os.path.join(self.data_dir, row['wav_path'])

        try:
            waveform, fs = torchaudio.load(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return None

        signal = waveform[0].numpy()
        signal = butter_lowpass_filter(signal)
        signal = signal[::4]  # Downsample
        signal = normalize_min_max(signal[:5000])  # Truncate/pad to 5000 samples

        spectrogram = extract_spectrogram(signal)
        spectrogram = normalize_min_max(spectrogram)

        pcg_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        spec_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        # Prepare metadata tensor
        meta = torch.tensor([
            0 if row['sex'] == 'Male' else 1,
            row['height'] if not pd.isna(row['height']) else 0,
            row['weight'] if not pd.isna(row['weight']) else 0,
            int(row['pregnancy']),
            row['AV'],
            row['PV'],
            row['TV'],
            row['MV']
        ], dtype=torch.float32)

        murmur_label = torch.tensor(row['murmur_encoded'], dtype=torch.long)
        outcome_label = torch.tensor(int(row['outcome']), dtype=torch.float32).unsqueeze(0)

        return pcg_tensor, spec_tensor, meta, outcome_label, murmur_label

def collate_fn(batch):
    # Filter out Nones
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Handle empty batch
    return torch.utils.data.default_collate(batch)
