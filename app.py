import numpy as np, pandas as pd, torch, torch.nn as nn, streamlit as st, matplotlib.pyplot as plt

CLASSES = ["N (normal)", "S (supraventricular)", "V (ventricular)", "F (fusion)", "Q (unknown)"]

def conv_block(c_in, c_out, k):
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2))

class ECGNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32, 5), conv_block(32, 64, 5), conv_block(64, 128, 3))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))
    def forward(self, x):
        return self.head(self.pool(self.features(x)))

@st.cache_resource
def load_model():
    m = ECGNet(); m.load_state_dict(torch.load("ecg.pt", map_location="cpu")); m.eval(); return m

st.title("ECG beat classifier")
st.caption("Learning project, not a medical device.")

up = st.sidebar.file_uploader("CSV of beats (187 values per row, optional label as 188th)", type="csv")
data = pd.read_csv(up, header=None).values if up else pd.read_csv("sample_beats.csv", header=None).values

if data.shape[1] > 187:
    pick = st.sidebar.selectbox("Show beats labeled", ["All"] + CLASSES)
    if pick != "All":
        data = data[data[:, 187].astype(int) == CLASSES.index(pick)]

i = st.sidebar.slider("Beat index", 0, len(data) - 1, 0)
beat = data[i, :187].astype(np.float32)
true = int(data[i, 187]) if data.shape[1] > 187 else None

with torch.no_grad():
    p = torch.softmax(load_model()(torch.tensor(beat)[None, None, :]), 1)[0].numpy()

fig, ax = plt.subplots(figsize=(8, 3)); ax.plot(beat); ax.set_xlabel("sample"); ax.set_ylabel("amplitude")
st.pyplot(fig)
st.subheader(f"Prediction: {CLASSES[p.argmax()]} ({p.max():.0%})")
if true is not None:
    st.write(f"Label in file: **{CLASSES[true]}**")
st.bar_chart(pd.DataFrame({"probability": p}, index=CLASSES))
