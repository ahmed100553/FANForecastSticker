###################################
# 1) IMPORTS
###################################
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, PositionalEmbedding, TokenEmbedding, TimeFeatureEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



###################################
# 2) LOADING THE DATA
###################################
def load_data():
    # Adjust paths as needed if they are in a different directory
    train = pd.read_csv("playground-series-s5e1/train.csv")
    test = pd.read_csv("playground-series-s5e1/test.csv")
    submission = pd.read_csv("playground-series-s5e1/sample_submission.csv")
    return train, test, submission

###################################
# 3) BASIC PREPROCESS & IMPUTATION
###################################
def preprocess_data(train: pd.DataFrame):
    # Convert date to datetime
    train['date'] = pd.to_datetime(train['date'])

    # Impute missing num_sold by group means (country, product, store, month)
    train['month'] = train['date'].dt.month
    group_cols = ['country','product','store','month']
    group_means = train.groupby(group_cols)['num_sold'].transform('mean')
    train['num_sold'] = train['num_sold'].fillna(group_means)

    # Fill any remaining NaNs with overall mean
    train['num_sold'].fillna(train['num_sold'].mean(), inplace=True)
    return train

###################################
# 4) AGGREGATE INTO A SINGLE SERIES
###################################
def aggregate_timeseries(train: pd.DataFrame):
    """
    Aggregate all (country,store,product) combos into one daily series
    Return a DataFrame with [date, num_sold] plus time features
    """
    agg_df = train.groupby('date', as_index=False)['num_sold'].sum().sort_values('date')

    # Create 4 time features so freq='h' in TimeFeatureEmbedding matches (month, day, weekday, year_mod)
    agg_df['month']   = agg_df['date'].dt.month
    agg_df['day']     = agg_df['date'].dt.day
    agg_df['weekday'] = agg_df['date'].dt.weekday
    agg_df['year']    = agg_df['date'].dt.year
    agg_df['year_mod'] = agg_df['year'] - 2000

    return agg_df

###################################
# 5) TRAIN/VALID SPLIT BY DATE
###################################
def split_data_by_date(agg_df: pd.DataFrame, date_str="2015-01-01"):
    """
    Splits into train/valid based on date_str boundary
    """
    split_date = pd.to_datetime(date_str)
    train_mask = agg_df['date'] < split_date
    valid_mask = agg_df['date'] >= split_date

    train_data = agg_df[train_mask].reset_index(drop=True)
    valid_data = agg_df[valid_mask].reset_index(drop=True)
    return train_data, valid_data

###################################
# 6) MAKE WINDOWS (SLIDING)
###################################
def make_windows(data, time_feat, input_size=96, pred_size=24):
    """
    data: 1D array of the target
    time_feat: 2D array of shape (N, 4) containing [month, day, weekday, year_mod]
    Return: x_enc, x_mark_enc, x_dec, x_mark_dec, y arrays
    """
    x_enc_list, x_mark_enc_list = [], []
    x_dec_list, x_mark_dec_list = [], []
    y_list = []
    L = len(data)
    for i in range(L - input_size - pred_size):
        x_enc = data[i : i+input_size]
        t_enc = time_feat[i : i+input_size]
        x_dec = data[i+input_size : i+input_size+pred_size]
        t_dec = time_feat[i+input_size : i+input_size+pred_size]
        y    = data[i+input_size : i+input_size+pred_size]

        x_enc_list.append(x_enc)
        x_mark_enc_list.append(t_enc)
        x_dec_list.append(x_dec)
        x_mark_dec_list.append(t_dec)
        y_list.append(y)

    return (
        np.array(x_enc_list),
        np.array(x_mark_enc_list),
        np.array(x_dec_list),
        np.array(x_mark_dec_list),
        np.array(y_list)
    )

###################################
# 7) CUSTOM DATASET
###################################
class TimeSeriesWindowDataset(Dataset):
    def __init__(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y):
        super().__init__()
        self.x_enc = x_enc
        self.x_mark_enc = x_mark_enc
        self.x_dec = x_dec
        self.x_mark_dec = x_mark_dec
        self.y = y
        
    def __len__(self):
        return len(self.x_enc)
    
    def __getitem__(self, idx):
        # Convert to torch and return
        # x_enc, x_dec => (Length, 1)
        return (
            torch.tensor(self.x_enc[idx],      dtype=torch.float32),
            torch.tensor(self.x_mark_enc[idx], dtype=torch.float32),
            torch.tensor(self.x_dec[idx],      dtype=torch.float32),
            torch.tensor(self.x_mark_dec[idx], dtype=torch.float32),
            torch.tensor(self.y[idx],          dtype=torch.float32)
        )

###################################
# 8) EMBEDDING + FAN LAYER + TRANSFORMER
###################################

# --- (B) FAN Layer
class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim//2))
        self.activation = nn.GELU()
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, src):
        # src: [B, L, d_model]
        g = self.activation(self.input_linear_g(src))  # => [B, L, out_dim - out_dim//2]
        p = self.input_linear_p(src)                   # => [B, L, out_dim//4]
        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate*torch.cos(p), gate*torch.sin(p), (1-gate)*g), dim=-1)
        return output

###################################
# 9) FANTimeSeriesModel (Transformer + FANLayer)
###################################
class FANTimeSeriesModel(nn.Module):
    """
    Transformer-based model for time-series forecasting with FAN layer integration.
    """
    def __init__(self, configs):
        super(FANTimeSeriesModel, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # FAN Layer
        self.fan_layer = FANLayer(configs.d_model, configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    exp_setting=configs.exp_setting,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    exp_setting=configs.exp_setting,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        x_enc:      [B, seq_len, enc_in]
        x_mark_enc: [B, seq_len, 4]
        x_dec:      [B, pred_len, dec_in]
        x_mark_dec: [B, pred_len, 4]
        """
        #print(f"x_enc shape: {x_enc.shape}")
        #print(f"x_mark_enc shape: {x_mark_enc.shape}")
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #print(f"enc_out shape after embedding: {enc_out.shape}")

        #print(f"x_dec shape: {x_dec.shape}")
        #print(f"x_mark_dec shape: {x_mark_dec.shape}")
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        #print(f"dec_out shape after embedding: {dec_out.shape}")

        # FAN layer
        enc_out = self.fan_layer(enc_out)
        #print(f"enc_out shape after FAN layer: {enc_out.shape}")

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        #print(f"enc_out shape after encoder: {enc_out.shape}")

        # Decoder
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #print(f"dec_out shape after decoder: {dec_out.shape}")

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

###################################
# 10) CONFIGS
###################################
class Configs:
    # Because we have only one aggregated series, we set enc_in=1 and dec_in=1.
    enc_in = 1
    dec_in = 1
    c_out = 1
    d_model = 64
    embed = 'timeF'
    freq = 'h'        # We'll keep 'h' so we have 4 time features in x_mark
    dropout = 0.1
    e_layers = 2
    d_layers = 1
    d_ff = 256
    n_heads = 8
    factor = 5
    activation = 'gelu'
    output_attention = False
    pred_len = 24
    exp_setting = 0

###################################
# 11) MAIN TRAINING LOGIC
###################################
def main():
    # 1) Load Data
    train, test, submission = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) Preprocess/impute
    train = preprocess_data(train)

    # 3) Aggregate
    agg_df = aggregate_timeseries(train)

    # 4) Split
    train_data, valid_data = split_data_by_date(agg_df, date_str="2015-01-01")

    # 5) Extract arrays
    y_train = train_data['num_sold'].values
    y_valid = valid_data['num_sold'].values

    # ---- SCALE the target arrays ----
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1,1)).ravel()
    y_valid_scaled = scaler.transform(y_valid.reshape(-1,1)).ravel()

    # 6) Time features
    time_feat_train = train_data[['month','day','weekday','year_mod']].values
    time_feat_valid = valid_data[['month','day','weekday','year_mod']].values

    # 7) Build windows (use scaled y)
    input_size = 96
    pred_size = 24
    x_enc_tr, x_mark_enc_tr, x_dec_tr, x_mark_dec_tr, y_tr = make_windows(
        y_train_scaled, time_feat_train, input_size, pred_size
    )
    x_enc_val, x_mark_enc_val, x_dec_val, x_mark_dec_val, y_val = make_windows(
        y_valid_scaled, time_feat_valid, input_size, pred_size
    )

    # 8) Add channel dim
    x_enc_tr = x_enc_tr[..., None]
    x_dec_tr = x_dec_tr[..., None]
    x_enc_val = x_enc_val[..., None]
    x_dec_val = x_dec_val[..., None]

    # 9) Dataset, Dataloader
    train_dataset = TimeSeriesWindowDataset(
        x_enc_tr, x_mark_enc_tr, x_dec_tr, x_mark_dec_tr, y_tr
    )
    val_dataset = TimeSeriesWindowDataset(
        x_enc_val, x_mark_enc_val, x_dec_val, x_mark_dec_val, y_val
    )

    # 10) Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

    # 11) Model, loss, optimizer
    configs = Configs()
    model = FANTimeSeriesModel(configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # We pick a threshold in the *scaled* domain.
    threshold = 0.0  # e.g. class=1 if prediction > 0, else 0

    num_epochs = 500

    # Print header for the table:
    print(
        f"{'Epoch':<6}"
        + f"{'TrainLoss':>12}"
        + f"{'ValLoss':>12}"
        + f"{'TrainAcc':>10}"
        + f"{'TrainPrec':>10}"
        + f"{'TrainRec':>10}"
        + f"{'TrainF1':>9}"
        + f"{'ValAcc':>9}"
        + f"{'ValPrec':>9}"
        + f"{'ValRec':>8}"
        + f"{'ValF1':>8}"
    )

    for epoch in range(num_epochs):
        ############################################
        #  A) Training
        ############################################
        model.train()
        train_loss = 0.0

        # For classification metrics, accumulate all preds/labels
        train_pred_classes = []
        train_true_classes = []

        for batch_data in train_loader:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = [bd.to(device) for bd in batch_data]
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)   # => [B, pred_len, 1]
            y_pred = y_pred.squeeze(-1)  # => [B, pred_len]

            # MSE loss for regression
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(x_enc)

            # ====== Classification metrics ======
            # Flatten [B, pred_len] => [B*pred_len]
            y_pred_flat = y_pred.detach().cpu().numpy().ravel()
            y_true_flat = y_true.detach().cpu().numpy().ravel()

            # Binarize
            pred_class = (y_pred_flat > threshold).astype(int)
            true_class = (y_true_flat > threshold).astype(int)

            train_pred_classes.append(pred_class)
            train_true_classes.append(true_class)

        train_loss /= len(train_loader.dataset)

        # Flatten & compute classification metrics
        train_pred_classes = np.concatenate(train_pred_classes, axis=0)
        train_true_classes = np.concatenate(train_true_classes, axis=0)

        train_acc  = accuracy_score(train_true_classes, train_pred_classes)
        train_prec = precision_score(train_true_classes, train_pred_classes, zero_division=0)
        train_rec  = recall_score(train_true_classes, train_pred_classes, zero_division=0)
        train_f1   = f1_score(train_true_classes, train_pred_classes, zero_division=0)

        ############################################
        #  B) Validation
        ############################################
        model.eval()
        val_loss = 0.0
        val_pred_classes = []
        val_true_classes = []

        with torch.no_grad():
            for batch_data in val_loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = [bd.to(device) for bd in batch_data]
                y_pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                y_pred = y_pred.squeeze(-1)

                loss = criterion(y_pred, y_true)
                val_loss += loss.item() * len(x_enc)

                # Classification
                y_pred_flat = y_pred.cpu().numpy().ravel()
                y_true_flat = y_true.cpu().numpy().ravel()

                pred_class = (y_pred_flat > threshold).astype(int)
                true_class = (y_true_flat > threshold).astype(int)

                val_pred_classes.append(pred_class)
                val_true_classes.append(true_class)

        val_loss /= len(val_loader.dataset)

        val_pred_classes = np.concatenate(val_pred_classes, axis=0)
        val_true_classes = np.concatenate(val_true_classes, axis=0)

        val_acc  = accuracy_score(val_true_classes, val_pred_classes)
        val_prec = precision_score(val_true_classes, val_pred_classes, zero_division=0)
        val_rec  = recall_score(val_true_classes, val_pred_classes, zero_division=0)
        val_f1   = f1_score(val_true_classes, val_pred_classes, zero_division=0)

        ############################################
        # Print in table form (one row per epoch)
        ############################################
        print(
            f"{epoch+1:<6}"
            + f"{train_loss:12.4f}"
            + f"{val_loss:12.4f}"
            + f"{train_acc:10.4f}"
            + f"{train_prec:10.4f}"
            + f"{train_rec:10.4f}"
            + f"{train_f1:9.4f}"
            + f"{val_acc:9.4f}"
            + f"{val_prec:9.4f}"
            + f"{val_rec:8.4f}"
            + f"{val_f1:8.4f}"
        )
    # Save model state_dict
    save_path = "fan_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
