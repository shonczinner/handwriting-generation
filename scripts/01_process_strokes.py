import pandas as pd

def process_strokes(raw_data, train_pct):
    total_len = len(raw_data)
    train_end = int(train_pct * total_len)

    # Split data
    train_data = raw_data[:train_end]

    # Compute normalization statistics from training data
    mu_dx = train_data['delta_x'].mean().item()
    sd_dx = train_data['delta_x'].std(ddof=0).item()
    mu_dy = train_data['delta_y'].mean().item()
    sd_dy = train_data['delta_y'].std(ddof=0).item()

    # Normalize data
    def normalize(df):
        df = df.copy()
        df['delta_x'] = (df['delta_x'] - mu_dx) / sd_dx
        df['delta_y'] = (df['delta_y'] - mu_dy) / sd_dy
        return df

    processed_data = normalize(raw_data)

    # Create stats DataFrame for saving
    stats_df = pd.DataFrame({
        'stat': ['mu_dx', 'sd_dx', 'mu_dy', 'sd_dy'],
        'value': [mu_dx, sd_dx, mu_dy, sd_dy]
    })

    return processed_data, stats_df

if __name__ == "__main__":
    from constants import (
        RAW_STROKES_PATH,
        PROCESSED_STROKES_PATH,
        PROCESSED_STROKES_STATS_PATH
    )
    from utils.config import Config

    config = Config()

    # Load the raw data
    raw_data = pd.read_csv(RAW_STROKES_PATH)

    # Process and normalize strokes
    processed_data, stats_df = process_strokes(raw_data, train_pct=config.train_pct)

    # Save processed data
    processed_data.to_csv(PROCESSED_STROKES_PATH, index=False)

    # Save normalization statistics
    stats_df.to_csv(PROCESSED_STROKES_STATS_PATH, index=False)

    print("Data processing complete. Files and normalization stats saved.")
