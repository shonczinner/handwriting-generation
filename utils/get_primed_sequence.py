import tkinter as tk
import time
import threading
import pandas as pd

class HandwritingRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recorder")

        self.canvas = tk.Canvas(root, bg="white", width=800, height=600)
        self.canvas.pack()

        self.drawing = False
        self.current_stroke = []
        self.strokes = []

        self.last_time = 0
        self.lock = threading.Lock()

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_motion)

    def start_stroke(self, event):
        self.drawing = True
        self.current_stroke = []
        self.last_time = time.time()
        threading.Thread(target=self.record_position, args=(event.x, event.y), daemon=True).start()

    def end_stroke(self, event):
        with self.lock:
            self.drawing = False
            if self.current_stroke:
                self.strokes.append(self.current_stroke)
                self.current_stroke = []

    def draw_motion(self, event):
        self.canvas.create_oval(event.x, event.y, event.x+1, event.y+1, fill="black")

    def record_position(self, x, y):
        while self.drawing:
            current_time = time.time()
            if (current_time - self.last_time) >= 0.015:  # every 20 ms
                mouse_x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
                mouse_y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()

                with self.lock:
                    self.current_stroke.append((mouse_x, mouse_y, current_time))
                self.last_time = current_time
            time.sleep(0.01)  # reduce CPU usage

    def get_stroke_data(self):
        return self.strokes
    
    def get_stroke_df(self):
        """
        Returns strokes as a DataFrame with columns: stroke_id, x, y, time
        """
        rows = []
        prev_x, prev_y, _ = self.strokes[0][0]
        for stroke_id, stroke in enumerate(self.strokes):
            for i, point in enumerate(stroke):
                x, y, _ = point
                last_point = 1 if i == len(stroke) - 1 else 0
                rows.append((x-prev_x, y-prev_y, last_point))
                prev_x=x
                prev_y=y
        return pd.DataFrame(rows, columns=["delta_x", "delta_y", "lift_point"])


import torch

def df_to_tensor(df, normalize_stats=None):
    """
    Converts a DataFrame with delta_x, delta_y, and lift_point into a normalized PyTorch tensor.

    Parameters:
        df (pd.DataFrame): Must contain ['delta_x', 'delta_y', 'lift_point']
        normalize_stats (str or dict, optional): 
            - Path to CSV with mu/sd values (columns: stat,value)
            - OR a dict with keys: mu_dx, sd_dx, mu_dy, sd_dy
            - OR None (default): normalize based on df's own mean/std

    Returns:
        torch.Tensor: Shape (1, seq_len, 3)
    """
    data = df.copy()

    if normalize_stats is not None:
        if isinstance(normalize_stats, str):  # Path to CSV
            stats = pd.read_csv(normalize_stats).set_index('stat')['value']
            mu_dx = stats['mu_dx']
            sd_dx = stats['sd_dx']
            mu_dy = stats['mu_dy']
            sd_dy = stats['sd_dy']
        elif isinstance(normalize_stats, dict):  # Provided directly
            mu_dx = normalize_stats['mu_dx']
            sd_dx = normalize_stats['sd_dx']
            mu_dy = normalize_stats['mu_dy']
            sd_dy = normalize_stats['sd_dy']
        else:
            raise ValueError("normalize_stats must be a path or dict")
    else:
        # Compute mean and std from the df itself
        mu_dx = data['delta_x'].mean()
        sd_dx = data['delta_x'].std(ddof=0)  # population std deviation
        mu_dy = data['delta_y'].mean()
        sd_dy = data['delta_y'].std(ddof=0)

    data['delta_x'] = (data['delta_x'] - mu_dx) / sd_dx
    data['delta_y'] = (data['delta_y'] - mu_dy) / sd_dy

    tensor = torch.tensor(data[['delta_x', 'delta_y', 'lift_point']].values, dtype=torch.float32)
    return tensor.unsqueeze(0)  # Add batch dimension: (1, seq_len, 3)


if __name__ == "__main__":
    from constants import PROCESSED_STROKES_STATS_PATH
    from utils.display_strokes import plot_tensor
    root = tk.Tk()
    app = HandwritingRecorder(root)
    root.mainloop()

    df = app.get_stroke_df()
    print(df.head())

    tensor = df_to_tensor(df,normalize_stats=PROCESSED_STROKES_STATS_PATH)
    print(tensor[0,:5])

    plot_tensor(tensor,denormalize_stats=PROCESSED_STROKES_STATS_PATH)
