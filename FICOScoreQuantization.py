import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
import pandas as pd

###############################################################################
# 1. DP logic with prefix sums (same approach as before)
###############################################################################

def build_prefix_sums_fico(fico):
    n = len(fico)
    prefix_sum = np.zeros(n+1)
    prefix_sqsum = np.zeros(n+1)
    for i in range(1, n+1):
        prefix_sum[i] = prefix_sum[i-1] + fico[i-1]
        prefix_sqsum[i] = prefix_sqsum[i-1] + fico[i-1]**2
    return prefix_sum, prefix_sqsum

def build_prefix_sums_defaults(defaults):
    n = len(defaults)
    prefix_sum_def = np.zeros(n+1)
    for i in range(1, n+1):
        prefix_sum_def[i] = prefix_sum_def[i-1] + defaults[i-1]
    return prefix_sum_def

def cost_mse_prefix(prefix_sum, prefix_sqsum, start, end):
    length = end - start + 1
    sum_slice = prefix_sum[end+1] - prefix_sum[start]
    sqsum_slice = prefix_sqsum[end+1] - prefix_sqsum[start]
    mean = sum_slice / length
    sse = sqsum_slice - 2*mean*sum_slice + length*(mean**2)
    return sse

def cost_loglik_prefix(prefix_sum_def, start, end):
    length = end - start + 1
    d_k = prefix_sum_def[end+1] - prefix_sum_def[start]
    eps = 1e-9
    if length == 0:
        return 0.0
    p_k = d_k / length
    nll = - ( d_k * np.log(p_k + eps) + (length - d_k) * np.log(1 - p_k + eps) )
    return nll

def find_buckets_mse(n, r, prefix_sum, prefix_sqsum):
    dp = np.full((n+1, r+1), np.inf)
    split = np.zeros((n+1, r+1), dtype=int)
    # Base case
    for i in range(1, n+1):
        dp[i, 1] = cost_mse_prefix(prefix_sum, prefix_sqsum, 0, i-1)
        split[i, 1] = 0
    # Fill DP
    for j_ in range(2, r+1):
        for i in range(j_, n+1):
            best_val = np.inf
            best_k = 0
            for k in range(j_-1, i):
                cost_val = dp[k, j_-1] + cost_mse_prefix(prefix_sum, prefix_sqsum, k, i-1)
                if cost_val < best_val:
                    best_val = cost_val
                    best_k = k
            dp[i, j_] = best_val
            split[i, j_] = best_k
    # Reconstruct
    boundaries = []
    curr_i = n
    curr_j = r
    while curr_j > 0:
        k = split[curr_i, curr_j]
        boundaries.append(k)
        curr_i = k
        curr_j -= 1
    boundaries.reverse()
    return dp[n, r], boundaries

def find_buckets_loglik(n, r, prefix_sum_def):
    dp = np.full((n+1, r+1), np.inf)
    split = np.zeros((n+1, r+1), dtype=int)
    # Base case
    for i in range(1, n+1):
        dp[i, 1] = cost_loglik_prefix(prefix_sum_def, 0, i-1)
        split[i, 1] = 0
    # Fill DP
    for j_ in range(2, r+1):
        for i in range(j_, n+1):
            best_val = np.inf
            best_k = 0
            for k in range(j_-1, i):
                cost_val = dp[k, j_-1] + cost_loglik_prefix(prefix_sum_def, k, i-1)
                if cost_val < best_val:
                    best_val = cost_val
                    best_k = k
            dp[i, j_] = best_val
            split[i, j_] = best_k
    # Reconstruct
    boundaries = []
    curr_i = n
    curr_j = r
    while curr_j > 0:
        k = split[curr_i, curr_j]
        boundaries.append(k)
        curr_i = k
        curr_j -= 1
    boundaries.reverse()
    return dp[n, r], boundaries

###############################################################################
# 2. TKINTER GUI
###############################################################################

class FICOQuantizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FICO Quantization | by MJ Yuan")

        # A frame for buttons & info
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, pady=10)

        # "Run" button
        self.run_button = tk.Button(top_frame, text="Run DP", command=self.on_run)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # A label for progress messages
        self.progress_label = tk.Label(top_frame, text="Waiting to run...")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # A text box for output
        self.output_text = tk.Text(root, width=125, height=20)
        self.output_text.pack(side=tk.TOP, padx=10, pady=10)

        # We'll store data in memory once loaded
        self.fico_sorted = None
        self.default_sorted = None
        self.n = 0
        self.r = 5  # example number of buckets

    def on_run(self):
        """
        Start a background thread to run the DP code, so GUI remains responsive.
        """
        self.run_button.config(state=tk.DISABLED)  # disable button
        self.progress_label.config(text="Running, please wait...")
        self.output_text.delete("1.0", tk.END)

        # Start a thread
        thread = threading.Thread(target=self.run_quantization_task)
        thread.start()

    def run_quantization_task(self):
        """
        Loads data, runs the DP, measures time, and updates GUI with results.
        Runs in a separate thread to avoid blocking the mainloop.
        """
        start_time = time.time()
        try:
            # 1) Load CSV & prepare data
            csv_file = "Task 3 and 4_Loan_Data.csv"
            df = pd.read_csv(csv_file)
            df = df[['fico_score', 'default']].dropna()

            fico_scores = df['fico_score'].values.astype(float)
            defaults = df['default'].values.astype(int)

            sort_idx = np.argsort(fico_scores)
            self.fico_sorted = fico_scores[sort_idx]
            self.default_sorted = defaults[sort_idx]
            self.n = len(self.fico_sorted)

            # 2) Build prefix sums
            prefix_sum, prefix_sqsum = build_prefix_sums_fico(self.fico_sorted)
            prefix_sum_def = build_prefix_sums_defaults(self.default_sorted)

            # 3) Run MSE-based DP
            mse_val, mse_boundaries = find_buckets_mse(self.n, self.r, prefix_sum, prefix_sqsum)
            mse_cut_points = [self.fico_sorted[idx] for idx in mse_boundaries if idx < self.n]

            # 4) Run Log-likelihood-based DP
            nll_val, ll_boundaries = find_buckets_loglik(self.n, self.r, prefix_sum_def)
            ll_cut_points = [self.fico_sorted[idx] for idx in ll_boundaries if idx < self.n]

            # 5) Format results
            end_time = time.time()
            elapsed = end_time - start_time

            results_text = (
                "=== MSE-based Bucketing ===\n"
                f"Min SSE: {mse_val}\n"
                f"Boundaries: {mse_boundaries}\n"
                f"Cut points (FICO): {mse_cut_points}\n\n"
                "=== Log-likelihood-based Bucketing ===\n"
                f"Min NLL: {nll_val}\n"
                f"Boundaries: {ll_boundaries}\n"
                f"Cut points (FICO): {ll_cut_points}\n\n"
                f"Time taken: {elapsed:.2f} seconds\n"
            )
        except Exception as e:
            results_text = "Error encountered:\n" + str(e)

        # Update GUI in the main thread via .after()
        self.root.after(0, self.on_task_complete, results_text)

    def on_task_complete(self, results_text):
        """
        Called (via .after()) when the background thread finishes.
        """
        self.output_text.insert(tk.END, results_text)
        self.progress_label.config(text="Done.")
        self.run_button.config(state=tk.NORMAL)

###############################################################################
# 3. MAIN "LAUNCH" CODE
###############################################################################
if __name__ == "__main__":
    root = tk.Tk()
    app = FICOQuantizationGUI(root)
    root.mainloop()
