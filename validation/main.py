import json
import os
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from itertools import cycle
from matplotlib.ticker import PercentFormatter, AutoMinorLocator, MultipleLocator
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp
from scipy.stats import t as t_dist, chi2
from tabulate import tabulate
import torch
import torch.nn as nn
from scipy.stats import chisquare
from scipy.stats import wilcoxon, ttest_rel

ACADEMIC_COLOR = "#4C72B0"

try:
    from validation.trust_predict import TrustTransfer_
    from validation.trust_transfer_model import TrustTransfer
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from trust_predict import TrustTransfer_
    from trust_transfer_model import TrustTransfer


class technical_validation():
    def __init__(self):
        self.read_data()
        # LLM data
        llm_sim_data = []

    def read_data(self):
        self.llm_sim_data = self.load_json("../data/LLM_Simulated_data.json")
        self.vr_based_data = self.load_json("../data/VR_Based_data.json")
        self.real_world_data = self.load_json("../data/Real_World_data.json")

    def load_json(self, filename):
        """Load JSON using a path relative to this file when not absolute."""
        path = Path(filename)
        if not path.is_absolute():
            path = Path(__file__).parent / filename
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)

    def save_json(self, filename, data):
        """Save JSON to a path relative to this file when not absolute; ensure directory exists."""
        path = Path(filename)
        if not path.is_absolute():
            path = Path(__file__).parent / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f)

    def time_by_five(self, numbers):
        return [x * 5 for x in numbers]

    def plot_trust_mean_extrema_table(self, list1, list2, list3, list_names=None):
        sci_palette = ["#4E79A7", "#A0CBE8", "#F28E2B"]

        if list_names is None:
            list_names = ["LLM-Simulated", "VR-Based Human-in-the-loop", "Real-World"]

        records = []
        for lst, modality in zip([list1, list2, list3], list_names):
            for p in lst:
                for v in p["trust"]:
                    records.append({'modality': modality, 'trust': v})
        df = pd.DataFrame(records)

        # ===== Descriptive statistics table =====
        stats = df.groupby('modality')['trust'].agg(
            N='count',
            Mean='mean',
            Std='std',
            Median='median',
            Q1=lambda x: np.percentile(x, 25),
            Q3=lambda x: np.percentile(x, 75),
            Min='min',
            Max='max'
        ).reset_index()

        print("\nTrust Score Descriptive Statistics by Modality:")
        print(stats.to_string(index=False, float_format="%.3f"))
        # Return statistics table
        return stats

    

    
    
    def plot_trust_distribution(self, list1, list2, list3, modality_names=None):
        """
        Bin trust values into 10 intervals and compute percentages per interval, grouped by three modalities.
        Also draw a binned CDF and run KS/Chi-square tests on the binned distributions.

        Args:
            list1, list2, list3: Three lists containing participant data.
            modality_names: Names of the three modality lists.

        Returns:
            DataFrame: distribution statistics
            matplotlib.figure: the plotted figure
            dict: test results based on the 10-bin distribution
        """
        # Set scientific plotting style
        sns.set(style="whitegrid", font_scale=1.2)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        # Default modality names
        if modality_names is None:
            modality_names = ['Modality 1', 'Modality 2', 'Modality 3']

        # Scientific color palette
        sci_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # Collect trust data and normalize to [0,1]
        all_data = []
        for list_idx, (participant_list, modality) in enumerate(zip([list1, list2, list3], modality_names)):
            for participant in participant_list:
                trust = participant['trust']
                print(len(trust))
                # Normalize trust values to [0,1]
                if max(trust) != min(trust):
                    trust_normalized = [(t - min(trust)) / (max(trust) - min(trust)) for t in trust]
                    print(max(trust), min(trust), trust_normalized)
                else:
                    trust_normalized = [0.5 for _ in trust]  # All values identical
                for t in trust_normalized:
                    all_data.append({
                        'Modality': modality,
                        'Normalized Trust': t,
                        'Group': list_idx
                    })

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Create 10 trust bins
        bins = np.linspace(0, 1, 11)  # 0-0.1, 0.1-0.2, ..., 0.9-1.0
        labels = [f'{i * 10:.0f}-{(i + 1) * 10:.0f}%' for i in range(10)]

        # Compute percentage per bin
        df['Trust Bin'] = pd.cut(df['Normalized Trust'], bins=bins, labels=labels, include_lowest=True)
        dist_df = df.groupby(['Modality', 'Trust Bin']).size().unstack(fill_value=0)
        dist_df = dist_df.div(dist_df.sum(axis=1), axis=0) * 100  # Convert to percentage

        # Create composite figure (bar chart + CDF)
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0])  # Left: bar chart
        ax2 = fig.add_subplot(gs[1])  # Right: CDF

        # Draw bar chart
        bar_width = 0.25
        x_pos = np.arange(len(labels))
        for i, modality in enumerate(modality_names):
            ax1.bar(x_pos + i * bar_width, dist_df.loc[modality],
                    width=bar_width, color=sci_colors[i],
                    edgecolor='white', linewidth=0.5,
                    label=modality)

        # Beautify bar chart
        ax1.set_title('Percentage Distribution of Normalized Trust Across Modalities',
                      fontsize=14, pad=20)
        ax1.set_xlabel('Trust Range', fontsize=12)
        ax1.set_ylabel('Percentage (%)', fontsize=12)
        ax1.set_xticks(x_pos + bar_width)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.yaxis.set_major_formatter(PercentFormatter(100))
        ax1.legend(title='Modality', frameon=True)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Add data labels
        for i, modality in enumerate(modality_names):
            for j, bin_label in enumerate(labels):
                value = dist_df.loc[modality, bin_label]
                if value > 5:  # Show only values > 5%
                    ax1.text(x_pos[j] + i * bar_width, value + 1, f'{value:.1f}%',
                             ha='center', va='bottom', fontsize=9)

        # ================== CDF over 10 bins and tests ==================
        # Compute bin centers (0.05, 0.15,..., 0.95)
        bin_centers = np.linspace(0.05, 0.95, 10)

        # Compute CDF (cumulative probability)
        cdf_data = {
            modality: np.cumsum(dist_df.loc[modality].values) / 100
            for modality in modality_names
        }

        # Plot CDF curves (10 points)
        for i, modality in enumerate(modality_names):
            ax2.plot(bin_centers, cdf_data[modality],
                     color=sci_colors[i], marker='o',
                     linewidth=2, label=f'{modality} CDF')

        # Beautify CDF plot
        ax2.set_title('Binned Cumulative Distribution (10 intervals)', fontsize=14)
        ax2.set_xlabel('Trust Bin Centers', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(frameon=True)

        # ================== KS test and Chi-square test ==================
        test_results = {}

        for i in range(len(modality_names)):
            for j in range(i + 1, len(modality_names)):
                mod1, mod2 = modality_names[i], modality_names[j]

                # Get probability distributions (%)
                prob1 = dist_df.loc[mod1].values / 100
                prob2 = dist_df.loc[mod2].values / 100

                # KS test (based on cumulative probability)
                ks_stat, ks_p = ks_2samp(cdf_data[mod1], cdf_data[mod2])

                # Chi-square test (using counts)
                chi_stat, chi_p = chisquare(
                    f_obs=dist_df.loc[mod2].values,
                    f_exp=dist_df.loc[mod1].values
                )
                print(dist_df.loc[mod2].values)
                print(dist_df.loc[mod1].values)

                test_results[f'{mod1} vs {mod2}'] = {
                    'KS': {
                        'statistic': ks_stat,
                        'p_value': ks_p,
                        'interpretation': 'Different (p < 0.05)' if ks_p < 0.05 else 'Similar (p ≥ 0.05)'
                    },
                    'Chi-square': {
                        'statistic': chi_stat,
                        'p_value': chi_p,
                        'interpretation': 'Different (p < 0.05)' if chi_p < 0.05 else 'Similar (p ≥ 0.05)'
                    }
                }

        # Print test results
        print("\nDistribution Comparison Results (based on 10-bin intervals):")
        for comp, res in test_results.items():
            print(f"\n{comp}:")
            print(
                f"  KS Test: stat={res['KS']['statistic']:.3f}, p={res['KS']['p_value']:.4f} ({res['KS']['interpretation']})")
            print(
                f"  Chi-square Test: stat={res['Chi-square']['statistic']:.3f}, p={res['Chi-square']['p_value']:.4f} ({res['Chi-square']['interpretation']})")

        plt.tight_layout()
        plt.show()
        return dist_df, fig, test_results

    def plot_trust_distribution_scientific(self, list1, list2, list3, modality_names=None):
        """
        Binned frequency distribution for trust scores across three modalities,
        with Kolmogorov–Smirnov tests on the raw trust distributions.
        Returns: proportion table, figure, KS results dict.
        """
        sns.set(style="whitegrid", font_scale=1.2, palette="pastel")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.edgecolor'] = '#222222'

        sci_colors = ['#76b7b2', '#fdc086', '#8da0cb']

        if modality_names is None:
            modality_names = ['Modality 1', 'Modality 2', 'Modality 3']

        # 收集信任数据
        all_data = []
        trust_dict = {}
        for list_idx, (participant_list, modality) in enumerate(zip([list1, list2, list3], modality_names)):
            trust_flat = []
            for participant in participant_list:
                trust_flat.extend(participant['trust'])
                for t in participant['trust']:
                    all_data.append({
                        'Modality': modality,
                        'Trust': t
                    })
            trust_dict[modality] = trust_flat
        df = pd.DataFrame(all_data)

        # Bins (0.0–0.1, ..., 0.9–1.0)
        bins = np.linspace(0, 1, 11)
        labels = [f'{bins[i]:.1f}–{bins[i + 1]:.1f}' for i in range(10)]
        df['Trust Bin'] = pd.cut(df['Trust'], bins=bins, labels=labels, include_lowest=True)
        freq_df = df.groupby(['Modality', 'Trust Bin']).size().unstack(fill_value=0)
        prop_df = freq_df.div(freq_df.sum(axis=1), axis=0)

        # Draw frequency distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.23
        x_pos = np.arange(len(labels))

        for i, modality in enumerate(modality_names):
            ax.bar(x_pos + i * bar_width, prop_df.loc[modality],
                   width=bar_width, color=sci_colors[i],
                   edgecolor='#333333', linewidth=1,
                   label=modality, alpha=0.93)

        ax.set_title('Trust Distribution Across Modalities (Binned, 0–1)', fontsize=15, pad=16)
        ax.set_xlabel('Trust Interval', fontsize=13)
        ax.set_ylabel('Proportion', fontsize=13)
        ax.set_xticks(x_pos + bar_width)
        ax.set_xticklabels(labels, rotation=0, ha='center')
        ax.legend(title='Modality', frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Data labels (only show when proportion > 8%)
        for i, modality in enumerate(modality_names):
            for j, bin_label in enumerate(labels):
                proportion = prop_df.loc[modality, bin_label]
                if proportion > 0.08:
                    ax.text(x_pos[j] + i * bar_width, proportion + 0.01, f'{proportion:.2f}',
                            ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

        # ==== KS test (based on raw trust scores) ====
        ks_results = {}
        for i in range(len(modality_names)):
            for j in range(i + 1, len(modality_names)):
                m1, m2 = modality_names[i], modality_names[j]
                ks_stat, ks_p = ks_2samp(trust_dict[m1], trust_dict[m2])
                ks_results[f'{m1} vs {m2}'] = {
                    'ks_stat': ks_stat,
                    'p_value': ks_p,
                    'interpretation': 'Similar (p ≥ 0.05)' if ks_p >= 0.05 else 'Different (p < 0.05)'
                }
        print(ks_results)

        return prop_df, fig, ks_results

    def plot_trust_change_tcds(self, trust_data, bins=10, title=None, drop_high_bin_outliers=False):
        """
        Draw a TCDS-style vertical bar chart of absolute trust change per previous trust bin.

        Args:
        - trust_data: 2D array [n_sequences, n_timesteps] or list of dicts with 'trust'
        - bins: number of equal-width bins (default 10)
        - title: optional title
        """
        # Standardize input to a 2D numpy array (assumed value range [0,1])
        if isinstance(trust_data, list) and len(trust_data) > 0 and isinstance(trust_data[0], dict) and 'trust' in trust_data[0]:
            trust_array = np.array([p['trust'] for p in trust_data], dtype=float)
        else:
            trust_array = np.asarray(trust_data, dtype=float)

        if trust_array.ndim != 2 or trust_array.shape[1] < 2:
            raise ValueError("trust_data must be 2D with shape [n_sequences, n_timesteps>=2]")

            # Compute absolute change between adjacent time steps
        changes = np.abs(np.diff(trust_array, axis=1))

        # Bin edges based on data range (equal-width bins)
        prev_vals = trust_array[:, :-1]
        data_min = float(np.nanmin(prev_vals))
        data_max = float(np.nanmax(prev_vals))
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
            data_min, data_max = 0.0, 1.0
        bin_edges = np.linspace(data_min, data_max, bins + 1)

        avg_changes, std_changes, counts = [], [], []
        bin_labels = []

        for i in range(bins):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            if i < bins - 1:
                mask = (prev_vals >= lower) & (prev_vals < upper)
            else:
                mask = (prev_vals >= lower) & (prev_vals <= upper)

            vals = changes[mask]
            # Optionally drop outliers by IQR for each bin
            if vals.size > 0 and drop_high_bin_outliers:
                v = vals.astype(float)
                if v.size >= 4:
                    q1, q3 = np.percentile(v, [25, 75])
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bd = q1 - 1.5 * iqr
                        upper_bd = q3 + 1.5 * iqr
                        v = v[(v >= lower_bd) & (v <= upper_bd)]
                # If all filtered out, fall back to original values to avoid zeros
                vals_use = v if v.size > 0 else vals
            else:
                vals_use = vals

            if vals_use.size > 0:
                avg_changes.append(float(vals_use.mean()))
                std_changes.append(float(vals_use.std(ddof=1)) if vals_use.size > 1 else 0.0)
                counts.append(int(vals_use.size))
            else:
                avg_changes.append(0.0)
                std_changes.append(0.0)
                counts.append(0)

            # Fixed 2-decimal bin labels
            bin_labels.append(f"{lower:.2f}-{upper:.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=300)
        x = np.arange(bins)

        bar_color = "#4C72B0"
        bars = ax.bar(x, avg_changes, width=0.7, edgecolor="black", linewidth=0.8, color=bar_color)
        ax.errorbar(x, avg_changes, yerr=std_changes, fmt="none", elinewidth=0.9, capsize=3, color="black")

        # Annotate sample size N
        for xi, h, n in zip(x, avg_changes, counts):
            ax.text(xi, h + max(ax.get_ylim()[1] * 0.01, 0.02), f"N={n}", ha="center", va="bottom", fontsize=8)

        # Axes and grid
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=35, ha="right")
        ax.set_xlim(-0.5, bins - 0.5)

        y_max = max(max(avg_changes) + (max(std_changes) if std_changes else 0), 1e-6)
        ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(max(y_max / 5, 0.05)))

        ax.set_xlabel("Trust Value Range", fontsize=10)
        ax.set_ylabel("Average Absolute Trust Change", fontsize=10)
        ax.set_title(title or "Average Absolute Trust Change by Trust Range", fontsize=12, pad=10)

        ax.grid(which="major", axis="y", linestyle="--", linewidth=0.6)
        ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5)

        fig.tight_layout()
        plt.show()

    def plot_trust_inc_dec_abs_means_three(self, list1, list2, list3, modality_names=None):
        """
        For each dataset, plot a two-bar figure showing mean |Δtrust| when
        trust increases vs when it decreases.
        """
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']

        datasets = [list1, list2, list3]
        inc_color = ACADEMIC_COLOR
        dec_color = "#DC5A32"

        for data_list, name in zip(datasets, modality_names):
            # Collect all adjacent differences
            deltas = []
            for p in data_list:
                t = np.asarray(p.get('trust', []), dtype=float)
                if t.size >= 2:
                    deltas.append(np.diff(t))
            if len(deltas) == 0:
                inc_vals = np.array([])
                dec_vals = np.array([])
            else:
                deltas = np.concatenate(deltas, axis=0)
                inc_vals = deltas[deltas > 0]
                dec_vals = deltas[deltas < 0]

            inc_abs_mean = float(np.mean(np.abs(inc_vals))) if inc_vals.size > 0 else 0.0
            dec_abs_mean = float(np.mean(np.abs(dec_vals))) if dec_vals.size > 0 else 0.0

            # Plot
            fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
            x = np.arange(2)
            heights = [inc_abs_mean, dec_abs_mean]
            colors = [inc_color, dec_color]
            bars = ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

            # Text annotations (value and N)
            labels = ["Increase", "Decrease"]
            counts = [int(inc_vals.size), int(dec_vals.size)]
            for xi, h, n in zip(x, heights, counts):
                ax.text(xi, h + max(0.01, 0.02 * (ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1)),
                        f"{h:.3f}\nN={n}", ha="center", va="bottom", fontsize=9)

            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_ylabel("Average |ΔTrust|", fontsize=10)
            ax.set_title(f"Average |ΔTrust| when Increase vs Decrease\n({name})", fontsize=12, pad=10)

            y_max = max(heights + [0.02])
            ax.set_ylim(0, y_max * 1.18)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(which="major", axis="y", linestyle="--", linewidth=0.6, color="0.6")
            ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, color="0.85")
            for s in ax.spines.values():
                s.set_linewidth(0.8); s.set_color("black")

            fig.tight_layout()
            plt.show()

    def plot_trust_fluctuation_early_vs_late_three(self, list1, list2, list3,
                                                   early_range=(1, 3), late_range=(8, 10),
                                                   modality_names=None):
        """
        For each dataset, compare early vs late window trust fluctuations (mean |Δtrust|).
        - early_range/late_range are 1-based inclusive task indices, e.g., (1,3) and (8,10)
        - Each subject's fluctuation is the mean absolute adjacent difference within the window
        - Show mean ± SEM and print paired-test results where applicable
        """
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']

        def window_mean_abs_delta(trust_series, a, b):
            # 1-based inclusive indices a..b; transitions are (a-1->a) ... (b-1->b)
            t = np.asarray(trust_series, dtype=float)
            n = t.size
            if n < 2 or a < 1 or b < a:
                return np.nan
            lo = max(a, 2)  # first valid transition ends at least at index 2 (1-based)
            hi = min(b, n - 0)  # last point index cannot exceed n
            # transition indices end at k in [lo..hi], start at k-1 (1-based)
            if hi - lo + 1 <= 0:
                return np.nan
            deltas = t[lo - 1:hi] - t[lo - 2:hi - 1]
            if deltas.size == 0:
                return np.nan
            return float(np.mean(np.abs(deltas)))

        datasets = [list1, list2, list3]
        color_early = ACADEMIC_COLOR
        color_late = "#A0CBE8"

        for data_list, name in zip(datasets, modality_names):
            early_vals, late_vals = [], []
            for p in data_list:
                e = window_mean_abs_delta(p.get('trust', []), early_range[0], early_range[1])
                l = window_mean_abs_delta(p.get('trust', []), late_range[0], late_range[1])
                if np.isfinite(e):
                    early_vals.append(e)
                else:
                    early_vals.append(np.nan)
                if np.isfinite(l):
                    late_vals.append(l)
                else:
                    late_vals.append(np.nan)

            early_vals = np.array(early_vals, float)
            late_vals = np.array(late_vals, float)
            mask = np.isfinite(early_vals) & np.isfinite(late_vals)
            early_use = early_vals[mask]
            late_use = late_vals[mask]

            # Summary statistics
            def mean_sem(x):
                if x.size == 0:
                    return 0.0, 0.0
                m = float(np.mean(x))
                se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
                return m, se
            m_e, se_e = mean_sem(early_use)
            m_l, se_l = mean_sem(late_use)

            # Paired test (Wilcoxon preferred; fallback to paired t; else insufficient)
            test_note = ""
            if early_use.size >= 3 and late_use.size >= 3:
                from scipy.stats import wilcoxon, ttest_rel
                try:
                    stat, p = wilcoxon(early_use, late_use)
                    test_note = f"Wilcoxon p={p:.4f}"
                except ValueError:
                    t_stat, p = ttest_rel(early_use, late_use, nan_policy='omit')
                    test_note = f"Paired t p={p:.4f}"
            else:
                test_note = f"Insufficient pairs (N={early_use.size})"

            # Plot
            fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=300)
            x = np.arange(2)
            heights = [m_e, m_l]
            errors = [se_e, se_l]
            colors = [color_early, color_late]
            ax.bar(x, heights, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.8, width=0.62)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Tasks {early_range[0]}–{early_range[1]}", f"Tasks {late_range[0]}–{late_range[1]}"])
            ax.set_ylabel("Mean |ΔTrust| per subject (SEM)", fontsize=10)
            ax.set_title(f"Trust Fluctuation: Early vs Late ({name})\n{test_note}", fontsize=12, pad=10)

            # Annotate N
            counts = [int(early_use.size), int(late_use.size)]
            for xi, h, n in zip(x, heights, counts):
                ax.text(xi, h + max(0.01, 0.02 * (ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1)),
                        f"N={n}", ha="center", va="bottom", fontsize=9)

            y_max = max(heights + [0.02])
            ax.set_ylim(0, y_max * 1.18)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(which="major", axis="y", linestyle="--", linewidth=0.6, color="0.6")
            ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, color="0.85")
            for s in ax.spines.values():
                s.set_linewidth(0.8); s.set_color("black")

            fig.tight_layout()
            plt.show()

    def plot_trust_change_abs_by_reward_three(self, list1, list2, list3, modality_names=None):
        """
        For three datasets, draw one figure per dataset: compare average |Δtrust| when reward > 0 vs ≤ 0.
        - Show only the mean per bar; remove SEM and p-value.
        - Standardize modality display names.
        """
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']

        # Normalize display names
        def _norm(name: str) -> str:
            disp = name.replace('_', ' ').replace('-', ' ').strip()
            key = disp.lower()
            mapping = {
                'llm-simulated': 'LLM Simulated',
                'llm simulated': 'LLM Simulated',
                'vr-based human-in-the-loop': 'VR Based Human-in-the-loop',
                'vr based human in the loop': 'VR Based Human-in-the-loop',
                'vr-based': 'VR Based',
                'vr based': 'VR Based',
                'real-world': 'Real World',
                'real world': 'Real World'
            }
            return mapping.get(key, disp.title())

        display_names = [_norm(n) for n in modality_names]

        datasets = [list1, list2, list3]
        color_pos = ACADEMIC_COLOR
        color_non = "#A7B1B7"

        for data_list, name in zip(datasets, display_names):
            pos_vals, nonpos_vals = [], []

            for p in data_list:
                trust = np.asarray(p.get('trust', []), dtype=float)
                result = p.get('task_result', None)
                if result is None:
                    continue
                result = np.asarray(result, dtype=float)
                min_len = min(trust.size - 1, result.size)
                if min_len <= 0:
                    continue
                dt = trust[1:min_len + 1] - trust[:min_len]
                r  = result[:min_len]
                pos_vals.extend(np.abs(dt[r > 0]).tolist())
                nonpos_vals.extend(np.abs(dt[r <= 0]).tolist())

            use_label_pos = "Success"
            use_label_non = "Failure"
            use_pos = np.array(pos_vals, float)
            use_non = np.array(nonpos_vals, float)

            def mean_sem(x):
                if x.size == 0:
                    return 0.0, 0.0
                m = float(np.mean(x))
                se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
                return m, se

            m_pos, se_pos = mean_sem(use_pos)
            m_non, se_non = mean_sem(use_non)

            # Plot
            fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=300)
            x = np.arange(2)
            heights = [m_pos, m_non]
            colors = [color_pos, color_non]
            ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.8, width=0.62)
            ax.set_xticks(x)
            ax.set_xticklabels([use_label_pos, use_label_non])
            ax.set_ylabel("Mean |ΔTrust|(MSE)", fontsize=10)
            ax.set_title(f"Average |ΔTrust| by Task Result ({name})", fontsize=12, pad=10)

            # Set a tighter top margin before annotating
            y_max = max(heights + [0.02])
            ax.set_ylim(0, y_max * 1.08)

            # Annotate exact mean value above bars (closer to the bar)
            offset = max(0.002, 0.008 * y_max)
            for xi, h in zip(x, heights):
                ax.text(xi, h + offset, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(which="major", axis="y", linestyle="--", linewidth=0.6, color="0.6")
            ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, color="0.85")
            for s in ax.spines.values():
                s.set_linewidth(0.8); s.set_color("black")

            fig.tight_layout()
            plt.show()

    def analyze_trust_dependence(self, trust_2d, max_lag=12, max_p=6, train_ratio=0.7, dataset_name=None):
        def _tcds_axes(ax):
            ax.grid(which="major", axis="both", linestyle="--", linewidth=0.6, color="0.6")
            ax.grid(which="minor", axis="both", linestyle=":", linewidth=0.5, color="0.85")
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            for s in ax.spines.values():
                s.set_linewidth(0.8); s.set_color("black")

        def _safe_max_lag(n, max_lag):
            return max(1, min(max_lag, n-2))

        def _safe_max_p(n, max_p):
            return max(1, min(max_p, n-2))

        def _acf_numpy(x, max_lag):
            x = np.asarray(x, dtype=float)
            n  = x.size
            L  = _safe_max_lag(n, max_lag)
            x  = x - x.mean()
            den = float(np.dot(x, x))
            acf = np.zeros(L+1, float)
            acf[0] = 1.0 if den > 0 else 0.0
            if den == 0:
                return acf
            for k in range(1, L+1):
                acf[k] = np.dot(x[:n-k], x[k:]) / den
            return acf

        def _pacf_yw(x, max_lag):
            x = np.asarray(x, dtype=float)
            n  = x.size
            L  = _safe_max_lag(n, max_lag)
            x  = x - x.mean()
            r  = np.array([np.dot(x[:n-k], x[k:]) / n for k in range(L+1)], float)
            pacf = np.zeros(L+1, float); pacf[0] = 1.0
            if r[0] == 0:
                return pacf
            phi = np.zeros((L+1, L+1), float)
            sig2= np.zeros(L+1, float)
            phi[1,1] = r[1]/r[0]; sig2[1] = r[0]*(1-phi[1,1]**2); pacf[1] = phi[1,1]
            for k in range(2, L+1):
                num = r[k] - np.sum(phi[k-1,1:k]*r[1:k][::-1])
                den = sig2[k-1] if sig2[k-1] != 0 else 1e-12
                phi[k,k] = num/den
                for j in range(1, k):
                    phi[k,j] = phi[k-1,j] - phi[k,k]*phi[k-1,k-j]
                sig2[k] = sig2[k-1]*(1-phi[k,k]**2)
                pacf[k] = phi[k,k]
            return pacf

        def _fit_ar_least_squares(x, p):
            x = np.asarray(x, float); n = x.size
            if n <= p + 1:
                return np.nan, np.full(p, np.nan), np.array([]), np.nan, np.nan, np.nan
            y = x[p:]
            X = np.column_stack([np.ones(n-p)] + [x[p-k:-k] for k in range(1, p+1)])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                return np.nan, np.full(p, np.nan), np.array([]), np.nan, np.nan, np.nan
            resid = y - X @ beta
            rss   = float(np.dot(resid, resid))
            k     = p + 1
            n_eff = len(y)
            sigma2= rss/n_eff if n_eff>0 else np.nan
            aic   = n_eff*(np.log(sigma2) if sigma2>0 else 0.0) + 2*k
            bic   = n_eff*(np.log(sigma2) if sigma2>0 else 0.0) + k*np.log(max(n_eff,2))
            return float(beta[0]), beta[1:], resid, rss, float(aic), float(bic)

        def _rolling_one_step_mse(x, p, train_ratio=0.7):
            x = np.asarray(x, float); n = x.size
            p = _safe_max_p(n, p)
            t0= max(p + 2, int(n*train_ratio))
            preds, obs = [], []
            for t in range(t0, n):
                c, a, resid, rss, aic, bic = _fit_ar_least_squares(x[:t], p)
                if np.isnan(aic): continue
                xhat = c + np.sum(a * x[t-p:t][::-1])
                preds.append(xhat); obs.append(x[t])
            if not preds: return np.nan
            preds, obs = np.array(preds), np.array(obs)
            return float(np.mean((preds-obs)**2))

        trust = np.asarray(trust_2d, float)
        if trust.ndim != 2: raise ValueError("trust_2d 必须是二维 [n_subjects, n_timesteps]")
        n_subj, n_t = trust.shape

        L = _safe_max_lag(n_t, max_lag)
        P = _safe_max_p(n_t, max_p)
        if L < max_lag or P < max_p:
            print(f"[Info] 序列长度={n_t}，自动设置 max_lag={L}、max_p={P}。")

        # 1) Group ACF
        acfs = [ _acf_numpy(row, L)[1:] for row in trust ]
        acfs = np.vstack(acfs)
        lags = np.arange(1, L+1)
        acf_mean = np.nanmean(acfs, axis=0)
        acf_std  = np.nanstd(acfs, axis=0, ddof=1)

        fig1, ax1 = plt.subplots(figsize=(6.8, 3.2), dpi=300)
        ax1.plot(lags, acf_mean, marker="o", lw=1.6, color=ACADEMIC_COLOR)
        ax1.fill_between(lags, acf_mean-acf_std, acf_mean+acf_std, color="0.85")
        ax1.hlines(0, 1, L, colors="black", linewidth=0.8)
        ax1.set_xlabel("Lag Order"); ax1.set_ylabel("Autocorrelation (mean ± SD)")
        ax1.set_title("Group-Level Autocorrelation Function" + (f" ({dataset_name})" if dataset_name else ""))
        ax1.set_xticks(lags)
        _tcds_axes(ax1)
        fig1.tight_layout(); plt.show()

        acf_peak_k = int(np.nanargmax(np.abs(acf_mean))) + 1
        acf_peak_val = float(acf_mean[acf_peak_k-1])

        # 2) Group PACF
        pacfs = [ _pacf_yw(row, L)[1:] for row in trust ]
        pacfs = np.vstack(pacfs)
        pacf_mean = np.nanmean(pacfs, axis=0)
        pacf_std  = np.nanstd(pacfs, axis=0, ddof=1)

        fig2, ax2 = plt.subplots(figsize=(6.8, 3.2), dpi=300)
        ax2.plot(lags, pacf_mean, marker="o", lw=1.6, color=ACADEMIC_COLOR)
        ax2.fill_between(lags, pacf_mean-pacf_std, pacf_mean+pacf_std, color="0.85")
        ax2.hlines(0, 1, L, colors="black", linewidth=0.8)
        ax2.set_xlabel("Lag Order"); ax2.set_ylabel("Partial Autocorrelation (mean ± SD)")
        ax2.set_title("Group-Level Partial Autocorrelation Function" + (f" ({dataset_name})" if dataset_name else ""))
        ax2.set_xticks(lags)
        _tcds_axes(ax2)
        fig2.tight_layout(); plt.show()

        pacf_dom_k = int(np.nanargmax(np.abs(pacf_mean))) + 1
        pacf_dom_val = float(pacf_mean[pacf_dom_k-1])

        # Keep only ACF/PACF conclusions
        acf_sentence = f"[ACF] Peak lag (group mean): k={acf_peak_k}, value={acf_peak_val:.2f}."
        pacf_sentence = f"[PACF] Dominant lag (group mean): k={pacf_dom_k}, value={pacf_dom_val:.2f}."
        print("=== Conclusions (Data-Driven) ===" + (f" [{dataset_name}]" if dataset_name else ""))
        print(acf_sentence)
        print(pacf_sentence)

    def ecological_dynamic_validity(self, list1, list2, list3, modality_names=None, sample_n=8, random_state=42,
                                    dynamic_thr=0.10):
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']
        data_lists = [list1, list2, list3]

        # Summary table
        records = []
        violin_df_list = []

        plt.figure(figsize=(15, 6))
        def _norm(name: str) -> str:
            disp = name.replace('_', ' ').replace('-', ' ').strip()
            key = disp.lower()
            mapping = {
                'llm-simulated': 'LLM Simulated',
                'llm simulated': 'LLM Simulated',
                'vr-based human-in-the-loop': 'VR Based Human-in-the-loop',
                'vr based human in the loop': 'VR Based Human-in-the-loop',
                'vr-based': 'VR Based',
                'vr based': 'VR Based',
                'real-world': 'Real World',
                'real world': 'Real World'
            }
            return mapping.get(key, disp.title())
        for i, (lst, modality) in enumerate(zip(data_lists, [_norm(m) for m in modality_names])):
            trust_all = np.array([p['trust'] for p in lst])
            n_person = trust_all.shape[0]

            # 统计
            n_static = np.sum(np.std(trust_all, axis=1) < 1e-6)
            n_dynamic = np.sum(np.std(trust_all, axis=1) >= dynamic_thr)
            records.append({
                'Modality': modality,
                'N Participants': n_person,
                'Static Count': n_static,
                'Static Ratio (%)': 100 * n_static / n_person,
                f'Dynamic Count (std≥{dynamic_thr:.2f})': n_dynamic,
                'Dynamic Ratio (%)': 100 * n_dynamic / n_person,
                'Mean Trust Change (abs)': np.mean(np.abs(np.diff(trust_all, axis=1))),
            })

            # (1) Sampled participant trajectories (left panel)
            np.random.seed(random_state)
            sel_idx = np.random.choice(n_person, min(sample_n, n_person), replace=False)
            plt.subplot(2, 3, i + 1)
            for j in sel_idx:
                plt.plot(range(1, trust_all.shape[1] + 1), trust_all[j], marker='o', alpha=0.8,
                         label=f'P{j}' if i == 0 else None)
            plt.ylim(0, 1)
            plt.title(f"{modality}\nSampled Trajectories")
            plt.xlabel("Task Index")
            plt.ylabel("Trust")
            plt.grid(True, linestyle='--', alpha=0.5)

            # (2) Violin plot data wrangling
            violin_df = pd.DataFrame(trust_all)
            violin_df = violin_df.melt(var_name='Task Index', value_name='Trust')
            violin_df['Modality'] = modality
            violin_df_list.append(violin_df)

        # (3) Violin plot (right panel): all modalities aligned by task
        violin_df_all = pd.concat(violin_df_list, ignore_index=True)
        plt.subplot(2, 1, 2)
        sns.violinplot(x="Task Index", y="Trust", hue="Modality", data=violin_df_all, split=True, inner="quartile",
                       linewidth=1.2)
        plt.ylim(0, 1)
        plt.title("Trust Distribution at Each Task Step (Violin Plot)")
        plt.xlabel("Task Index")
        plt.ylabel("Trust")
        plt.legend(loc='best', frameon=True)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # (4) Summary table
        stats_df = pd.DataFrame(records)
        print("\nEcological and Dynamic Validity - Trust Dynamic Table:")
        print(stats_df)

        return stats_df

    def prepare_data(self, data_list):
        """Prepare training data and return tensors X and y."""
        X, y = [], []
        for data in data_list:
            min_len = min(len(data['trust']) - 1,
                          len(data['state']),
                          len(data['robot_decision_making']),
                          len(data['human_decision_making']))

            for i in range(min_len):
                features = [
                    data['trust'][i],
                    data["task_result"][i] / 5,
                    data['state'][i],
                    data['robot_decision_making'][i],
                    data['human_decision_making'][i],
                ]
                X.append(features)
                y.append(data['trust'][i + 1])

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def train_model(self, data_list):
        """Train model via TrustTransfer.train_trust_transfer and return the trained model."""
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Prepare data
        X, y = self.prepare_data(data_list)

        # Initialize model
        model = TrustTransfer(input_size=5, hidden_size=20, output_size=1)

        # Train
        model.train_trust_transfer(X, y, num_epochs=10000, learning_rate=0.001)

        # Compute training RMSE
        with torch.no_grad():
            preds = model(X)
            rmse = torch.sqrt(nn.functional.mse_loss(preds.squeeze(), y))
        print(rmse.item())

        return model

    def test_model(self, model, test_data_list):
        """Test model via TrustTransfer.test_model and return RMSE."""
        X_test, y_test = self.prepare_data(test_data_list)
        _, rmse, _ = model.test_model(X_test, y_test)
        print(rmse)
        return rmse.item()

    def run_experiment(self,list1, list2, list3, input_size=5, hidden_size=20, output_size=1, num_epochs=600):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        results = []
        x_llm,y_llm  = self.prepare_data(list1)
        x_vr_train,y_vr_train = self.prepare_data(list2[:20])
        x_vr_test, y_vr_test = self.prepare_data(list2[20:])
        x_real_train, y_real_train = self.prepare_data(list3[:20])
        x_real_test, y_real_test = self.prepare_data(list3[20:])

        # 1. VR only
        model1 = TrustTransfer_(input_size, hidden_size, output_size)
        model1.fit(x_vr_train, y_vr_train, num_epochs=num_epochs)
        _, rmse_vr_only, _ = model1.test_model(x_vr_test, y_vr_test)
        results.append(["VR only", rmse_vr_only, None])

        # 2. LLM pretrain + VR finetune
        model2 = TrustTransfer_(input_size, hidden_size, output_size)
        model2.fit(x_llm, y_llm, num_epochs=num_epochs)
        model2.fit(x_vr_train, y_vr_train, num_epochs=num_epochs//2)
        _, rmse_llm_vr, _ = model2.test_model(x_vr_test, y_vr_test)
        results.append(["LLM pretrain + VR finetune", rmse_llm_vr, None])

        # 3. Real only
        model3 = TrustTransfer_(input_size, hidden_size, output_size)
        model3.fit(x_real_train, y_real_train, num_epochs=num_epochs)
        _, rmse_real_only, _ = model3.test_model(x_real_test, y_real_test)
        results.append(["Real only", None, rmse_real_only])

        # 4. VR pretrain + Real finetune
        model4 = TrustTransfer_(input_size, hidden_size, output_size)
        model4.fit(x_vr_train, y_vr_train, num_epochs=num_epochs)
        model4.fit(x_vr_train, y_vr_train, num_epochs=num_epochs//2)
        _, rmse_vr_real, _ = model4.test_model(x_real_test, y_real_test)
        results.append(["VR pretrain + Real finetune", None, rmse_vr_real])

        # 5. LLM pretrain + VR finetune + Real finetune
        model5 = TrustTransfer_(input_size, hidden_size, output_size)
        model5.fit(x_llm, y_llm, num_epochs=num_epochs)
        model5.fit(x_vr_train, y_vr_train, num_epochs=num_epochs//2)
        model5.fit(x_vr_train, y_vr_train, num_epochs=num_epochs//2)
        _, rmse_llm_vr_real, _ = model5.test_model(x_real_test, y_real_test)
        results.append(["LLM pretrain + VR finetune + Real finetune", None, rmse_llm_vr_real])

        import pandas as pd
        df = pd.DataFrame(results, columns=["Training Pipeline", "VR Test RMSE", "Real Test RMSE"])
        print(df)
        return df

    def plot_dynamic_change_ratio(self, list1, list2, list3, modality_names=None):
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']
        import numpy as np
        import matplotlib.pyplot as plt

        data_lists = [list1, list2, list3]
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        fig_hist, axs_hist = plt.subplots(1, 3, figsize=(18, 4))

        for i, (lst, modality) in enumerate(zip(data_lists, modality_names)):
            trust_all = np.array([p['trust'] for p in lst])
            delta = np.abs(np.diff(trust_all, axis=1))
            change = (delta > 1e-8).mean(axis=0)
            nochange = (delta <= 1e-8).mean(axis=0)

            # 折线图
            axs[i].plot(range(1, len(change) + 1), change, '-o', label='Changed', color='tab:blue')
            axs[i].plot(range(1, len(nochange) + 1), nochange, '-o', label='Unchanged', color='tab:green')
            axs[i].set_title(modality)
            axs[i].set_xlabel('Task Step')
            axs[i].set_ylabel('Proportion')
            axs[i].set_ylim(0, 1)
            axs[i].legend()
            axs[i].grid(alpha=0.4)

            # Histogram (distribution of per-step change/nochange proportions)
            axs_hist[i].hist(change, bins=8, alpha=0.7, label='Changed', color='tab:blue', edgecolor='k')
            axs_hist[i].hist(nochange, bins=8, alpha=0.7, label='Unchanged', color='tab:green', edgecolor='k')
            axs_hist[i].set_title(f"{modality}\nProportion Distribution")
            axs_hist[i].set_xlabel('Proportion')
            axs_hist[i].set_ylabel('Frequency')
            axs_hist[i].legend()
            axs_hist[i].grid(alpha=0.4)

        plt.tight_layout()
        fig.tight_layout()
        fig_hist.tight_layout()
        plt.show()

    def plot_dynamic_change_ratio_v2(self, list1, list2, list3, modality_names=None):
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']
        import numpy as np
        import matplotlib.pyplot as plt

        data_lists = [list1, list2, list3]
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        fig_bar, axs_bar = plt.subplots(1, 3, figsize=(18, 4))

        for i, (lst, modality) in enumerate(zip(data_lists, modality_names)):
            trust_all = np.array([p['trust'] for p in lst])
            delta = np.abs(np.diff(trust_all, axis=1))
            change = (delta > 1e-8).mean(axis=0)
            nochange = (delta <= 1e-8).mean(axis=0)
            steps = np.arange(1, len(change) + 1)

            # Line plot
            axs[i].plot(steps, change, '-o', label='Changed', color='tab:blue')
            axs[i].plot(steps, nochange, '-o', label='Unchanged', color='tab:green')
            axs[i].set_title(modality)
            axs[i].set_xlabel('Task Step')
            axs[i].set_ylabel('Proportion')
            axs[i].set_ylim(0, 1)
            axs[i].legend()
            axs[i].grid(alpha=0.4)

            # Per-step bar chart
            width = 0.35
            axs_bar[i].bar(steps - width / 2, change, width=width, label='Changed', color='tab:blue', alpha=0.8)
            axs_bar[i].bar(steps + width / 2, nochange, width=width, label='Unchanged', color='tab:green', alpha=0.8)
            axs_bar[i].set_title(modality)
            axs_bar[i].set_xlabel('Task Step')
            axs_bar[i].set_ylabel('Proportion')
            axs_bar[i].set_ylim(0, 1)
            axs_bar[i].legend()
            axs_bar[i].grid(axis='y', alpha=0.4)

        fig.tight_layout()
        fig_bar.tight_layout()
        plt.show()

    def plot_dynamic_change_ratio_bar_scientific(self, list1, list2, list3, modality_names=None):
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']

        color_changed = "#346BA2"  
        color_unchanged = "#A7B1B7"  
        width = 0.38

        data_lists = [list1, list2, list3]
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        # normalize display names
        def _norm(name: str) -> str:
            disp = name.replace('_', ' ').replace('-', ' ').strip()
            key = disp.lower()
            mapping = {
                'llm-simulated': 'LLM Simulated',
                'llm simulated': 'LLM Simulated',
                'vr-based human-in-the-loop': 'VR Based Human-in-the-loop',
                'vr based human in the loop': 'VR Based Human-in-the-loop',
                'vr-based': 'VR Based',
                'vr based': 'VR Based',
                'real-world': 'Real World',
                'real world': 'Real World'
            }
            return mapping.get(key, disp.title())
        display_names = [_norm(n) for n in modality_names]
        for i, (lst, modality) in enumerate(zip(data_lists, display_names)):
            trust_all = np.array([p['trust'] for p in lst])
            delta = np.abs(np.diff(trust_all, axis=1))
            change = (delta > 1e-8).mean(axis=0)
            nochange = (delta <= 1e-8).mean(axis=0)
            steps = np.arange(1, len(change) + 1)

            axs[i].bar(steps - width / 2, change, width=width, label='Changed', color=color_changed, alpha=0.95)
            axs[i].bar(steps + width / 2, nochange, width=width, label='Unchanged', color=color_unchanged, alpha=0.95)
            axs[i].set_title(modality)
            axs[i].set_xlabel('Task Step')
            axs[i].set_ylabel('Proportion')
            axs[i].set_ylim(0, 1)
            axs[i].set_xticks(steps)
            axs[i].legend(frameon=False, fontsize=10)
            axs[i].grid(axis='y', alpha=0.25, linestyle='--')

        fig.tight_layout()
        plt.show()

    def plot_sampled_trust_curves_fixed_range(self, list1, list2, list3, modality_names=None, sample_n=5,
                                              random_state=5):
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']
        import numpy as np
        import matplotlib.pyplot as plt

        data_lists = [list1, list2, list3]
        plt.figure(figsize=(18, 4.5))
        # normalize display names
        def _norm(name: str) -> str:
            disp = name.replace('_', ' ').replace('-', ' ').strip()
            key = disp.lower()
            mapping = {
                'llm-simulated': 'LLM Simulated',
                'llm simulated': 'LLM Simulated',
                'vr-based human-in-the-loop': 'VR Based Human-in-the-loop',
                'vr based human in the loop': 'VR Based Human-in-the-loop',
                'vr-based': 'VR Based',
                'vr based': 'VR Based',
                'real-world': 'Real World',
                'real world': 'Real World'
            }
            return mapping.get(key, disp.title())
        display_names = [_norm(n) for n in modality_names]
        for i, (lst, modality) in enumerate(zip(data_lists, display_names)):
            trust_all = np.array([p['trust'] for p in lst])
            n_person, n_step = trust_all.shape
            idx_good = np.where((trust_all[:, 0] >= 0.2) & (trust_all[:, 0] <= 0.7))[0]
            np.random.seed(random_state)
            if len(idx_good) >= sample_n:
                idx = np.random.choice(idx_good, sample_n, replace=False)
            else:
                # 如不够就全取，然后补齐
                rest = [ix for ix in range(n_person) if ix not in idx_good]
                rest_needed = sample_n - len(idx_good)
                idx = np.concatenate([idx_good, np.random.choice(rest, rest_needed, replace=False)])
            plt.subplot(1, 3, i + 1)
            for j in idx:
                plt.plot(range(1, n_step + 1), trust_all[j], marker='o', alpha=0.85)
            plt.ylim(0, 1)
            plt.title(f"{modality}\nTrust Trajectories")
            plt.xlabel("Task Index")
            plt.ylabel("Trust")
            plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_trust_direction_factor_heatmaps(self,list1, list2, list3, modality_names=None):
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']
        data_lists = [list1, list2, list3]
        direction_types = ["Increase", "Decrease", "Unchanged"]
        factor_names = [
            ("Robot-State Agreement", lambda robot_dec, state: robot_dec == state, "Consistent"),
            ("Task Success", lambda result: result > 0, "Success"),
            ("Threat State", lambda state: state == 1, "No Threat")
        ]

        plt.figure(figsize=(18, 4.5))
        for idx_mod, (lst, modality) in enumerate(zip(data_lists, modality_names)):
            # Compute positive ratios per factor for each trust-change direction
            result_matrix = np.zeros((3, 3))
            records = []
            for participant in lst:
                trust = participant['trust']
                robot_dec = participant['robot_decision_making']
                state = participant['state']
                result = participant.get('task_result', None)
                min_len = min(len(trust) - 1, len(robot_dec), len(state))
                for i in range(min_len):
                    delta_trust = trust[i + 1] - trust[i]
                    if delta_trust > 1e-8:
                        direction = "Increase"
                    elif delta_trust < -1e-8:
                        direction = "Decrease"
                    else:
                        direction = "Unchanged"
                    agreement_val = robot_dec[i] == state[i]
                    task_succ_val = (result[i] > 0) if result is not None else None
                    no_threat_val = (state[i] == 1)
                    records.append({
                        "Direction": direction,
                        "Robot-State Agreement": agreement_val,
                        "Task Success": task_succ_val,
                        "Threat State": no_threat_val
                    })
            df = pd.DataFrame(records)
            for i, (fac, _, _) in enumerate(factor_names):
                for j, direction in enumerate(direction_types):
                    sub = df[df["Direction"] == direction]
                    if len(sub) == 0:
                        result_matrix[i, j] = np.nan
                    else:
                        # Positive ratio
                        fac_val = sub[fac]
                        # Drop missing
                        fac_val = fac_val[~pd.isnull(fac_val)]
                        if len(fac_val) == 0:
                            result_matrix[i, j] = np.nan
                        else:
                            result_matrix[i, j] = np.mean(fac_val)

            ax = plt.subplot(1, 3, idx_mod + 1)
            sns.heatmap(
                result_matrix,
                annot=True, fmt=".2f", cmap="YlGnBu", cbar=idx_mod == 2, vmin=0, vmax=1,
                xticklabels=direction_types,
                yticklabels=[x[0] for x in factor_names],
                ax=ax, square=True
            )
            ax.set_title(modality, fontsize=14)
            ax.set_xlabel("Trust Change Direction")
            ax.set_ylabel("Task Factor")
        plt.suptitle("Association between Trust Change Direction and Task Factors", fontsize=16, y=1.05)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    def analyze_trust_change_and_factors_v2(self, list1, list2, list3, modality_names=None, figsize=(16, 10)):
        if modality_names is None:
            modality_names = ['LLM-Simulated', 'VR-Based Human-in-the-loop', 'Real-World']
        data_lists = [list1, list2, list3]
        df_all = []

        for lst, modality in zip(data_lists, modality_names):
            for participant in lst:
                trust = participant['trust']
                robot_dec = participant['robot_decision_making']
                human_dec = participant['human_decision_making']
                state = participant['state']
                result = participant.get('task_result', None)
                min_len = min(len(trust) - 1, len(state), len(robot_dec), len(human_dec))
                for i in range(min_len):
                    delta_trust = trust[i + 1] - trust[i]
                    direction = "Increase" if delta_trust > 1e-8 else (
                        "Decrease" if delta_trust < -1e-8 else "Unchanged")
                    agreement = "Consistent" if robot_dec[i] == state[i] else "Inconsistent"
                    task_success = None
                    if result is not None:
                        task_success = "Success" if result[i] > 0 else "Failure"
                    adopted = "Adopted" if human_dec[i] == robot_dec[i] else "Not Adopted"
                    df_all.append({
                        "Modality": modality,
                        "DeltaTrust": delta_trust,
                        "Direction": direction,
                        "Robot-State Agreement": agreement,
                        "Task Success": task_success,
                        "Adoption": adopted
                    })

        df = pd.DataFrame(df_all)

        # Normalize modality display names for scientific style
        def _norm(name: str) -> str:
            disp = name.replace('_', ' ').replace('-', ' ').strip()
            key = disp.lower()
            mapping = {
                'llm-simulated': 'LLM Simulated',
                'llm simulated': 'LLM Simulated',
                'vr-based human-in-the-loop': 'VR Based Human-in-the-loop',
                'vr based human in the loop': 'VR Based Human-in-the-loop',
                'vr-based': 'VR Based',
                'vr based': 'VR Based',
                'real-world': 'Real World',
                'real world': 'Real World'
            }
            return mapping.get(key, disp.title())

        df['ModalityDisplay'] = df['Modality'].apply(_norm)

        plt.figure(figsize=(14, 4))
        ax1 = plt.subplot(1, 3, 1)
        sns.boxplot(x="ModalityDisplay", y="DeltaTrust", hue="Robot-State Agreement", data=df, showfliers=False, ax=ax1)
        ax1.set_title('ΔTrust: Robot-State Agreement')
        ax1.set_xlabel("Modality")
        ax1.set_ylabel("ΔTrust")

        ax2 = plt.subplot(1, 3, 2)
        sns.boxplot(x="ModalityDisplay", y="DeltaTrust", hue="Task Success", data=df, showfliers=False, ax=ax2)
        ax2.set_title('ΔTrust: Task Success/Failure')
        ax2.set_xlabel("Modality")
        ax2.set_ylabel("ΔTrust")

        ax3 = plt.subplot(1, 3, 3)
        sns.boxplot(x="ModalityDisplay", y="DeltaTrust", hue="Adoption", data=df, showfliers=False, ax=ax3)
        ax3.set_title('ΔTrust: Adoption of Robot Advice')
        ax3.set_xlabel("Modality")
        ax3.set_ylabel("ΔTrust")

        plt.tight_layout()
        plt.show()
        return df

if __name__ == '__main__':
    technical_validation = technical_validation()
    # Three-modality trust table (mean, variance, extrema)
    technical_validation.plot_trust_mean_extrema_table(technical_validation.llm_sim_data,
                                                       technical_validation.vr_based_data,
                                                       technical_validation.real_world_data,
                                                       ["llm-simulated", "vr_based", "real_world"])
    
    # Randomly sampled trajectories + count unchanged participants

    technical_validation.plot_dynamic_change_ratio_bar_scientific(technical_validation.llm_sim_data,
                                                                  technical_validation.vr_based_data,
                                                                  technical_validation.real_world_data,
                                                                  ["llm-simulated", "vr_based", "real_world"])
    technical_validation.plot_sampled_trust_curves_fixed_range(technical_validation.llm_sim_data,
                                                               technical_validation.vr_based_data,
                                                               technical_validation.real_world_data,
                                                               ["llm-simulated", "vr_based", "real_world"])

    # Dependence analysis for each dataset
    technical_validation.analyze_trust_dependence(
        np.array([p['trust'] for p in technical_validation.llm_sim_data], float),
        max_lag=12, max_p=6, train_ratio=0.7, dataset_name="llm-simulated"
    )
    technical_validation.analyze_trust_dependence(
        np.array([p['trust'] for p in technical_validation.vr_based_data], float),
        max_lag=12, max_p=6, train_ratio=0.7, dataset_name="vr_based"
    )
    technical_validation.analyze_trust_dependence(
        np.array([p['trust'] for p in technical_validation.real_world_data], float),
        max_lag=12, max_p=6, train_ratio=0.7, dataset_name="real_world"
    )

    # Average |Δtrust| by reward sign (+5 vs -5), one figure per modality
    technical_validation.plot_trust_change_abs_by_reward_three(
        technical_validation.llm_sim_data,
        technical_validation.vr_based_data,
        technical_validation.real_world_data,
        modality_names=["llm-simulated", "vr_based", "real_world"]
    )

    technical_validation.run_experiment(technical_validation.llm_sim_data,
                                        technical_validation.vr_based_data,
                                        technical_validation.real_world_data)

    technical_validation.analyze_trust_change_and_factors_v2(technical_validation.llm_sim_data,
                                                            technical_validation.vr_based_data,
                                                            technical_validation.real_world_data,
                                                          ["llm-simulated", "vr_based", "real_world"])