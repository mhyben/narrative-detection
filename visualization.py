import os
import webbrowser

import pandas as pd
import numpy as np
from os.path import join

from pandas import DataFrame

import pipeline

def visualize(embedding_2d: np.ndarray, retrieved_df: DataFrame, output_dir='./results', open_browser=True):
    """Create interactive visualization of the narrative map."""

    print("Creating interactive visualization...")
    try:
        import datamapplot

        os.makedirs(output_dir, exist_ok=True)
        plot = datamapplot.create_interactive_plot(
            embedding_2d,
            retrieved_df['cluster_summary'].astype('str'),
            hover_text=retrieved_df['text'],
            enable_search=True
        )
        output_path = join(output_dir, "narrative_map.html")
        plot.save(output_path)
        print(f"Interactive visualization saved to {output_path}")

        # Save data for further analysis
        np.save(join(output_dir, "narrative_map.npy"), embedding_2d)
        retrieved_df.to_csv(join(output_dir, "narrative_analysis.csv"), index=False)

        if open_browser:
            try:
                output_path_abs = os.path.abspath(output_path)
                print(f"Opening visualization in default browser: {output_path_abs}")
                webbrowser.open('file://' + output_path_abs)
            except Exception as e:
                print(f"Failed to open browser: {str(e)}")
                print("Please open the file manually: " + output_path)

    except ImportError:
        print("Warning: datamapplot library not found. Saving data without visualization.")
        os.makedirs(output_dir, exist_ok=True)
        np.save(join(output_dir, "narrative_map.npy"), embedding_2d)
        retrieved_df.to_csv(join(output_dir, "narrative_analysis.csv"), index=False)
        print(f"Data saved to {output_dir} directory.")

def regenerate(out_dir='./results'):
    embedding_2d = np.load(join(out_dir, "narrative_map.npy"))
    retrieved_df = pd.read_csv(join(out_dir, "narrative_analysis.csv"))
    retrieved_df['cluster_summary'] = retrieved_df['cluster_summary'].str.replace(r'[()\[\]{}!@#$%^&*]', '', regex=True)

    pipeline.visualize(embedding_2d, retrieved_df, out_dir, open_browser=True)

