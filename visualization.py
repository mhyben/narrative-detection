import os
import webbrowser

import pandas as pd
import numpy as np
from os.path import join

from pandas import DataFrame

import pipeline

def visualize(embedding_2d: np.ndarray, retrieved_df: DataFrame, output_dir='./results', open_browser=False):
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

        # By default, do not open the browser as frontend will handle display
        if open_browser:
            try:
                output_path_abs = os.path.abspath(output_path)
                print(f"Opening visualization in default browser: {output_path_abs}")
                webbrowser.open('file://' + output_path_abs)
            except Exception as e:
                print(f"Failed to open browser: {str(e)}")
                print("Please open the file manually: " + output_path)

    except ImportError as e:
        print(f"Warning: datamapplot library not found: {str(e)}. Creating a simple HTML visualization.")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data for further analysis
        np.save(join(output_dir, "narrative_map.npy"), embedding_2d)
        retrieved_df.to_csv(join(output_dir, "narrative_analysis.csv"), index=False)
        
        # Create a simple HTML visualization
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Narrative Map</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .container { max-width: 800px; margin: 0 auto; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Narrative Map</h1>
                <p>This is a simple visualization of the narrative map. The datamapplot library is not installed.</p>
                <h2>Cluster Summary</h2>
                <table>
                    <tr>
                        <th>Cluster</th>
                        <th>Summary</th>
                        <th>Count</th>
                    </tr>
        """
        
        # Add cluster summaries to the HTML
        cluster_counts = retrieved_df.groupby(['micro_cluster', 'cluster_summary']).size().reset_index(name='count')
        for _, row in cluster_counts.iterrows():
            cluster = str(row['micro_cluster'])
            summary = str(row['cluster_summary']) if row['cluster_summary'] else "No summary available"
            count = str(row['count'])
            html_content += f"""
                    <tr>
                        <td>{cluster}</td>
                        <td>{summary}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_path = join(output_dir, "narrative_map.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Simple HTML visualization saved to {output_path}")

def regenerate(out_dir='./results'):
    embedding_2d = np.load(join(out_dir, "narrative_map.npy"))
    retrieved_df = pd.read_csv(join(out_dir, "narrative_analysis.csv"))
    retrieved_df['cluster_summary'] = retrieved_df['cluster_summary'].str.replace(r'[()\[\]{}!@#$%^&*]', '', regex=True)

    pipeline.visualize(embedding_2d, retrieved_df, out_dir, open_browser=True)

