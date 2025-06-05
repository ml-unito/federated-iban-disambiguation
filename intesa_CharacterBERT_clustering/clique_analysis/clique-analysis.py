from enum import Enum
import pandas as pd
import typer as typer
import networkx as nx
import matplotlib.pyplot as plt

app = typer.Typer()


class OutputFormat(str, Enum):
    NX = "nx"
    DOT = "dot"


def generate_clique_analysis_nx(df, iban):
    ibandata = df[df["iban"] == iban]
    ibandata = ibandata.drop_duplicates(subset=["name1", "name2"])

    predicted = ibandata[ibandata["predicted"] == 0]
    actual = ibandata[ibandata["label.1"] == 0]

    # Create graphs
    G_predicted = nx.Graph()
    G_actual = nx.Graph()

    # Add nodes and edges for predicted
    for _, row in predicted.iterrows():
        G_predicted.add_node(f"{row['name1']}_", label=row['name1'])
        G_predicted.add_node(f"{row['name2']}_", label=row['name2'])
        G_predicted.add_edge(f"{row['name1']}_", f"{row['name2']}_")

    # Add nodes and edges for actual
    for _, row in actual.iterrows():
        G_actual.add_node(row['name1'], label=row['name1'])
        G_actual.add_node(row['name2'], label=row['name2'])
        G_actual.add_edge(row['name1'], row['name2'])

    # Plot graphs side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plt.suptitle(f"Clique Analysis for IBAN: {iban}")

    # Predicted graph
    pos_pred = nx.spring_layout(G_predicted, seed=42)
    nx.draw(G_predicted, pos_pred, ax=axes[0], with_labels=True, labels=nx.get_node_attributes(G_predicted, 'label'), node_shape='s', node_color='skyblue')
    axes[0].set_title("Predicted")

    # Actual graph
    pos_actual = nx.spring_layout(G_actual, seed=42)
    nx.draw(G_actual, pos_actual, ax=axes[1], with_labels=True, labels=nx.get_node_attributes(G_actual, 'label'), node_shape='s', node_color='lightgreen')
    axes[1].set_title("Actual")

    plt.show()


def generate_clique_analysis_dot(df, iban):
    ibandata = df[df["iban"] == iban]
    # remove duplicated rows based on name1 and name2 fields
    ibandata = ibandata.drop_duplicates(subset=["name1", "name2"])

    predicted = ibandata[ibandata["predicted"] == 0]
    actual = ibandata[ibandata["label.1"] == 0]

    nodes = set(ibandata["name1"]).union(set(ibandata["name2"]))

    with open(f"output/{iban}_clique_analysis.dot", "w") as f:
        f.write("graph G {\n")
        f.write("rankdir=UD;\n")
        f.write(f"label = \"{ iban }\";\n")

        # Predicted graph
        f.write("subgraph cluster_predicted {\n")
        f.write("label = \"Predicted\";\n")

        for node in nodes:
            f.write(f'"{node}_" [label="{node}", shape=box];\n')
        for _, row in predicted.iterrows():
            f.write(f'"{row["name1"]}_" -- "{row["name2"]}_";\n')
        f.write("}\n")

        # Actual graph
        f.write("subgraph cluster_actual {\n")
        f.write("label = \"Actual\";\n")

        for node in nodes:
            f.write(f'"{node}" [label="{node}", shape=box];\n')

        for _, row in actual.iterrows():
            f.write(f'"{row["name1"]}" -- "{row["name2"]}";\n')
        f.write("}\n")

        f.write("}\n")  # End of the graph


@app.command()
def run(iban: str, fname: str = "data/couple_prediction_df.csv", 
        output_format: OutputFormat = "dot"):
    """
    Generate a clique analysis dot file for the given IBAN.
    
    Args:
        fname: The CSV file containing the data.
        iban: The IBAN to analyze.
    """
    df = pd.read_csv(fname)

    if output_format == "nx":
        generate_clique_analysis_nx(df, iban)
    elif output_format == "dot":
        generate_clique_analysis_dot(df, iban)
    else:
        typer.echo(f"Unsupported output format: {output_format}. Use 'nx' or 'dot'.")


if __name__ == "__main__":
    app()
