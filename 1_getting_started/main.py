from taipy.gui import Gui, notify
import networkx as nx
import pandas as pd
from collections import defaultdict

# === Global Variables ===
csv_data = None
top_n = 10
selected_field = "All"
author_search = ""
network_metrics = {}
centrality_df = pd.DataFrame()
graph_html = ""

# === GUI Page ===
page = """
# Co-Authorship Network Analysis

<|layout|columns=1 3|gap=20px|>
<|
## Upload Data
<|file_selector|label=Upload CSV|on_action=upload_csv|extensions=.csv|>

## Filters
<|{selected_field}|selector|label=Field|lov=All;Health;Computer Science;Engineering|dropdown|>
<|{top_n}|slider|min=5|max=50|step=1|label=Number of Top Authors|on_change=update_network|>
<|{author_search}|input|label=Filter authors|on_change=update_network|>
|>

<|
### Network Graph
<|{graph_html}|raw|>

### Centrality Metrics
<|{centrality_df}|table|width=100%|>
|>
|>
"""

# === Functions ===

def upload_csv(state):
    global csv_data
    try:
        csv_data = pd.read_csv(state.file_selector)
        notify(state, "success", "CSV uploaded successfully!")
        update_network(state)
    except Exception as e:
        notify(state, "error", f"Failed to load file: {e}")

def update_network(state):
    global centrality_df, graph_html

    if csv_data is None:
        return

    df = csv_data.copy()
    if selected_field != "All" and "Field" in df.columns:
        df = df[df["Field"] == selected_field]

    # === Build Graph ===
    id_to_name = {}
    coauthor_weights = defaultdict(int)

    for _, row in df.iterrows():
        try:
            ids = row["Author(s) ID"].split("; ")
            names = row["Author full names"].split("; ")
        except Exception:
            continue
        for id_, name in zip(ids, names):
            id_to_name[id_.strip()] = name.strip()
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                pair = tuple(sorted([ids[i].strip(), ids[j].strip()]))
                coauthor_weights[pair] += 1

    G = nx.Graph()
    for (id1, id2), w in coauthor_weights.items():
        n1 = id_to_name.get(id1, id1)
        n2 = id_to_name.get(id2, id2)
        G.add_edge(n1, n2, weight=w)

    if author_search:
        G = G.subgraph([n for n in G.nodes if author_search.lower() in n.lower()] + 
                       [n for u, v in G.edges if author_search.lower() in u.lower() or author_search.lower() in v.lower()])

    # === Calculate Centrality Metrics ===
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigen = nx.eigenvector_centrality(G, max_iter=1000)

    # Create DataFrame with all centrality metrics
    data = [{
        "Author": node,
        "Degree": round(degree[node] * len(G), 2),  # Multiply by len(G) to get actual degree
        "Betweenness": round(betweenness[node], 3),
        "Closeness": round(closeness[node], 3),
        "Eigenvector": round(eigen[node], 3)
    } for node in G.nodes]

    # Sort by degree centrality and take top N
    data.sort(key=lambda x: x["Degree"], reverse=True)
    centrality_df = pd.DataFrame(data[:top_n])

    # === Visualization ===
    pos = nx.spring_layout(G)
    import plotly.graph_objects as go

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, hover_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        hover_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n if n in centrality_df["Author"].values else "" for n in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=[degree[n] for n in G.nodes()],
            size=[15 if n in centrality_df["Author"].values else 8 for n in G.nodes()],
            colorscale='YlGnBu',
            showscale=True
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=600, showlegend=False)
    graph_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

# === Launch ===
Gui(page).run()

