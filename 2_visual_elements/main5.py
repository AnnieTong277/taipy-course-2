from taipy.gui import Gui, notify
import taipy.gui.builder as tgb
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# -------- Load dataset --------
df = pd.read_csv("test_scopus_raw copy.csv")

# -------- Build full co-author network --------
def build_full_coauthor_network():
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = [a.strip() for a in row["Author full names"].split(';')]
        for i in range(len(authors)):
            G.add_node(authors[i])
            for j in range(i + 1, len(authors)):
                G.add_edge(authors[i], authors[j])
    return G

# -------- Filter components by size --------
def filter_components_by_size(threshold, mode):
    G = build_full_coauthor_network()
    components = list(nx.connected_components(G))
    if mode == "More Than":
        filtered = [c for c in components if len(c) > threshold]
    else:
        filtered = [c for c in components if len(c) < threshold]
    return G, filtered

# -------- Create Plotly graph --------
def create_plotly_network(G_sub, title):
    pos = nx.kamada_kawai_layout(G_sub, scale=10)
    edge_x, edge_y = [], []
    for u, v in G_sub.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text = [], [], []
    for node in G_sub.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    show_labels = len(G_sub.nodes) <= 15
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        text=node_text if show_labels else None,
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            size=8 if not show_labels else 12,
            color='skyblue',
            line=dict(width=1)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig

# -------- Convert Matplotlib figure to base64 <img> --------
def create_base64_image(G_sub, title):
    fig_img = plt.figure(figsize=(10, 5))
    pos = nx.kamada_kawai_layout(G_sub, scale=10)
    nx.draw(G_sub, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=6)
    plt.title(title)

    buf = BytesIO()
    fig_img.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig_img)
    return f"<img src='data:image/png;base64,{encoded_img}' width='1000'/>"

# -------- Callback function --------
def run_subnetwork_analysis(state):
    state.is_loading = True
    state.subnetworks_output = ""
    state.subnetwork_figures = [None] * 5
    state.subnetwork_htmls = [None] * 5

    try:
        G_full, filtered_components = filter_components_by_size(state.network_threshold, state.network_mode)

        if not filtered_components:
            notify(state, "error", f"No sub-networks found with {state.network_mode.lower()} {state.network_threshold} authors.")
            state.subnetworks_output = f"No sub-networks with {state.network_mode.lower()} {state.network_threshold} authors."
            return

        output_text = f"Found {len(filtered_components)} sub-networks with {state.network_mode.lower()} {state.network_threshold} authors:\n\n"
        figs = [None] * 5
        htmls = [None] * 5

        for idx, comp_nodes in enumerate(filtered_components[:5], 1):
            authors_list = ", ".join(sorted(comp_nodes))
            output_text += f"### Network {idx} ({len(comp_nodes)} authors):\n<span style='font-size:1.2px !important'>{authors_list}</span>\n{'-'*60}\n"

            G_sub = G_full.subgraph(comp_nodes)

            if len(comp_nodes) > 30:
                htmls[idx-1] = f"<b>Network {idx} too large to display as an interactive graph. ({len(comp_nodes)} authors)</b>"
                figs[idx-1] = None
            else:
                figs[idx-1] = create_plotly_network(G_sub, f"Network {idx}: {len(comp_nodes)} Authors")
                htmls[idx-1] = None

        state.subnetworks_output = output_text
        state.subnetwork_figures = figs
        state.subnetwork_htmls = htmls

    finally:
        state.is_loading = False

# -------- GUI layout --------
with tgb.Page() as page:
    with tgb.part(class_name="container"):
        tgb.text("# Subnetwork Co-author Analysis", mode="md")

        with tgb.part(class_name="card"):
            tgb.text("## Subnetwork Settings", mode="md")
            tgb.slider("Threshold (N):", value="{network_threshold}", min=1, max=50)
            tgb.selector(value="{network_mode}", lov=["Less Than", "More Than"], dropdown=True)
            tgb.button(label="Run Analysis", on_action=run_subnetwork_analysis)

            tgb.indicator(visible="{is_loading}", label="Loading network analysis...")

            tgb.text("{subnetworks_output}", mode="md")
            tgb.text("### Subnetwork Graphs (Up to 5)", mode="md")

            tgb.chart(figure="{subnetwork_figures[0]}")
            tgb.text("{subnetwork_htmls[0]}", mode="html")
            tgb.chart(figure="{subnetwork_figures[1]}")
            tgb.text("{subnetwork_htmls[1]}", mode="html")
            tgb.chart(figure="{subnetwork_figures[2]}")
            tgb.text("{subnetwork_htmls[2]}", mode="html")
            tgb.chart(figure="{subnetwork_figures[3]}")
            tgb.text("{subnetwork_htmls[3]}", mode="html")
            tgb.chart(figure="{subnetwork_figures[4]}")
            tgb.text("{subnetwork_htmls[4]}", mode="html")

# -------- State Variables --------
network_threshold = 10
network_mode = "Less Than"
subnetworks_output = ""
subnetwork_figures = [None] * 5
subnetwork_htmls = [None] * 5
is_loading = False

# -------- Run the App --------
Gui(page=page).run(title="Subnetwork Co-author Analysis", dark_mode=False, debug=True, port="auto")
