from taipy.gui import Gui, navigate, Icon, notify
import taipy.gui.builder as tgb
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
import os

# ===== Load Dataset =====
df = pd.read_csv("test_scopus_raw copy.csv")

def get_top_authors(top_n):
    author_publications = {}
    author_collaborations = defaultdict(set)  # Track unique collaborators for each author
    id_to_name = {}
    id_to_titles = defaultdict(list)
    authors_data = zip(
        df['Author(s) ID'].str.split('; '),
        df['Author full names'].str.split('; '),
        df['Title']
    )
    for ids, names, title in authors_data:
        ids = [id_.strip() for id_ in ids]
        names = [name.strip() for name in names]
        for id_, name in zip(ids, names):
            id_to_name[id_] = name
            id_to_titles[id_].append(title)
            author_publications[id_] = author_publications.get(id_, 0) + 1
            for other_id in ids:
                if other_id != id_:
                    author_collaborations[id_].add(other_id)
    df_pub_counts = pd.DataFrame([
        {
            "Author ID": id_,
            "Author Name": id_to_name[id_],
            "Publication Count": count,
            "Collaboration Count": len(author_collaborations[id_])
        }
        for id_, count in author_publications.items()
    ])
    df_pub_counts = df_pub_counts.sort_values(by="Publication Count", ascending=False)
    # Create graph for centrality calculations
    G = nx.Graph()
    for _, row in df_pub_counts.head(top_n).iterrows():
        author_id = row["Author ID"]
        G.add_node(author_id)
    for _, row in df.iterrows():
        author_ids = str(row['Author(s) ID']).split(';')
        author_ids = [id_.strip() for id_ in author_ids if id_.strip()]
        for author_id in author_ids:
            if author_id in G.nodes():
                for coauthor_id in author_ids:
                    if coauthor_id != author_id and coauthor_id in G.nodes():
                        G.add_edge(author_id, coauthor_id)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    df_pub_counts = df_pub_counts.head(top_n)
    df_pub_counts['Degree Centrality'] = df_pub_counts['Author ID'].map(degree_centrality)
    df_pub_counts['Betweenness Centrality'] = df_pub_counts['Author ID'].map(betweenness_centrality)
    df_pub_counts['Closeness Centrality'] = df_pub_counts['Author ID'].map(closeness_centrality)
    df_pub_counts['Eigenvector Centrality'] = df_pub_counts['Author ID'].map(eigenvector_centrality)
    centrality_columns = ['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality', 'Eigenvector Centrality']
    df_pub_counts[centrality_columns] = df_pub_counts[centrality_columns].round(3)
    return df_pub_counts, id_to_name, id_to_titles

def create_network_visualization(top_n):
    top_authors_df, id_to_name, id_to_titles = get_top_authors(top_n)
    top_ids = set(top_authors_df["Author ID"])
    G = nx.Graph()
    for _, row in top_authors_df.iterrows():
        author_id = row["Author ID"]
        G.add_node(author_id, size=row["Publication Count"] * 50)
    for _, row in df.iterrows():
        author_ids = str(row['Author(s) ID']).split(';')
        author_ids = [id_.strip() for id_ in author_ids if id_.strip()]
        for author_id in author_ids:
            if author_id in top_ids:
                for coauthor_id in author_ids:
                    if coauthor_id != author_id and coauthor_id in top_ids:
                        G.add_edge(author_id, coauthor_id)
    degree_centrality = nx.degree_centrality(G)
    top_degree_authors = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    nodes = list(G.nodes())
    edges = list(G.edges())
    x_vals = []
    y_vals = []
    hover_texts = []
    colors = []
    sizes = []
    for node in nodes:
        x_vals.append(pos[node][0])
        y_vals.append(pos[node][1])
        hover_texts.append(f"Name: {id_to_name.get(node, node)}<br>ID: {node}")
        colors.append(degree_centrality[node])
        sizes.append(20 if node in top_degree_authors else 10)
    node_trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        hoverinfo='text',
        text=hover_texts,
        marker=dict(
            showscale=True,
            colorscale='Turbo',
            reversescale=True,
            color=colors,
            size=sizes,
            cmin=0,
            cmax=1,
            colorbar=dict(
                thickness=15,
                title='Degree Centrality',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"<br>Co-Authorship Network (Top {top_n} Authors)",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=40),
            annotations=[dict(
                text="",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig

# ===== Top Authors & Metrics State =====
top_n = 12
author_table, _, _ = get_top_authors(top_n)
network_fig = create_network_visualization(top_n)

def on_slider_change(state, var_name, var_value):
    state.top_n = var_value
    state.author_table, _, _ = get_top_authors(var_value)
    state.network_fig = create_network_visualization(var_value)

def on_button_click(state):
    state.author_table, _, _ = get_top_authors(state.top_n)
    state.network_fig = create_network_visualization(state.top_n)

# ===== Single Author State =====
ALL_AUTHORS = sorted(set(
    name.strip() for row in df['Author full names'].dropna()
    for name in row.split(';')
))
selected_author = ALL_AUTHORS[0] if ALL_AUTHORS else ""
author_suggestions = ALL_AUTHORS

def get_coauthors(name):
    coauthors = set()
    for _, row in df.iterrows():
        ids = [x.strip() for x in str(row['Author(s) ID']).split(';')]
        names = [x.strip() for x in str(row['Author full names']).split(';')]
        if name in names:
            for id_, n in zip(ids, names):
                if n != name:
                    coauthors.add((id_, n))
    return list(coauthors)

def create_author_network(name):
    coauthors = get_coauthors(name)
    if not coauthors:
        return None
    G = nx.Graph()
    G.add_node(name)
    for cid, cname in coauthors:
        G.add_node(cname)
        G.add_edge(name, cname)
    pos = nx.kamada_kawai_layout(G, scale=10)
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    edge_x, edge_y = [], []
    for a, b in G.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=0.5, color='#888'))
    node_trace = go.Scatter(x=node_x, y=node_y, text=node_text,
                            mode='markers+text', textposition="top center",
                            marker=dict(size=12, color='skyblue'))
    return go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(title=f"Co-author Network for {name}",
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)))

author_network = create_author_network(selected_author)

def filter_authors(state, var_name, var_value):
    if not var_value:
        state.author_suggestions = ALL_AUTHORS
    else:
        state.author_suggestions = [a for a in ALL_AUTHORS if var_value.lower() in a.lower()]

def on_author_select(state, var_name, var_value):
    if var_value:
        state.selected_author = var_value
        state.author_network = create_author_network(var_value)

# ===== Subnetwork State =====
def build_full_coauthor_network():
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = [a.strip() for a in row["Author full names"].split(';')]
        for i in range(len(authors)):
            G.add_node(authors[i])
            for j in range(i + 1, len(authors)):
                G.add_edge(authors[i], authors[j])
    return G

def filter_components_by_size(thresh, mode):
    G = build_full_coauthor_network()
    comps = list(nx.connected_components(G))
    if mode == "More Than":
        return G, [c for c in comps if len(c) > thresh]
    return G, [c for c in comps if len(c) < thresh]

def create_plotly_network(G_sub, title):
    pos = nx.kamada_kawai_layout(G_sub, scale=10)
    edge_x, edge_y, node_x, node_y, text = [], [], [], [], []
    for u, v in G_sub.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    for node in G_sub.nodes():
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        text.append(node)
    return go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888')),
        go.Scatter(x=node_x, y=node_y, text=text, mode='markers+text',
                   textposition="top center", marker=dict(size=12, color='skyblue'))
    ], layout=go.Layout(title=title, margin=dict(t=60), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)))

network_threshold = 10
network_mode = "Less Than"
subnetworks_output = ""
subnetwork_figures = [None] * 5
subnetwork_htmls = [None] * 5
is_loading = False

def run_subnetwork_analysis(state):
    state.is_loading = True
    state.subnetworks_output = ""
    state.subnetwork_figures = [None] * 5
    state.subnetwork_htmls = [None] * 5
    G, comps = filter_components_by_size(state.network_threshold, state.network_mode)
    if not comps:
        state.subnetworks_output = "No subnetworks found."
        state.is_loading = False
        return
    result = ""
    figs, htmls = [None]*5, [None]*5
    for i, comp in enumerate(comps[:5]):
        nodes = sorted(comp)
        result += f"### Network {i+1} ({len(nodes)} authors):<br>{', '.join(nodes)}<br><hr>"
        subG = G.subgraph(nodes)
        if len(nodes) > 30:
            htmls[i] = f"<b>Network {i+1} too large to display.</b>"
        else:
            figs[i] = create_plotly_network(subG, f"Network {i+1}")
    state.subnetworks_output = result
    state.subnetwork_figures = figs
    state.subnetwork_htmls = htmls
    state.is_loading = False

# ===== Page Definitions =====
def page_top_authors():
    with tgb.Page() as page:
        with tgb.part(class_name="container"):
            tgb.text("# Top Author Co-Authorship Network", mode="md")
            with tgb.part(class_name="card"):
                tgb.text("## Top Authors Analysis", mode="md")
                with tgb.layout(columns="1 1"):
                    with tgb.part():
                        tgb.text("Select number of top authors to display:", mode="md")
                        tgb.slider(
                            value="{top_n}",
                            min=1,
                            max=50,
                            step=1,
                            on_change=on_slider_change
                        )
                    with tgb.part():
                        tgb.button(
                            "Update Visualization",
                            class_name="plain apply_button",
                            on_action=on_button_click
                        )
            tgb.html("br")
            with tgb.layout(columns="2 3"):
                tgb.table(data="{author_table}", height="400px", width="600px", class_name="equal-size")
                tgb.chart(figure="{network_fig}", height="800px", width="900px", class_name="equal-size")
    return page

def page_single_author():
    with tgb.Page() as page:
        tgb.text("# Single Author Analysis", mode="md")
        tgb.selector(value="{selected_author}", lov="{author_suggestions}",
                     dropdown=True, filter=True, on_change=on_author_select, on_filter=filter_authors)
        tgb.chart(figure="{author_network}")
    return page

def page_subnetwork():
    with tgb.Page() as page:
        tgb.text("# Subnetwork Analysis", mode="md")
        tgb.slider("Threshold (N):", value="{network_threshold}", min=1, max=50)
        tgb.selector(value="{network_mode}", lov=["Less Than", "More Than"])
        tgb.button("Analyze", on_action=run_subnetwork_analysis)
        tgb.indicator(visible="{is_loading}", label="Analyzing...")
        tgb.text("{subnetworks_output}", mode="md")
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
    return page

# ===== Menu and Main GUI =====
with tgb.Page() as root_page:
    tgb.menu(
        label="Navigation",
        lov=[
            ("page1", Icon("images/chart-bar.png", "Top Author & Metric")),
            ("page2", Icon("images/user.png", "Single Author Analysis")),
            ("page3", Icon("images/network.png", "Subnetwork Analysis"))
        ],
        on_action=lambda state, action, info: navigate(state, to=info["args"][0])
    )

pages = {
    "/": root_page,
    "page1": page_top_authors(),
    "page2": page_single_author(),
    "page3": page_subnetwork()
}

Gui(pages=pages, css_file="2_visual_elements/styles.css").run(
    title="Co-Authorship Analysis Dashboard", 
    dark_mode=False, 
    debug=True, 
    port=int(os.environ.get("PORT", 5000)),
    host="0.0.0.0"
)

