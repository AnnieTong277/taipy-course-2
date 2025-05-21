from taipy.gui import Gui, notify
import taipy.gui.builder as tgb
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict

# Load your Scopus dataset
print("Loading dataset...")
df = pd.read_csv("test_scopus_raw copy.csv")
print("Dataset loaded successfully!")
print("Columns:", df.columns.tolist())
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nNumber of rows in dataset:", len(df))


def get_top_authors(top_n):
    print(f"\n=== Getting top {top_n} authors ===")
    author_publications = {}
    author_collaborations = defaultdict(set)  # Track unique collaborators for each author
    
    # Create Author ID → Name and Title mapping
    id_to_name = {}
    id_to_titles = defaultdict(list)
    
    authors_data = zip(
        df['Author(s) ID'].str.split('; '),
        df['Author full names'].str.split('; '),
        df['Title']
    )
    
    for ids, names, title in authors_data:
        # Clean and process author IDs and names
        ids = [id_.strip() for id_ in ids]
        names = [name.strip() for name in names]
        
        for id_, name in zip(ids, names):
            id_to_name[id_] = name
            id_to_titles[id_].append(title)
            author_publications[id_] = author_publications.get(id_, 0) + 1
            
            # Add all other authors in this paper as collaborators
            for other_id in ids:
                if other_id != id_:
                    author_collaborations[id_].add(other_id)
    
    print(f"Total unique authors found: {len(author_publications)}")
    
    # Create DataFrame with both ID and name
    df_pub_counts = pd.DataFrame([
        {
            "Author ID": id_,
            "Author Name": id_to_name[id_],
            "Publication Count": count,
            "Collaboration Count": len(author_collaborations[id_])  # Add collaboration count
        }
        for id_, count in author_publications.items()
    ])
    
    df_pub_counts = df_pub_counts.sort_values(by="Publication Count", 
                                            ascending=False)
    
    print("\nTop authors by publication count:")
    print(df_pub_counts.head(top_n))
    
    # Create graph for centrality calculations
    G = nx.Graph()
    
    # Add nodes
    for _, row in df_pub_counts.head(top_n).iterrows():
        author_id = row["Author ID"]
        G.add_node(author_id)
    
    print(f"\nNumber of nodes in graph: {len(G.nodes())}")
    
    # Add edges
    edge_count = 0
    for _, row in df.iterrows():
        author_ids = str(row['Author(s) ID']).split(';')
        author_ids = [id_.strip() for id_ in author_ids if id_.strip()]
        for author_id in author_ids:
            if author_id in G.nodes():
                for coauthor_id in author_ids:
                    if coauthor_id != author_id and coauthor_id in G.nodes():
                        G.add_edge(author_id, coauthor_id)
                        edge_count += 1
    
    print(f"Number of edges in graph: {edge_count}")
    
    # Calculate centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Add centrality metrics to DataFrame
    df_pub_counts = df_pub_counts.head(top_n)
    df_pub_counts['Degree Centrality'] = df_pub_counts['Author ID'].map(degree_centrality)
    df_pub_counts['Betweenness Centrality'] = df_pub_counts['Author ID'].map(betweenness_centrality)
    df_pub_counts['Closeness Centrality'] = df_pub_counts['Author ID'].map(closeness_centrality)
    df_pub_counts['Eigenvector Centrality'] = df_pub_counts['Author ID'].map(eigenvector_centrality)
    
    # Round centrality values to 3 decimal places
    centrality_columns = ['Degree Centrality', 'Betweenness Centrality', 
                         'Closeness Centrality', 'Eigenvector Centrality']
    df_pub_counts[centrality_columns] = df_pub_counts[centrality_columns].round(3)
    
    return df_pub_counts, id_to_name, id_to_titles

def create_network_visualization(top_n):
    print(f"\n=== Creating network visualization for top {top_n} authors ===")
    top_authors_df, id_to_name, id_to_titles = get_top_authors(top_n)
    top_ids = set(top_authors_df["Author ID"])
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for _, row in top_authors_df.iterrows():
        author_id = row["Author ID"]
        G.add_node(author_id, size=row["Publication Count"] * 50)
    
    # Add edges (only between top authors)
    for _, row in df.iterrows():
        author_ids = str(row['Author(s) ID']).split(';')
        author_ids = [id_.strip() for id_ in author_ids if id_.strip()]
        for author_id in author_ids:
            if author_id in top_ids:
                for coauthor_id in author_ids:
                    if coauthor_id != author_id and coauthor_id in top_ids:
                        G.add_edge(author_id, coauthor_id)
    
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Identify top authors by degree centrality
    top_degree_authors = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]
    
    # Use spring layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    # Precompute node attributes
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
    
    # Create node trace with fixed color scale
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
    
    # Create edge trace
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
    
    # Create final figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"<br>Co-Authorship Network (Top {top_n} Authors)",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
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
    
    print("Network visualization created successfully")
    return fig

def get_coauthors(author_name):
    coauthors = set()
    
    # Print debug information
    print(f"Searching for author: {author_name}")
    
    # Search in both Author ID and Author full names
    for _, row in df.iterrows():
        try:
            author_ids = str(row['Author(s) ID']).split(';')
            author_names = str(row['Author full names']).split(';')
            
            # Debug print
            if any(author_name.lower() in name.lower().strip() for name in author_names):
                print(f"Found match in names: {author_names}")
            
            # Check if the author is in this paper (case-insensitive)
            if any(author_name.lower() in name.lower().strip() for name in author_names) or \
               any(author_name.lower() in id_.lower().strip() for id_ in author_ids):
                # Add all other authors as coauthors
                for id_, name in zip(author_ids, author_names):
                    id_ = id_.strip()
                    name = name.strip()
                    if name.lower() != author_name.lower() and id_.lower() != author_name.lower():
                        coauthors.add((id_, name))
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    print(f"Found {len(coauthors)} coauthors")
    return list(coauthors)

def create_author_network(author_name):
    coauthors = get_coauthors(author_name)
    if not coauthors:
        return None
    
    G = nx.Graph()
    
    # Add main author
    G.add_node(author_name, size=300)
    
    # Add coauthors
    for coauthor_id, coauthor_name in coauthors:
        G.add_node(coauthor_id, size=200)
        G.add_edge(author_name, coauthor_id)
    
    # Use spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(G.nodes[node]['size'])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Co-Author Network for {author_name}',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

def on_slider_change(state, var_name, var_value):
    state.top_n = var_value
    state.network_fig = create_network_visualization(var_value)

def on_button_click(state):
    state.network_fig = create_network_visualization(state.top_n)
    notify(state, "success", "Visualization updated!")

def on_search(state):
    author_name = state.search_text.strip()
    if not author_name:
        notify(state, "error", "Please enter an author name or ID")
        return
    
    state.author_network = create_author_network(author_name)
    if state.author_network is None:
        notify(state, "error", f"No co-authors found for {author_name}")
    else:
        notify(state, "success", f"Network updated for {author_name}")

def get_author_suggestions(search_text):
    """Get author name suggestions based on search text."""
    if not search_text:
        return []
    
    # Get unique author names from the dataset
    all_authors = set()
    for names in df['Author full names'].str.split(';'):
        all_authors.update(name.strip() for name in names)
    
    # Filter authors based on search text
    suggestions = [
        name for name in all_authors 
        if search_text.lower() in name.lower()
    ]
    
    # Sort by length and limit to top 10 suggestions
    return sorted(suggestions, key=len)[:10]

def on_search_text_change(state, var_name, var_value):
    """Update author suggestions when search text changes."""
    state.author_suggestions = get_author_suggestions(var_value)

# Create the page
with tgb.Page() as page:
    with tgb.part(class_name="container"):
        tgb.text("# Top Author Co-Authorship Network", mode="md")
        
        # Top authors section
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
        with tgb.layout(columns="1 1"):
            tgb.chart(figure="{network_fig}", height="600px")

# Initialize variables
top_n = 12
network_fig = create_network_visualization(top_n)

# Run the application
Gui(page=page).run(title="Top Author Network", dark_mode=False, debug=True, port="auto")