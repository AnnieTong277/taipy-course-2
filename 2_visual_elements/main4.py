from taipy.gui import Gui, notify
import taipy.gui.builder as tgb
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("test_scopus_raw copy.csv")
print("Dataset loaded successfully!")

def get_all_authors():
    """Get a list of all unique author names from the dataset."""
    all_authors = set()
    
    # Debug: Print detailed information about author names
    print("\nAnalyzing author names in dataset:")
    print("First 5 rows of Author full names column:")
    print(df['Author full names'].head())
    
    # Check for specific author
    print("\nSearching for 'Hansbro' in raw data:")
    hansbro_rows = df[df['Author full names'].str.contains('Hansbro', case=False, na=False)]
    print("\nRows containing 'Hansbro':")
    print(hansbro_rows['Author full names'].tolist())
    
    # Process all author names
    for names in df['Author full names'].str.split(';'):
        if isinstance(names, list):
            # Debug: Print any names containing 'Hansbro'
            hansbro_matches = [name.strip() for name in names if 'Hansbro' in name]
            if hansbro_matches:
                print("\nFound Hansbro in split names:", hansbro_matches)
            all_authors.update(name.strip() for name in names if name.strip())
    
    authors_list = sorted(list(all_authors))
    
    # Debug: Print all authors containing 'Hansbro'
    hansbro_authors = [author for author in authors_list if 'Hansbro' in author]
    print("\nAll authors containing 'Hansbro':", hansbro_authors)
    
    return authors_list

# Cache all authors at startup
print("\nLoading all authors...")
ALL_AUTHORS = get_all_authors()
print(f"Total number of unique authors: {len(ALL_AUTHORS)}")

def filter_authors(state, var_name, var_value):
    """Filter authors based on input text with more flexible matching."""
    if not var_value:
        # Show all authors when empty
        state.author_suggestions = ALL_AUTHORS
        return
    
    # Convert search term to lowercase for case-insensitive matching
    search_term = var_value.lower()
    
    # More flexible filtering
    filtered = []
    for name in ALL_AUTHORS:
        name_lower = name.lower()
        if search_term in name_lower:
            filtered.append(name)
    
    # Show all matching authors
    state.author_suggestions = filtered

def on_author_select(state, var_name, var_value):
    """Handle author selection."""
    if var_value:
        print(f"Selected author: {var_value}")
        state.author_network = create_author_network(var_value)
        if state.author_network is None:
            notify(state, "error", f"No co-authors found for {var_value}")
        else:
            notify(state, "success", f"Network updated for {var_value}")

def get_coauthors(author_name):
    """Get coauthors for a given author name."""
    coauthors = set()
    
    for _, row in df.iterrows():
        names = row['Author full names'].split(';')
        ids = row['Author(s) ID'].split(';')
        
        # Clean up the data
        names = [name.strip() for name in names]
        ids = [id_.strip() for id_ in ids]
        
        # Check if author is in this row
        if author_name in names:
            # Add all other authors as coauthors
            for name, id_ in zip(names, ids):
                if name != author_name:
                    coauthors.add((id_, name))
    
    return list(coauthors)

def create_author_network(author_name):
    # First get all coauthors
    coauthors = get_coauthors(author_name)
    if not coauthors:
        return None
    
    # Create the main graph
    G = nx.Graph()
    
    # Add main author and coauthors
    G.add_node(author_name)
    for coauthor_id, coauthor_name in coauthors:
        G.add_node(coauthor_name)
        G.add_edge(author_name, coauthor_name)
    
    # Build ego network (subnet) for the target author
    sub_nodes = list(nx.ego_graph(G, author_name, radius=1).nodes())
    sub_G = G.subgraph(sub_nodes)
    
    # Use kamada_kawai_layout for better visualization
    pos = nx.kamada_kawai_layout(sub_G, scale=10)
    
    # Build node and edge traces
    node_x, node_y, node_text = [], [], []
    
    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    edge_x, edge_y = [], []
    for u, v in sub_G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='skip',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            size=12,
            color='skyblue',
            line=dict(width=2)
        )
    )
    
    # Create final figure with improved layout
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'<br>Co-Author Network for {author_name}',
            titlefont_size=20,
            width=800,
            height=500,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

# Now initialize variables after all functions are defined
selected_author = ALL_AUTHORS[0] if ALL_AUTHORS else ""
author_suggestions = ALL_AUTHORS
author_network = create_author_network(selected_author) if selected_author else None

# Create the page
with tgb.Page() as page:
    with tgb.part(class_name="container"):
        tgb.text("# Author Network Search", mode="md")
        
        # Author search section
        with tgb.part(class_name="card"):
            tgb.text("## Search by Author Name", mode="md")
            with tgb.layout(columns="1"):
                with tgb.part():
                    tgb.text("Type to search for an author:", mode="md")
                    # Searchable selector with improved filtering
                    tgb.selector(
                        value="{selected_author}",
                        lov="{author_suggestions}",
                        dropdown=True,
                        on_change=on_author_select,
                        on_filter=filter_authors,
                        class_name="fullwidth",
                        filter=True,
                        multiple=False,
                        height="600px",
                        lov_size=len(ALL_AUTHORS)
                    )
                    tgb.text("Select an author to view their network", mode="md")
            
            # Network visualization
            tgb.chart(figure="{author_network}")

# Run the application
print("\nStarting the application...")
Gui(page=page).run(title="Author Network Search", dark_mode=False, debug=True, port="auto")
