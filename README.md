# Co-Authorship Network Analysis Dashboard

A Taipy-based multipage dashboard for analyzing co-authorship networks in academic publications.

## Features

- **Top Authors & Metrics**: Visualize the network of top N authors by publication count
- **Single Author Network**: Search and analyze individual author's co-author network
- **Subnetwork Analysis**: Filter and visualize subnetworks by size

## Deployment

This application is deployed on Render.com. To deploy your own version:

1. Fork this repository
2. Create a new Web Service on Render.com
3. Connect your forked repository
4. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python 2_visual_elements/final.py`

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python 2_visual_elements/final.py
   ```

## Requirements

- Python 3.9+
- See requirements.txt for Python package dependencies