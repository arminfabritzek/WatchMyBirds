import os
import json
import math
import re
import time
import logging
import cv2
from flask import Flask, send_from_directory, Response
from dash import Dash, html, dcc, callback_context, ALL, Input, Output, State
import dash_bootstrap_components as dbc

def create_web_interface(params):
    """
    Creates and returns a web interface (Dash app and Flask server) for the project.
    Expects the following keys in params:
      - output_dir: Directory where detection images are saved.
      - video_capture: Object with a method generate_frames(width, fps).
      - output_resize_width: The width to which video frames should be resized.
      - STREAM_FPS: Frame rate for the video stream.
      - IMAGE_WIDTH: Width for thumbnail images.
      - RECENT_IMAGES_COUNT: Number of recent images to display.
      - PAGE_SIZE: Number of images per gallery page.
    """
    # Unpack parameters
    output_dir = params.get("output_dir", "/output")
    video_capture = params.get("video_capture")
    output_resize_width = params.get("output_resize_width", 640)
    STREAM_FPS = params.get("STREAM_FPS", 1)
    IMAGE_WIDTH = params.get("IMAGE_WIDTH", 150)
    RECENT_IMAGES_COUNT = params.get("RECENT_IMAGES_COUNT", 3)
    PAGE_SIZE = params.get("PAGE_SIZE", 20)

    logger = logging.getLogger(__name__)

    # -----------------------------
    # Helper Functions for the Gallery
    # -----------------------------
    def get_captured_images_by_date():
        """Returns a dictionary grouping images by date."""
        try:
            files = [f for f in os.listdir(output_dir) if f.endswith("_frame_annotated.jpg")]
            files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)

            images_by_date = {}
            for filename in files:
                match = re.match(r"(\d{8})_\d{6}_frame.*\.jpg", filename)
                if match:
                    date_str = match.group(1)  # Extract YYYYMMDD
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"  # YYYY-MM-DD
                    if formatted_date not in images_by_date:
                        images_by_date[formatted_date] = []
                    images_by_date[formatted_date].append(filename)

            return images_by_date
        except Exception as e:
            logger.error(f"Error retrieving captured images: {e}")
            return {}

    def get_captured_images():
        """Returns a list of captured images (annotated) sorted by modification time (newest first)."""
        try:
            files = [f for f in os.listdir(output_dir) if f.endswith("_frame_annotated.jpg")]
            files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
            return files
        except Exception as e:
            logger.error(f"Error retrieving captured images: {e}")
            return []

    def derive_zoomed_filename(annotated_filename: str,
                               annotated_suffix: str = "_frame_annotated",
                               zoomed_suffix: str = "_frame_zoomed") -> str:
        """Derives the zoomed image filename from the annotated filename."""
        return annotated_filename.replace(annotated_suffix, zoomed_suffix)

    def create_thumbnail(image_filename: str, index: int):
        """Creates a clickable thumbnail button for the given image."""
        zoomed_filename = derive_zoomed_filename(image_filename)
        style = {
            "cursor": "pointer",
            "border": "none",
            "background": "none",
            "padding": "5px",
            "width": f"{IMAGE_WIDTH}px"
        }
        return html.Button(
            html.Img(
                src=f"/images/{zoomed_filename}",
                alt=f"Thumbnail of {zoomed_filename}",
                style=style
            ),
            id={'type': 'thumbnail', 'index': index},
            n_clicks=0,
            style={"border": "none", "background": "none", "padding": "0"}
        )

    def create_image_modal(image_filename: str, index: int):
        """Creates a modal dialog to display the full-size image with a download button."""
        original_filename = image_filename.replace("_frame_annotated", "_frame_original")  # Adjust as needed
        return dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(image_filename), close_button=False),
                dbc.ModalBody(
                    html.Img(
                        src=f"/images/{image_filename}",
                        style={"width": "100%"},
                        id={'type': 'modal-image', 'index': index}
                    )
                ),
                dbc.ModalFooter([
                    html.A(
                        dbc.Button("Download", color="secondary", target="_blank"),
                        href=f"/images/{original_filename}",
                        download=original_filename,
                        style={"textDecoration": "none"}
                    ),
                    dbc.Button(
                        "Close",
                        id={'type': 'close', 'index': index},
                        className="ml-auto",
                        n_clicks=0
                    )
                ]),
            ],
            id={'type': 'modal', 'index': index},
            is_open=False,
            size="lg",
        )

    def generate_recent_gallery():
        """Generates a gallery of the most recent captured images."""
        images = get_captured_images()
        recent_images = images[:RECENT_IMAGES_COUNT]
        thumbnails = [create_thumbnail(img, i) for i, img in enumerate(recent_images)]
        modals = [create_image_modal(img, i) for i, img in enumerate(recent_images)]
        return html.Div(thumbnails + modals, id="recent-gallery", style={"textAlign": "center"})

    def generate_navbar():
        """Creates the navbar with the logo for gallery pages."""
        return dbc.NavbarSimple(
            brand=html.Img(
                src="/assets/the_birdwatcher.jpeg",
                className="img-fluid",
                style={"width": "20vw", "minWidth": "100px", "maxWidth": "200px"}
            ),
            brand_href="/",
            children=[
                dbc.NavItem(dbc.NavLink("Live Stream", href="/", className="mx-auto")),
                dbc.NavItem(dbc.NavLink("Image Gallery", href="/gallery", className="mx-auto"))
            ],
            color="primary",
            dark=True,
            fluid=True,
            className="justify-content-center"
        )

    def generate_gallery():
        """Generates the main gallery page with daily subgallery links, including the logo."""
        images_by_date = get_captured_images_by_date()

        grid_items = []
        for date, images in images_by_date.items():
            thumbnail = images[0]  # Use the first image of the day as the representative
            grid_items.append(
                html.Div([
                    html.A(
                        html.Img(
                            src=f"/images/{thumbnail}",
                            style={"width": "150px", "cursor": "pointer", "margin": "5px"}
                        ),
                        href=f"/gallery/{date}"  # Link to subgallery
                    ),
                    html.P(date, style={"textAlign": "center", "marginTop": "5px"})
                ], style={"textAlign": "center"})
            )

        content = html.Div(
            grid_items,
            style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"}
        )
        return dbc.Container([
            generate_navbar(),
            html.H1("Image Gallery", className="text-center my-3"),
            content
        ], fluid=True)

    def generate_subgallery(date):
        """Generates a subgallery for a specific date, including the logo."""
        images_by_date = get_captured_images_by_date()
        images = images_by_date.get(date, [])

        grid_items = []
        modals = []
        for i, img in enumerate(images):
            grid_items.append(
                html.Div(
                    create_thumbnail(img, i),
                    style={"flex": "1 0 21%", "margin": "5px", "maxWidth": "150px"}
                )
            )
            modals.append(create_image_modal(img, i))

        content = html.Div(
            grid_items + modals,
            style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"}
        )
        return dbc.Container([
            generate_navbar(),
            dbc.Button("Back to Gallery", href="/gallery", color="secondary", className="mb-3"),
            html.H2(f"Images from {date}", className="text-center"),
            content
        ], fluid=True)

    # -----------------------------
    # Flask Server and Routes
    # -----------------------------
    # Create Flask server without overriding the static asset defaults.
    server = Flask(__name__)

    def setup_web_routes(app):
        # Route to serve images from the output directory.
        def serve_image(filename):
            image_path = os.path.join(output_dir, filename)
            if not os.path.exists(image_path):
                return "Image not found", 404
            return send_from_directory(output_dir, filename)
        app.route("/images/<path:filename>")(serve_image)
        # Route to serve the video feed using video_capture.generate_frames.
        app.route("/video_feed")(lambda: Response(
            video_capture.generate_frames(output_resize_width, STREAM_FPS),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        ))
    setup_web_routes(server)

    # -----------------------------
    # Dash App Setup and Layouts
    # -----------------------------

    external_stylesheets = [dbc.themes.BOOTSTRAP]
    # Pass the absolute assets folder to Dash.
    app = Dash(__name__, server=server, external_stylesheets=external_stylesheets,
               assets_folder=os.path.join(os.getcwd(), "assets"),
               assets_url_path="/assets")
    app.config.suppress_callback_exceptions = True

    def stream_layout():
        """Layout for the live stream page."""
        return dbc.Container([
            dbc.NavbarSimple(
                brand=html.Img(
                    src="/assets/the_birdwatcher.jpeg",
                    className="img-fluid",
                    style={"width": "20vw", "minWidth": "100px", "maxWidth": "200px"}
                ),
                brand_href="/",
                children=[
                    dbc.NavItem(dbc.NavLink("Live Stream", href="/", className="mx-auto")),
                    dbc.NavItem(dbc.NavLink("Image Gallery", href="/gallery", className="mx-auto"))
                ],
                color="primary",
                dark=True,
                fluid=True,
                className="justify-content-center"
            ),
            dbc.Row([
                dbc.Col(html.H1("Real-Time Stream", className="text-center"), width=12, className="my-3")
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Loading(
                        id="loading-video",
                        type="default",
                        children=html.Img(
                            id="video-feed",
                            src="/video_feed",
                            style={"width": "100%", "maxWidth": "800px", "display": "block", "margin": "0 auto"}
                        )
                    ),
                    width=12
                )
            ], className="my-3"),
            dbc.Row([
                dbc.Col(html.H2("Recent Detections", className="text-center"), width=12, className="mt-4")
            ]),
            dbc.Row([
                dbc.Col(generate_recent_gallery(), width=12)
            ], className="mb-5")
        ], fluid=True)

    def gallery_layout():
        """Layout for the image gallery page (same as landing page)."""
        return generate_gallery()

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content")
    ])

    # -----------------------------
    # Dash Callbacks
    # -----------------------------
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname")
    )
    def display_page(pathname):
        if pathname.startswith("/gallery/"):
            date = pathname.split("/")[-1]
            return generate_subgallery(date)
        elif pathname == "/gallery":
            return generate_gallery()
        elif pathname == "/" or pathname == "":
            return stream_layout()
        else:
            return "404 Not Found"

    @app.callback(
        Output({"type": "modal", "index": ALL}, "is_open"),
        [Input({'type': 'thumbnail', 'index': ALL}, 'n_clicks'),
         Input({'type': 'close', 'index': ALL}, 'n_clicks'),
         Input({'type': 'modal-image', 'index': ALL}, 'n_clicks')],  # New Input for image clicks
        [State({"type": "modal", "index": ALL}, "is_open")]
    )
    def toggle_modal(thumbnail_clicks, close_clicks, modal_image_clicks, current_states):
        ctx = callback_context
        if not ctx.triggered:
            return current_states
        triggered_prop = ctx.triggered[0]['prop_id']
        triggered_id = json.loads(triggered_prop.split('.')[0])
        new_states = [False] * len(current_states)
        if triggered_id["type"] == "thumbnail":
            new_states[triggered_id["index"]] = True
        elif triggered_id["type"] in ["close", "modal-image"]:  # Close modal on clicking close button or image
            new_states[triggered_id["index"]] = False
        return new_states

    @app.callback(
        Output("loading-spinner", "style"),
        [Input("video-feed", "loading_state")],
        prevent_initial_call=True
    )
    def show_hide_spinner(loading_state):
        if loading_state and loading_state.get('is_loading'):
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        [Output("gallery-content", "children"),
         Output("page-store", "data")],
        [Input("prev-page", "n_clicks"),
         Input("next-page", "n_clicks"),
         Input("page-store", "data"),
         Input("gallery-interval", "n_intervals")]
    )
    def update_gallery_and_page(prev_clicks, next_clicks, current_page, n_intervals):
        ctx = callback_context
        triggered_prop = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        images = get_captured_images()
        total_pages = math.ceil(len(images) / PAGE_SIZE) if images else 1

        if triggered_prop == "prev-page" and current_page > 0:
            current_page -= 1
        elif triggered_prop == "next-page" and current_page < total_pages - 1:
            current_page += 1

        start_idx = current_page * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, len(images))
        subset = images[start_idx:end_idx]

        grid_items = []
        modals = []
        for i, img in enumerate(subset):
            grid_items.append(
                html.Div(
                    create_thumbnail(img, i),
                    style={"flex": "1 0 21%", "margin": "5px", "maxWidth": "150px"}
                )
            )
            modals.append(create_image_modal(img, i))
        gallery_div = html.Div(
            grid_items + modals,
            style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"}
        )
        return gallery_div, current_page

    # -----------------------------
    # Function to Start the Web Interface
    # -----------------------------
    def run(debug=False, host="0.0.0.0", port=8050):
        app.run_server(host=host, port=port, debug=debug)

    return {"app": app, "server": server, "run": run}

# ------------------------------------------------------------------
# Module test: Run the web interface if executed directly.
# ------------------------------------------------------------------
if __name__ == '__main__':
    # For testing purposes, we create a dummy video_capture.
    class DummyVideoCapture:
        def generate_frames(self, width, fps):
            import numpy as np
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            while True:
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(1 / fps)

    params = {
        "output_dir": "./output",
        "video_capture": DummyVideoCapture(),
        "output_resize_width": 640,
        "STREAM_FPS": 1,
        "IMAGE_WIDTH": 150,
        "RECENT_IMAGES_COUNT": 3,
        "PAGE_SIZE": 20,
    }
    os.makedirs(params["output_dir"], exist_ok=True)
    interface = create_web_interface(params)
    interface["run"](debug=True)