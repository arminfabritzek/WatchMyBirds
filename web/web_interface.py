import os
import json
import math
from urllib.parse import parse_qs
import re
import logging
import csv
from flask import Flask, send_from_directory, Response
from dash import Dash, html, dcc, callback_context, ALL, Input, Output, State
import dash_bootstrap_components as dbc
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Caching settings for gallery functions
_CACHE_TIMEOUT = 10  # seconds
_cached_images = {
    "images": None,
    "timestamp": 0
}

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
    detection_manager = params.get("detection_manager")  # New detection manager
    output_resize_width = params.get("output_resize_width", 640)
    STREAM_FPS = params.get("STREAM_FPS", 1)
    IMAGE_WIDTH = params.get("IMAGE_WIDTH", 150)
    RECENT_IMAGES_COUNT = params.get("RECENT_IMAGES_COUNT", 3)
    PAGE_SIZE = params.get("PAGE_SIZE", 50)

    logger = logging.getLogger(__name__)

    # ----------------------------------------------------
    # Helper Functions for CSV-Based Gallery Retrieval
    # ----------------------------------------------------
    def get_all_images():
        """
        Reads all per-day CSV files (from folders named YYYYMMDD) and returns a list of annotated image paths,
        sorted by timestamp (newest first). Each returned path is relative to output_dir.
        Assumes that each CSV file has at least:
          [0] timestamp (YYYYMMDD_HHMMSS),
          [2] annotated image filename.
        """
        images = []
        # List subfolders in output_dir that match a day folder (e.g. "20250318")
        for item in os.listdir(output_dir):
            subfolder = os.path.join(output_dir, item)
            if os.path.isdir(subfolder) and re.match(r'\d{8}', item):
                csv_file = os.path.join(subfolder, "images.csv")
                if os.path.exists(csv_file):
                    try:
                        with open(csv_file, newline="") as f:
                            reader = csv.reader(f)
                            # Skip header row if present
                            header = next(reader, None)
                            # Check if header looks like text (e.g. "timestamp")
                            if header and "timestamp" not in header[0].lower():
                                # No header; rewind
                                f.seek(0)
                                reader = csv.reader(f)
                            for row in reader:
                                try:
                                    # Expect at least 3 columns: [0]=timestamp, [2]=annotated image filename
                                    if len(row) < 3:
                                        continue
                                    timestamp = row[0].strip()
                                    annotated_name = row[2].strip()
                                    if not timestamp or not annotated_name:
                                        continue
                                    # Construct a relative path: foldername/annotated_name
                                    rel_path = os.path.join(item, annotated_name)
                                    images.append((timestamp, rel_path))
                                except Exception as row_err:
                                    logger.error(f"Error processing row {row} in file {csv_file}: {row_err}")
                                    continue
                    except Exception as file_err:
                        logger.error(f"Error reading CSV file {csv_file}: {file_err}")
                        continue
        # Sort images by timestamp descending (lexical order works with YYYYMMDD_HHMMSS)
        images.sort(key=lambda x: x[0], reverse=True)
        # Return only the relative paths
        return [img[1] for img in images]

    def get_captured_images():
        """
        Returns a list of captured annotated images using the CSV-based approach.
        Uses caching to avoid repeated disk reads.
        """
        now = time.time()
        if _cached_images["images"] is not None and (now - _cached_images["timestamp"]) < _CACHE_TIMEOUT:
            return _cached_images["images"]
        images = get_all_images()
        _cached_images["images"] = images
        _cached_images["timestamp"] = now
        return images

    def get_captured_images_by_date():
        """
        Returns a dictionary grouping images by date (YYYY-MM-DD) using the CSV-based image list.
        The grouping is done by extracting the date from the annotated filename (which is expected to start with YYYYMMDD).
        """
        images = get_captured_images()
        images_by_date = {}
        for filename in images:
            base = os.path.basename(filename)
            match = re.match(r"(\d{8})_\d{6}_frame.*\.jpg", base)
            if match:
                date_str = match.group(1)  # Extract YYYYMMDD
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                if formatted_date not in images_by_date:
                    images_by_date[formatted_date] = []
                images_by_date[formatted_date].append(filename)
        return images_by_date

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
        original_filename = image_filename.replace("_frame_annotated", "_frame_original")
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
            generate_navbar(), # Call the function here
            html.H1("Image Gallery", className="text-center my-3"),
            content
        ], fluid=True)

    def generate_subgallery(date, page=1):
        images_by_date = get_captured_images_by_date()
        images = images_by_date.get(date, [])
        total_images = len(images)
        total_pages = math.ceil(total_images / PAGE_SIZE) or 1

        page = max(1, min(page, total_pages))
        page_images = images[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

        # Create a list of page links
        page_links = []
        for p in range(1, total_pages + 1):
            style = {"margin": "5px"}
            # Different styling if it’s the active page
            if p == page:
                link = dbc.Button(str(p), color="primary", disabled=True, style=style)
            else:
                link = dbc.Button(
                    str(p),
                    color="secondary",
                    href=f"/gallery/{date}?page={p}",
                    style=style
                )
            page_links.append(link)

        pagination_controls = html.Div(
            page_links,
            style={"textAlign": "center", "marginBottom": "20px"}
        )

        grid_items = [html.Div(create_thumbnail(img, i), style={"margin": "5px"})
                      for i, img in enumerate(page_images)]
        modals = [create_image_modal(img, i) for i, img in enumerate(page_images)]

        content = html.Div(
            grid_items + modals,
            style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"}
        )

        return dbc.Container([
            generate_navbar(),
            dbc.Button("Back to Gallery", href="/gallery", color="secondary", className="mb-3"),
            html.H2(f"Images from {date}", className="text-center"),
            pagination_controls,
            content
        ], fluid=True)

    def generate_video_feed():
        # Load placeholder once
        static_placeholder_path = "assets/static_placeholder.jpg"
        if os.path.exists(static_placeholder_path):
            static_placeholder = cv2.imread(static_placeholder_path)
            if static_placeholder is not None:
                original_h, original_w = static_placeholder.shape[:2]
                ratio = original_h / float(original_w)
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * ratio)
                static_placeholder = cv2.resize(static_placeholder, (placeholder_w, placeholder_h))
            else:
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * 9 / 16)
                static_placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
        else:
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * 9 / 16)
            static_placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)

        # Parameters for text overlay
        padding_x_percent = 0.005
        padding_y_percent = 0.04
        min_font_size = 12
        min_font_size_percent = 0.05

        while True:
            start_time = time.time()
            # Retrieve the most recent display frame (raw or annotated)
            frame = detection_manager.get_display_frame()
            if frame is not None:
                # Assume we want to resize using the original resolution from video capture.
                w, h = detection_manager.video_capture.resolution
                output_resize_height = int(h * output_resize_width / w)
                resized_frame = cv2.resize(frame, (output_resize_width, output_resize_height))
                # Overlay current timestamp
                pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            else:
                # If no frame, use the placeholder.
                pil_image = Image.fromarray(cv2.cvtColor(static_placeholder, cv2.COLOR_BGR2RGB))

            draw = ImageDraw.Draw(pil_image)
            img_width, img_height = pil_image.size
            padding_x = int(img_width * padding_x_percent)
            padding_y = int(img_height * padding_y_percent)
            scaled_font_size = max(min_font_size, int(img_height * min_font_size_percent))
            try:
                custom_font = ImageFont.truetype("assets/WRP_cruft.ttf", scaled_font_size)
            except IOError:
                custom_font = ImageFont.load_default()
            timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
            bbox = draw.textbbox((0, 0), timestamp_text, font=custom_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = img_width - text_width - padding_x
            text_y = img_height - text_height - padding_y
            draw.text((text_x, text_y), timestamp_text, font=custom_font, fill="white")
            # Convert back to OpenCV BGR format
            frame_with_timestamp = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', frame_with_timestamp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            elapsed = time.time() - start_time
            desired_frame_time = 1.0 / STREAM_FPS
            if elapsed < desired_frame_time:
                time.sleep(desired_frame_time - elapsed)
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
        app.route("/video_feed")(lambda: Response(
            generate_video_feed(),
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
        [Input("url", "pathname"), Input("url", "search")]
    )
    def display_page(pathname, search):
        if pathname.startswith("/gallery/"):
            # The URL path is of the form "/gallery/<date>"
            date = pathname.split("/")[-1]
            # Parse query string for page number (default to 1)
            page = 1
            if search:
                params = parse_qs(search.lstrip('?'))
                if 'page' in params:
                    try:
                        page = int(params['page'][0])
                    except ValueError:
                        page = 1
            return generate_subgallery(date, page)
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
        elif triggered_id["type"] in ["close", "modal-image"]:
            new_states[triggered_id["index"]] = False
        return new_states

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