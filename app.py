# -*- coding: utf-8 -*-
"""
Created on Wed Nov 7 00:53:07 2021

@author: Regina and Moe
"""
import base64
from io import BytesIO  as _BytesIO
import time

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import plotly.graph_objects as go
from PIL import Image
import requests

from model import detect, filter_boxes, detr, transform
from model import CLASSES, DEVICE

import dash_reusable_components as drc
import utils

import dash_bootstrap_components as dbc
import tensorflow as tf
import tensorflow_hub as hub

# Dash component wrappers
def Row(children=None, **kwargs):
    return html.Div(children, className="row", **kwargs)

def Column(children=None, width=1, **kwargs):
    nb_map = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}

    return html.Div(children, className=f"{nb_map[width]} columns", **kwargs)


# plotly.py helper functions
def a_pil_to_b64(im, enc="png"):
    io_buf = _BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")

    return f"data:img/{enc};base64, " + encoded


def pil_to_fig(im, showlegend=False, title=None):
    img_width, img_height = im.size
    fig = go.Figure()
    # This trace is added to help the autoresize logic work.
    fig.add_trace(go.Scatter(
        x=[img_width * 0.05, img_width * 0.95],
        y=[img_height * 0.95, img_height * 0.05],
        showlegend=False, mode="markers", marker_opacity=0, 
        hoverinfo="none", legendgroup='Image'))

    fig.add_layout_image(dict(
        source=a_pil_to_b64(im), sizing="stretch", opacity=1, layer="below",
        x=0, y=0, xref="x", yref="y", sizex=img_width, sizey=img_height,))

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width])
    
    fig.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,
        range=[img_height, 0])
    
    fig.update_layout(title=title, showlegend=showlegend)

    return fig


def tf_to_b64(tensor, ext="jpeg"):
    buffer = BytesIO()

    image = tf.cast(tf.clip_by_value(tensor[0], 0, 255), tf.uint8).numpy()
    Image.fromarray(image).save(buffer, format=ext)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{ext};base64, {encoded}"


def image_card(src, header=None):
    return dbc.Card(
        [
            dbc.CardHeader(header),
            dbc.CardBody(html.Img(src=src, style={"width": "100%"})),
        ]
    )


# Load ML model
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")


def add_bbox(fig, x0, y0, x1, y1, 
             showlegend=True, name=None, color=None, 
             opacity=0.5, group=None, text=None):
    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        fill="toself",
        opacity=opacity,
        marker_color=color,
        hoveron="fills",
        name=name,
        hoverlabel_namelength=0,
        text=text,
        legendgroup=group,
        showlegend=showlegend,
    ))


# colors for visualization
COLORS = ['#fe938c','#86e7b8','#f9ebe0','#208aae','#fe4a49', 
          '#291711', '#5f4b66', '#b98b82', '#87f5fb', '#63326e'] * 50

RANDOM_URLS = open('random_urls.txt').read().split('\n')[:-1]
print("Running on:", DEVICE)

# Start Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP]) # BOOTSTRAP
server = app.server  # Expose the server variable for deployments

colors = {
    'title': '#FFFFFF',
    'text': '#AOAABA',
    'background': '#161A1D'
}

app.layout = html.Div(className='container', children=[
    Row(html.H2("Object  Detection  and  Image  Processing",
        style={'textAlign':'center', 'color': colors['text']})),

    html.Hr(),
    Row(html.P("Input Image URL:")),
    Row([
        Column(width=8, children=[
            dcc.Input(id='input-url', style={'width': '100%'}, placeholder='Insert URL...'),
        ]),
        Column(html.Button("Detect", id="button-run", n_clicks=0), width=2),
        Column(html.Button("Random Image", id='button-random', n_clicks=0), width=2)
    ]),

    Row(dcc.Graph(id='image-output', style={"height": "70vh"}),),
    Row(dcc.Graph(id='image-gamma', style={"height": "70vh"}),),
    Row(Column(width=10, children=[
            html.P('Select Gamma'),
            dcc.Slider(
                id='slider-gamma', min=0, max=16, step=0.05, value=2.2, 
                marks={0: '0', 0.5: '0.5', 1: '1', 2.2: '2.2', 4: '4', 8: '8', 12: '12', 16: '16'},
                tooltip={'always_visible': True, 'placement':'bottom'})
    ])),
    Row(dcc.Graph(id='image-BandW-PIL', style={"height": "70vh"}),),
    Row(dcc.Graph(id='image-BandW-thresh', style={"height": "70vh"}),),
    Row(Column(width=10, children=[
            html.P('Select Threshold:'),
            dcc.Slider(
                id='slider-threshold', min=0, max=255, step=1, value=125, 
                marks={0: '0', 50: '50', 100: '100', 150: '150', 200: '200', 250: '',255: '255'},
                tooltip={'always_visible': True, 'placement':'bottom'})
    ])),   
    Row(dcc.Graph(id="graph-histogram",
        figure={"layout": {"paper_bgcolor": "#272a31","plot_bgcolor": "#272a31",}},
        config={"displayModeBar": False},),),

    Row(dcc.Graph(id='model-output', style={"height": "70vh"})),

    Row([
        Column(width=7, children=[
            html.P('Non-maximum suppression (IoU):'),
            Row([
                Column(width=3, children=dcc.Checklist(
                    id='checklist-nms', 
                    options=[{'label': 'Enabled', 'value': 'enabled'}],
                    value=[])),

                Column(width=9, children=dcc.Slider(
                    id='slider-iou', min=0, max=1, step=0.05, value=0.5, 
                    marks={0: '0', 1: '1'})),
            ])
        ]),
        Column(width=5, children=[
            html.P('Confidence Threshold:'),
            dcc.Slider(
                id='slider-confidence', min=0, max=1, step=0.05, value=0.7, 
                marks={0: '0', 1: '1'},
                tooltip={'always_visible': True, 'placement':'bottom'})
        ])
    ])   
])


@app.callback(
    [Output('button-run', 'n_clicks'),
     Output('input-url', 'value')],
    [Input('button-random', 'n_clicks')],
    [State('button-run', 'n_clicks')])
def randomize(random_n_clicks, run_n_clicks):
    return run_n_clicks+1, RANDOM_URLS[random_n_clicks%len(RANDOM_URLS)]

@app.callback(
    [Output('model-output', 'figure'),
     Output('slider-iou', 'disabled')],
    [Input('button-run', 'n_clicks'),
     Input('input-url', 'n_submit'),
     Input('slider-iou', 'value'),
     Input('slider-confidence', 'value'),
     Input('checklist-nms', 'value')],
    [State('input-url', 'value')])
def run_model(n_clicks, n_submit, iou, confidence, checklist, url):
    apply_nms = 'enabled' in checklist
    try:
        im = Image.open(requests.get(url, stream=True).raw)
    except:
        return go.Figure().update_layout(title='Incorrect URL')
    
    tstart = time.time()
    
    scores, boxes = detect(im, detr, transform, device=DEVICE)
    scores, boxes = filter_boxes(scores, boxes, confidence=confidence, iou=iou, apply_nms=apply_nms)
    
    scores = scores.data.numpy()
    boxes = boxes.data.numpy()

    tend = time.time()

    fig = pil_to_fig(im, showlegend=True, title=f'Object Detection ({tend-tstart:.2f}s)')
    existing_classes = set()

    for i in range(boxes.shape[0]):
        class_id = scores[i].argmax()
        label = CLASSES[class_id]
        confidence = scores[i].max()
        x0, y0, x1, y1 = boxes[i]

        # only display legend when it's not in the existing classes
        showlegend = label not in existing_classes
        text = f"class={label}<br>confidence={confidence:.3f}"

        add_bbox(
            fig, x0, y0, x1, y1,
            opacity=0.7, group=label, name=label, color=COLORS[class_id], 
            showlegend=showlegend, text=text,
        )

        existing_classes.add(label)

    return fig, not apply_nms


@app.callback(
    [Output('image-output', 'figure')],
    [Input('button-run', 'n_clicks'),
     Input('input-url', 'n_submit')],
    [State('input-url', 'value')])
def run_model_im2(n_clicks, n_submit, url):
    # apply_nms = 'enabled' in checklist
    try:
        im = Image.open(requests.get(url, stream=True).raw)
    except:
        return go.Figure().update_layout(title='Incorrect URL')
    fig = pil_to_fig(im, showlegend=True, title=f'Original Image')
    return [go.Figure(fig)]

@app.callback(
    [Output('image-gamma', 'figure')],
    [Input('button-run', 'n_clicks'),
     Input('slider-gamma', 'value'),
     Input('input-url', 'n_submit')],
    [State('input-url', 'value')])
def run_model_im3(n_clicks, gamma, n_submit, url):
    try:
        im = Image.open(requests.get(url, stream=True).raw)
    except:
        return go.Figure().update_layout(title='Incorrect URL')
    gamma1 = gamma
    row = im.size[0]
    col = im.size[1]
    result_img1 = Image.new("RGB", (row, col))
    for x in range(0 , row):
        for y in range(0, col):
            r = pow(im.getpixel((x,y))[0]/255, (1/gamma1))*255
            g = pow(im.getpixel((x,y))[1]/255, (1/gamma1))*255
            b = pow(im.getpixel((x,y))[2]/255, (1/gamma1))*255
            if r >= 255:
                r = 255
            if g >= 255:
                g = 255
            if b >= 255:
                b = 255
            result_img1.putpixel((x,y), (int(r), int(g), int(b)))


    fig = pil_to_fig(result_img1, showlegend=True, title=f'Gamma Corrected Image')
    return [go.Figure(fig)]


@app.callback(
    [Output('image-BandW-PIL', 'figure')],
    [Input('button-run', 'n_clicks'),
     Input('input-url', 'n_submit')],
    [State('input-url', 'value')])
def run_model_im4(n_clicks, n_submit, url):
    try:
        im = Image.open(requests.get(url, stream=True).raw).convert('1')
    except:
        return go.Figure().update_layout(title='Incorrect URL')
    fig = pil_to_fig(im, showlegend=True, title=f'Thresholding with Black and White Dithering')
    return [go.Figure(fig)]

@app.callback(
    [Output('image-BandW-thresh', 'figure')],
    [Input('button-run', 'n_clicks'),
     Input('slider-threshold', 'value'),
     Input('input-url', 'n_submit')],
    [State('input-url', 'value')])
def run_model_im5(n_clicks, thresh, n_submit, url):
    try:
        im = Image.open(requests.get(url, stream=True).raw)
    except:
        return go.Figure().update_layout(title='Incorrect URL')
    #thresh = 200
    fn = lambda x : 255 if x > thresh else 0
    imT = im.convert('L').point(fn, mode='1')    
    fig = pil_to_fig(imT, showlegend=True, title=f'Thresholding with Binary Segmentation')
    return [go.Figure(fig)]

@app.callback(
    Output("graph-histogram", "figure"), 
    [Input('button-run', 'n_clicks'),
     Input('input-url', 'n_submit')],
    [State('input-url', 'value')])
def update_histogram(n_clicks, n_submit, url):
    try:
        im = Image.open(requests.get(url, stream=True).raw)
    except:
        return go.Figure().update_layout(title='Incorrect URL')

    return utils.show_histogram(im)

@app.callback(
    [Output('image-thresholding-output', 'figure')],
    [Input('button-run', 'n_clicks'),
     Input('input-url', 'n_submit')],
    [State('input-url', 'value')])
def run_model_threshim(n_clicks, n_submit, url):
    # apply_nms = 'enabled' in checklist
    try:
        im = Image.open(requests.get(url, stream=True).raw)
    except:
        return go.Figure().update_layout(title='Incorrect URL')

    npim = drc.b64_to_numpy(drc.pil_to_b64(im))
    binary_mask = npim *2
    imthreshpil = drc.b64_to_pil(drc.numpy_to_b64(binary_mask))
    fig = pil_to_fig(imthreshpil, showlegend=True, title=f'Thresholding')
    return [go.Figure(fig)]


@app.callback(
    Output("graph-histogram-colorsOLD", "figure"), 
    [Input("interactive-image", "figure")],)
def update_histogram(figure):
    # Retrieve the image stored inside the figure
    enc_str = figure["layout"]["images"][0]["source"].split(";base64,")[-1]
    # Creates the PIL Image object from the b64 png encoding
    im_pil = a_b64_to_pil(string=enc_str)

    return utils.show_histogram(im_pil)


@app.callback(
    [Output("original-img", "children"), Output("enhanced-img", "children")],
    [Input("img-upload", "contents")],
    [State("img-upload", "filename")],
)
def enhance_image(img_str, filename):
    if img_str is None:
        return dash.no_update, dash.no_update

    # sr_str = img_str # PLACEHOLDER
    low_res = preprocess_b64(img_str)
    super_res = model(tf.cast(low_res, tf.float32))
    sr_str = tf_to_b64(super_res)

    lr = image_card(img_str, header="Original Image")
    sr = image_card(sr_str, header="Enhanced Image")

    return lr, sr


if __name__ == '__main__':
    app.run_server(debug=True)
