import numpy as np
from dash import Dash, dcc, html, Output, Input, State, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt

app = Dash(__name__)

t = np.linspace(0, 10, 1000)
init = {
    'amplitude': 1.0,
    'frequency': 0.5,
    'phase': 0.0,
    'noise_mean': 0.0,
    'noise_cov': 0.1,
    'cutoff': 5.0,
    'filter_order': 4
}

def custom_filter(y, window_size=10):
    filtered = np.convolve(y, np.ones(window_size) / window_size, mode='same')
    return filtered

def butter_filter(y, cutoff, order):
    nyq = 0.5 / (t[1] - t[0])
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, y)

def generate_signals(amplitude, frequency, phase, noise_mean, noise_cov, seed=42):
    np.random.seed(seed)
    y_clean = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    noise = np.random.normal(noise_mean, np.sqrt(noise_cov), t.shape)
    y_noisy = y_clean + noise
    return y_clean, y_noisy

app.layout = html.Div([
    html.H2("Lab5, additional task:"),
    dcc.Graph(id='main-plot', figure={'data': []}),
    html.Div([
        html.Label("Amplitude"),
        dcc.Slider(
            0.0, 2.0, step=0.1,
            value=init['amplitude'],
            id='amp-slider',
            marks={0.0: '0.0', 2.0: '2.0'}
        )
    ]),
    html.Div([
        html.Label("Frequency"),
        dcc.Slider(
            0.1, 2.0, step=0.1,
            value=init['frequency'],
            id='freq-slider',
            marks={0.1: '0.1', 2.0: '2.0'}
        )
    ]),
    html.Div([
        html.Label("Phase"),
        dcc.Slider(
            0.0, 2 * np.pi, step=0.1,
            value=init['phase'],
            id='phase-slider',
            marks={0.0: '0', round(2*np.pi, 2): f'{round(2*np.pi,2)}'}
        )
    ]),
    html.Div([
        html.Label("Noise Mean"),
        dcc.Slider(
            -1.0, 1.0, step=0.1,
            value=init['noise_mean'],
            id='noise-mean-slider',
            marks={-1.0: '-1.0', 1.0: '1.0'}
        )
    ]),
    html.Div([
        html.Label("Noise Covariance"),
        dcc.Slider(
            0.0, 1.0, step=0.05,
            value=init['noise_cov'],
            id='noise-cov-slider',
            marks={0.0: '0', 1.0: '1'}
        )
    ]),
    html.Div([
        html.Label("Cutoff Frequency"),
        dcc.Slider(
            0.1, 10.0, step=0.1,
            value=init['cutoff'],
            id='cutoff-slider',
            marks={0.1: '0.1', 10.0: '10'}
        )
    ]),
    html.Div([
        html.Label("Filter Order"),
        dcc.Slider(
            1, 10, step=1,
            value=init['filter_order'],
            id='order-slider',
            marks={1: '1', 10: '10'}
        )
    ]),
    html.Br(),
    html.Div([
        html.Label("Вибір відображення графіків:"),
        dcc.Dropdown(
            id='graph-display',
            options=[
                {'label': "-", 'value': "-"},
                {'label': "Всі на одному", 'value': "all_in_one"},
                {'label': "Кожен окремо", 'value': "separate"},
                {'label': "Гармоніка + фільтр", 'value': "clean_filtered"},
                {'label': "Гармоніка + шум", 'value': "clean_noisy"},
            ],
            value="-"
        )
    ]),
    html.Br(),
    html.Div([
        html.Label("Choose Filter:"),
        dcc.Dropdown(
            id='filter-type',
            options=[
                {'label': "Butterworth", 'value': "Butterworth"},
                {'label': "Custom Filter", 'value': "Custom Filter"}
            ],
            value="Butterworth"
        )
    ]),
    html.Br(),
    html.Button("Reset", id='reset-button', n_clicks=0)
])

@app.callback(
    Output('main-plot', 'figure'),
    Input('amp-slider', 'value'),
    Input('freq-slider', 'value'),
    Input('phase-slider', 'value'),
    Input('noise-mean-slider', 'value'),
    Input('noise-cov-slider', 'value'),
    Input('cutoff-slider', 'value'),
    Input('order-slider', 'value'),
    Input('filter-type', 'value'),
    Input('graph-display', 'value')
)
def update_plot(amp, freq, phase, noise_mean, noise_cov, cutoff, order, filter_type, graph_display):
    if graph_display == "-":
        return go.Figure(data=[])
    y_clean, y_noisy = generate_signals(amp, freq, phase, noise_mean, noise_cov)
    if filter_type == "Butterworth":
        y_filtered = butter_filter(y_noisy, cutoff, int(order))
    else:
        y_filtered = custom_filter(y_noisy, window_size=25)
    if graph_display == "separate":
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=t, y=y_clean, mode='lines', name='Clean'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=y_noisy, mode='lines', name='Noisy', line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=y_filtered, mode='lines', name='Filtered', line=dict(dash='dash', color='green')), row=3, col=1)
        fig.update_layout(height=900, title_text="Signal Visualization")
    elif graph_display == "clean_filtered":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_clean, mode='lines', name='Clean'))
        fig.add_trace(go.Scatter(x=t, y=y_filtered, mode='lines', name='Filtered', line=dict(dash='dash', color='green')))
        fig.update_layout(title_text="Clean and Filtered Signal", height=500)
    elif graph_display == "clean_noisy":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_clean, mode='lines', name='Clean'))
        fig.add_trace(go.Scatter(x=t, y=y_noisy, mode='lines', name='Noisy', line=dict(color='orange')))
        fig.update_layout(title_text="Clean and Noisy Signal", height=500)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_clean, mode='lines', name='Clean'))
        fig.add_trace(go.Scatter(x=t, y=y_noisy, mode='lines', name='Noisy', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=t, y=y_filtered, mode='lines', name='Filtered', line=dict(dash='dash', color='green')))
        fig.update_layout(title_text="Signal Visualization (All in One)", height=500)
    fig.update_layout(showlegend=True)
    return fig

@app.callback(
    Output('amp-slider', 'value'),
    Output('freq-slider', 'value'),
    Output('phase-slider', 'value'),
    Output('noise-mean-slider', 'value'),
    Output('noise-cov-slider', 'value'),
    Output('cutoff-slider', 'value'),
    Output('order-slider', 'value'),
    Output('filter-type', 'value'),
    Output('graph-display', 'value'),
    Input('reset-button', 'n_clicks')
)
def reset_controls(n_clicks):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    return (init['amplitude'],
            init['frequency'],
            init['phase'],
            init['noise_mean'],
            init['noise_cov'],
            init['cutoff'],
            init['filter_order'],
            "Butterworth",
            "-")

if __name__ == '__main__':
    app.run(debug=True)
