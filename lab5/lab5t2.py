import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

init_amplitude = 1.0
init_frequency = 0.5
init_phase = 0.0
init_noise_mean = 0.0
init_noise_cov = 0.1
init_cutoff = 5.0
init_filter_order = 4
show_noise_flag = True
show_filtered_flag = True

t = np.linspace(0, 10, 1000)

last_noise_params = {'mean': init_noise_mean, 'cov': init_noise_cov}
current_noise = np.random.normal(init_noise_mean, np.sqrt(init_noise_cov), t.shape)

def harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_covariance, show_noise):
    global current_noise, last_noise_params
    y_clean = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    if noise_mean != last_noise_params['mean'] or noise_covariance != last_noise_params['cov']:
        current_noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), t.shape)
        last_noise_params['mean'] = noise_mean
        last_noise_params['cov'] = noise_covariance

    y_noisy = y_clean + current_noise
    return y_clean, y_noisy

def butter_lowpass_filter(data, cutoff, order=4):
    nyq = 0.5 / (t[1] - t[0])
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.55)

line_clean, = ax.plot(t, np.zeros_like(t), 'blue', label='Clean Harmonic')
line_noisy, = ax.plot(t, np.zeros_like(t), 'orange', label='Noisy Harmonic', alpha=0.7)
line_filtered, = ax.plot(t, np.zeros_like(t), 'green', label='Filtered', linestyle='--')

ax.legend()
ax.set_ylim(-2, 2)

amp_ax = plt.axes([0.25, 0.45, 0.65, 0.03])
freq_ax = plt.axes([0.25, 0.40, 0.65, 0.03])
phase_ax = plt.axes([0.25, 0.35, 0.65, 0.03])
noise_mean_ax = plt.axes([0.25, 0.30, 0.65, 0.03])
noise_cov_ax = plt.axes([0.25, 0.25, 0.65, 0.03])
cutoff_ax = plt.axes([0.25, 0.20, 0.65, 0.03])
filter_order_ax = plt.axes([0.25, 0.15, 0.65, 0.03])

amp_slider = Slider(amp_ax, 'Amplitude', 0.0, 2.0, valinit=init_amplitude)
freq_slider = Slider(freq_ax, 'Frequency', 0.1, 2.0, valinit=init_frequency)
phase_slider = Slider(phase_ax, 'Phase', 0.0, 2 * np.pi, valinit=init_phase)
noise_slider = Slider(noise_mean_ax, 'Noise Mean', -1.0, 1.0, valinit=init_noise_mean)
cov_slider = Slider(noise_cov_ax, 'Noise Covariance', 0.0, 1.0, valinit=init_noise_cov)
cutoff_slider = Slider(cutoff_ax, 'Cutoff Frequency', 0.1, 10.0, valinit=init_cutoff)
filter_order_slider = Slider(filter_order_ax, 'Filter Order', 1, 10, valinit=init_filter_order, valstep=1)

reset_ax = plt.axes([0.25, 0.05, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset')

checkbox_ax = plt.axes([0.8, 0.025, 0.15, 0.12])
checkbox = CheckButtons(checkbox_ax, ['Show Noise', 'Show Filtered'], [show_noise_flag, show_filtered_flag])

def update(val=None):
    amplitude = amp_slider.val
    frequency = freq_slider.val
    phase = phase_slider.val
    noise_mean = noise_slider.val
    noise_cov = cov_slider.val
    cutoff = cutoff_slider.val
    order = filter_order_slider.val
    show_noise, show_filtered = checkbox.get_status()

    y_clean, y_noisy = harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_cov, show_noise)

    line_clean.set_ydata(y_clean)
    line_noisy.set_ydata(y_noisy if show_noise else np.full_like(y_noisy, np.nan))

    y_filtered = butter_lowpass_filter(y_noisy, cutoff, int(order))
    line_filtered.set_ydata(y_filtered if show_filtered else np.full_like(y_filtered, np.nan))

    fig.canvas.draw_idle()

for slider in [amp_slider, freq_slider, phase_slider, noise_slider, cov_slider, cutoff_slider, filter_order_slider]:
    slider.on_changed(update)

checkbox.on_clicked(update)

def reset(event):
    amp_slider.reset()
    freq_slider.reset()
    phase_slider.reset()
    noise_slider.reset()
    cov_slider.reset()
    cutoff_slider.reset()
    filter_order_slider.reset()

reset_button.on_clicked(reset)

update()

plt.show()
