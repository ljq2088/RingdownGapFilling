import matplotlib.pyplot as plt

def visualize_waveform(original_signal, masked_signal, reconstructed_signal, title='Waveform Reconstruction'):
    plt.figure(figsize=(12, 6))
    plt.plot(original_signal, 'b--', label='Original Signal')
    plt.plot(masked_signal, 'r-', label='Masked Signal (With Gap)')
    plt.plot(reconstructed_signal, 'g-', label='Reconstructed Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()
def visualize_waveform_with_noise(original_signal, masked_signal,masked_data, reconstructed_signal, title='Waveform Reconstruction'):
    plt.figure(figsize=(12, 6))
    plt.plot(original_signal, 'b--', label='Original Signal')
    plt.plot(masked_signal, 'r-', label='Masked Signal (With Gap)')
    plt.plot(masked_data, 'y-', label='Masked Data')
    plt.plot(reconstructed_signal, 'g-', label='Reconstructed Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()
def visualize_waveform_decomposition(original_signal, signal_22,signal_21,signal_33,signal_44, reconstructed_signal, title='Signal Decomposition'):
    plt.figure(figsize=(12, 6))
    plt.plot(original_signal, 'b--', label='Original Signal')
    plt.plot(signal_22, 'r-', label='22 Mode Signal')
    plt.plot(signal_21, 'g-', label='21 Mode Signal')    
    plt.plot(signal_33, 'y-', label='33 Mode Signal')
    plt.plot(signal_44, 'k-', label='44 Mode Signal')
    plt.plot(reconstructed_signal, 'm-', label='Reconstructed Signal')

    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()