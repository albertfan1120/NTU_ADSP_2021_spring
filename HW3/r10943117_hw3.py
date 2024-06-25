import numpy as np
from scipy.io.wavfile import write


def get_score_table():
    score_table = {}
    score_table[1] = 0
    score_table[2] = 2
    score_table[3] = 4
    score_table[4] = 5
    score_table[5] = 7
    score_table[6] = 9
    score_table[7] = 11
    
    return score_table



def getmusic(score, beat, name, config):   
    # Samples per second
    sps = 44100

    # Frequency / pitch of the sine wave
    do_freq_hz = config['do_freq']

    # Duration
    duration_s = config['duration']

    # NumpPy magic
    score_table = get_score_table()
    result_wave = np.empty((0))
    for i in range(len(score)):
        each_sample_number = np.arange(beat[i] * duration_s * sps)
        
        freq_hz = do_freq_hz * 2 ** (score_table[score[i]] / 12)
        waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
    
        waveform_quiet = waveform * config['amplitude']
        result_wave = np.concatenate((result_wave, waveform_quiet))
    
    write(name + '.wav', sps, np.int16(result_wave * 32767))


if __name__ == "__main__":
    score = [1, 1, 5, 5, 6, 6, 5]
    beat = [1, 1, 1, 1, 1, 1, 2]
    
    config = {
        'do_freq': 523,   # do frequency
        'duration': 1.0,  # beat duration in second
        'amplitude': 0.5  # each amplitude in sin
    }
    
    getmusic(score, beat, 'twinkle', config)