import os
import numpy as np
import matplotlib.pyplot as plt

# INSTRUMENT_CLASSES contains the classes present in INSTRUMENTS
INSTRUMENT_CLASSES = ['Piano', 'Chromatic Percussion', 'Organ', 'Guitar',
                      'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe',
                      'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic',
                      'Percussive', 'Sound Effects']

def program_to_instrument_class(program_number):
    # Check that the supplied program is in the valid range
    if program_number < 0 or program_number > 127:
        raise ValueError('Invalid program number {}, should be between 0 and'
                         ' 127'.format(program_number))
    # Just grab the name from the instrument mapping list
    return INSTRUMENT_CLASSES[int(program_number)//8]

def plot_hist(data, path_outfile):
    print('[Fig] >> {}'.format(path_outfile))
    data_mean = np.mean(data)
    data_std = np.std(data)

    print('mean:', data_mean)
    print(' std:', data_std)

    plt.figure(dpi=100)
    plt.hist(data, bins=50)
    plt.title('mean: {:.3f}_std: {:.3f}'.format(data_mean, data_std))
    plt.savefig(path_outfile)
    plt.close()

def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):

    if verbose:
        print('[*] Scanning...')

    cnt, file_list = 0, []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue

                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')

    if is_sort:
        file_list.sort()

    return file_list
