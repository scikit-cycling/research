from __future__ import print_function

import os
import glob
import hashlib
import zipfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

ARCHIVE = 'https://zenodo.org/record/1202440/files/power_regression.zip'
PATH_DATA = 'data'
CHECKSUM = 'bb566e25d38ba078d58a789131be052e8cb96c4130b89d624fe149a97cd8973b'


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _check_and_unzip(zip_file):
    checksum_download = _sha256(output_file)
    if checksum_download != CHECKSUM:
        os.remove(output_file)
        raise IOError('The file downloaded was corrupted. Try again '
                      'to execute this script.')

    print('Decompressing the archive ...')
    zip_ref = zipfile.ZipFile(output_file, 'r')
    zip_ref.extractall(PATH_DATA)
    zip_ref.close()


if __name__ == '__main__':

    output_file = os.path.join(PATH_DATA, 'power_data.zip')
    filenames = glob.glob(os.path.join('data', '*', '*', '*.fit'))

    if os.path.exists(output_file) or len(filenames) > 0:
        if len(filenames) > 0:
            print('Data were already downloaded. Remove them by hand if you '
                  'do not want to use them.')
        else:
            _check_and_unzip(output_file)
    else:
        print('No data are available ...')

        if not os.path.exists(PATH_DATA):
            print('Creating a directory to store the data ...')
            os.makedirs(PATH_DATA)

        print('Downloading the data from {} ...'.format(ARCHIVE))

        urlretrieve(ARCHIVE, filename=output_file)
        _check_and_unzip(output_file)

    print('Done!')
