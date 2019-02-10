"""Implement various file/url helper functions."""

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests
import tqdm

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.deep_eos'))

LOG = logging.getLogger(__name__)


def check_sha256(model_path: Path, sha256: str) -> bool:
    """Check sha256 and return comparison result.

    :param model_path: model path
    :param sha256: reference sha256 sum
    :return: comparison result (boolean)
    """
    sha = hashlib.sha256()

    with open(model_path, 'rb') as f_p:
        while True:
            data = f_p.read(65536)
            if not data:
                break
            sha.update(data)

    return str(sha.hexdigest()) == sha256


def get_cached_path(language: str) -> Union[None, Path]:
    """Download model from cloud and returns path it.

    :param language: language code for model
    :return: model
    """
    cache_dir = Path('models')

    model_path = None

    if language in ('de', 'ud-german', 'german'):
        base_path = 'https://schweter.eu/cloud/deep_eos/models/best_ud_german_model-v1.pt'
        sha256_sum = 'a4f625cd005763d167f9383ffa1fd505f7981d77cdae996287e79a4e3861418a'

        cached = cached_path(base_path, cache_dir)

        if cached:
            model_path = cached

    # sha256 check
    if model_path:
        checksum_matches = check_sha256(model_path, sha256_sum)

        if not checksum_matches:
            LOG.error('Checksum of downloaded %s does not match with reference sha256 %s',
                      model_path, sha256_sum)
            model_path = None
        else:
            LOG.info('Checksum of %s is OK', model_path)

    return model_path


def cached_path(url: str, cache_dir: Path) -> Union[None, Path]:  # pylint: disable=too-many-locals # noqa: E501
    """Perform caching of a specific url for a given path.

    If the local path does not exist, download file from url.

    :param url: url of file
    :param cache_dir: local cache dir
    :return: path of local file
    """
    dataset_cache = Path(CACHE_ROOT) / cache_dir
    parsed = urlparse(url)

    path = None

    if parsed.scheme in ['https']:

        dataset_cache.mkdir(parents=True, exist_ok=True)

        cache_path = dataset_cache / url.split('/')[-1]

        if cache_path.exists():
            return cache_path

        response = requests.head(url)

        if not response:
            LOG.error('Could not retrieve model from %s', url)

        _, temp_filename = tempfile.mkstemp()

        LOG.info('%s not found in cache, downloading it to %s', url, temp_filename)

        req = requests.get(url, stream=True)
        content_length = req.headers.get('Content-Length')
        file_size = int(content_length) if content_length else 0
        chunk_size = 1024
        num_bars = int(file_size / chunk_size)

        with open(temp_filename, 'wb') as temp_file:
            for chunk in tqdm.tqdm(req.iter_content(chunk_size=chunk_size),
                                   total=num_bars,
                                   unit='B',
                                   leave=True):
                temp_file.write(chunk)

        shutil.copyfile(temp_filename, str(cache_path))
        os.remove(temp_filename)

        path = cache_path

    return path
