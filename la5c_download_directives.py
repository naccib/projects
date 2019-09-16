"""
This scripts downloads all preprocessed data from the UCLA Consortium for Neuropsychiatric Phenomics LA5c Study
automatically.
"""

from argparse import ArgumentParser
from typing import Union

from bs4 import BeautifulSoup
from requests import get, head

from io import BytesIO
from zipfile import ZipFile

def download_and_extract(url: str, out_dir: str):
    file_size = int(head(url).headers['Content-Length'].strip())
    file_size = file_size / 1024 / 1024

    print(f" * Downloading ZIP from {url}. (size={round(file_size)} MiB)")

    zip_stream = get(url, stream=True)
    zip_file = ZipFile(BytesIO(zip_stream.content))

    print(f" * Unziping to {out_dir}...")
    zip_file.extractall(path=out_dir)


parser = ArgumentParser(description='Download preprocessed data from the UCLA LA5c Study')

parser.add_argument('url', metavar='U', type=str, help='URL to be scraped for links')
parser.add_argument('out', metavar='O', type=str, help='Output folder')
parser.add_argument('--dump', metavar='D', type=bool, help='If true, dumps the URLs to url_list.txt')
parser.add_argument('--pipeline', metavar='P', type=str, help='The pipeline to be downloaded. Defaults to "freesufer"')

args = parser.parse_args()

URL: str = args.url
OUT: str = args.out
DUMP: bool = args.dump
PIPE: str = args.pipeline

if URL is None:
    print("A URL must be defined as the first parameter")
    quit(-1)

if OUT is None:
    OUT = '.'
    print('Output was no explictly defined. Using current directory.')

if PIPE is None:
    PIPE = 'freesurfer'


print(f"Sending GET request to {URL}...")

html = get(URL)

soup = BeautifulSoup(html.text, 'html.parser')

blocks = soup.find_all('div', attrs={
    'class': 'detail-block'
})

# This is a fully empirical solution:
# the block we want is observed to be the last everytime
link_block = blocks[3]

# First filter node that have a "href" attribute,
# then map the node to the attribute "href"
possible_links = [node.attrs['href'] for node in link_block.find_all('a') if 'href' in node.attrs]

freesurfer_links = [link for link in possible_links if PIPE in link]

if DUMP:
    file = open('url_list.txt', 'w+')
    for link in freesurfer_links:
        file.write(f"{link}\n")


link_count = len(freesurfer_links)

print(f"Found {link_count} files.")

for (i, link) in enumerate(freesurfer_links[2:]):
    print(f"\nLink {i + 1} of {link_count}:")
    download_and_extract(link, OUT)