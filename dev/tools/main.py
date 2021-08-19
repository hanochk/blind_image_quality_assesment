#!/usr/bin/env python3
import argparse
import sys
import tarfile
import os
from termcolor import colored
# import dohq_artifactory as jart
from pyartifactory import Artifactory

#Args
def parse_args ( returnFlag = 'args' ):
    parser = argparse.ArgumentParser()
    # requiredNamed = parser.add_argument_group('required arguments')
    group = parser.add_mutually_exclusive_group()
    # requiredNamed.add_argument('--username', '-u', help='Artifactory Username', required=True)
    # requiredNamed.add_argument('--password', '-p', help='Artifactory Password', required=True)
    parser.add_argument('--username', '-u', help='Artifactory Username')
    parser.add_argument('--password', '-p', help='Artifactory Password')
    parser.add_argument('--createtarball', '-T', help='Set to create Tarball file from source directory',
        action='store_true')
    parser.add_argument('--tarsourcedir', '-t', help=f'Source Directory to create tarball file from, [Default (current path): {os.getcwd()}]',
        default=os.getcwd())
    parser.add_argument('--arturl', '-U', help='Artifactory URL',
        default='https://aic.jfrog.io/artifactory')
    parser.add_argument('--artpath', '-a', help='Artifactory path',
        default='algo-modules-local')
    parser.add_argument('--deployfile', '-f', help='Local file path to deploy',)
    parser.add_argument('--downloadpath', '-d', help='Download path dir',
        default='./')
    group.add_argument('--deploy', '-D', help='Set to Deploy an artifact, [Default: False]',
        action='store_true')
    group.add_argument('--download', '-P', help='Set to Download an Artifact, [Default: False]',
        action='store_true')
    group.add_argument('--info', '-i', help='Set to Print Dir/Artifact Info, [Default: False]',
        action='store_true')
    group.add_argument('--repo', '-r', help='Set to Print Repo info, [Default: False]',
        action='store_true')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.1')
    if returnFlag == 'args':
        return parser.parse_args()
    else:
        return parser.print_help()

# Create Tarball gz file
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname="")
    return True

# User and password OR API_KEY
def main(args):
    USERNAME = args.username
    PASSWORD = args.password
    ARTIFACTORY_URL = args.arturl
    ARTIFACTORY_PATH = args.artpath
    LOCAL_DIRECTORY_PATH = args.downloadpath
    TARBALL = args.createtarball
    TARSOURCEDIR = args.tarsourcedir
    DEPLOY = args.deploy
    DOWNLOAD = args.download
    INFO = args.info
    REPO = args.repo
    art = None

    # Create Tarball
    if TARBALL:
        if args.deployfile is None:
            LOCAL_FILE_LOCATION = f'{os.path.dirname(TARSOURCEDIR)}/{os.path.basename(TARSOURCEDIR)}.tar.gz'
            # print (f'{os.path.dirname(TARSOURCEDIR)}/{os.path.basename(TARSOURCEDIR)}.tar.gz')
        else:
            LOCAL_FILE_LOCATION = args.deployfile
        createtar = make_tarfile (output_filename=LOCAL_FILE_LOCATION, source_dir=TARSOURCEDIR)
        if createtar is True:
            print (colored(f'The tarball file was created in: ', 'green'), LOCAL_FILE_LOCATION)
    else:
        LOCAL_FILE_LOCATION = args.deployfile
    
    # Artifactory Connection
    if None not in (USERNAME, PASSWORD):
        art = Artifactory(url=ARTIFACTORY_URL, auth=(USERNAME,PASSWORD), api_version=2)

    #Deploy
    if art is not None and True in (DEPLOY, REPO, INFO, DOWNLOAD):
        if DEPLOY:
            if LOCAL_FILE_LOCATION is None or ARTIFACTORY_PATH is None:
                print (parse_args(returnFlag="help"))
                print (colored('--deploy/-D requires --deployfile/-f and --artpath/-a', 'red'), f'--deployfile:{LOCAL_FILE_LOCATION}, --artpath:{ARTIFACTORY_PATH}')
            else:
                try:
                    artifact = art.artifacts.deploy(LOCAL_FILE_LOCATION, ARTIFACTORY_PATH)
                    print (colored(f'The file ({LOCAL_FILE_LOCATION}) was deployed to : ', 'green'), colored(f'{ARTIFACTORY_URL}/{ARTIFACTORY_PATH}', 'yellow'))
                    print (colored('Artifact deployed info: ', 'green'), artifact)
                except Exception as e:
                    print (colored('Failed to deploy file with error: ', 'red'), e)

        #Repo Info
        if REPO:
            repo = art.repositories.get_repo(ARTIFACTORY_PATH)
            print (colored(f'Repo info for of {ARTIFACTORY_PATH}: ', 'green'), repo)

        #Artifactory or Dir Info
        if INFO:
            artifact_info = art.artifacts.info(ARTIFACTORY_PATH)
            print (colored('Dir or Artifact info: ', 'green'), artifact_info)

        #Download
        if DOWNLOAD:
            try:
                print (f'Downloading artifact -{ARTIFACTORY_PATH}- to {LOCAL_DIRECTORY_PATH}')
                artifact = art.artifacts.download(ARTIFACTORY_PATH, LOCAL_DIRECTORY_PATH)
                print (colored('Downloaded Artifact to: ', 'green'), LOCAL_DIRECTORY_PATH)
            except Exception as e:
                print (colored('Failed to download artifact with error: ', 'red'), e)
    elif None in (USERNAME, PASSWORD) and True in (DEPLOY, REPO, INFO, DOWNLOAD):
        print (parse_args(returnFlag="help"))
        print (colored('Username/Password is missing: ', 'red'), f'[--deploy | --download | --info | --repo] arguments require username and password to perform these actions')


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
"""