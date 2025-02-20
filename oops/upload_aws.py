""" upload one directory from the current working directory to aws """
from pathlib import Path
import os
import glob
import boto3

def upload_dir(localDir, awsInitDir, bucketName="ubc-cv-sherlock", tag="", prefix='/'):
    """
    from current working directory, upload a 'localDir' with all its subcontents (files and subdirectories...)
    to a aws bucket
    Parameters
    ----------
    localDir :   localDirectory to be uploaded, with respect to current working directory
    awsInitDir : prefix 'directory' in aws
    bucketName : bucket in aws
    tag :        tag to select files, like *png
                 NOTE: if you use tag it must be given like --tag '*txt', in some quotation marks... for argparse
    prefix :     to remove initial '/' from file names

    Returns
    -------
    None
    """
    s3 = boto3.resource('s3')
    cwd = str(Path.cwd())
    p = Path(os.path.join(Path.cwd(), localDir))
    mydirs = list(p.glob('**'))
    for mydir in mydirs:
        fileNames = glob.glob(os.path.join(mydir, tag))
        fileNames = [f for f in fileNames if not Path(f).is_dir()]
        rows = len(fileNames)
        for i, fileName in enumerate(fileNames):
            fileName = str(fileName).replace(cwd, '')
            if fileName.startswith(prefix):  # only modify the text if it starts with the prefix
                fileName = fileName.replace(prefix, "", 1) # remove one instance of prefix
            print(f"fileName {fileName}")

            awsPath = os.path.join(awsInitDir, str(fileName))
            s3.meta.client.upload_file(fileName, bucketName, awsPath)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--localDir", help="which dir to upload to aws", default='none')
    parser.add_argument("--bucketName", help="to which bucket to upload in aws", default='ubc-cv-sherlock')
    parser.add_argument("--awsInitDir", help="to which 'directory' in aws", default='oops')
    parser.add_argument("--tag", help="some tag to select files, like *png", default='*')
    args = parser.parse_args()

    # cd whatever is above your dir, then run it
    # (below assuming this script is in ~/git/hu-libraries/netRoutines/uploadDir2Aws.py )
    # in the example below you have directory structure ~/Downloads/IO
    # you copy full directory of ~/Downloads/IO to aws bucket markus1 to 'directory' 2020/IO
    # NOTE: if you use tag it must be given like --tag '*txt', in some quotation marks...

    dirs = ['oops_val_v1', 'oops_val_v2', 'oops_val_v3', 'oops_val_v4', 'oops_val_v5', 'oops_val_v12', 'oops_val_v13', 'oops_val_v14', 'oops_val_v15']
    if args.localDir == 'none':
        for abcDir in dirs:
            upload_dir(localDir=abcDir, bucketName=args.bucketName, awsInitDir=args.awsInitDir, tag=args.tag)
    else:
        # cd ~/Downloads
        # python ~/git/hu-libraries/netRoutines/uploadDir2Aws.py --localDir IO --bucketName markus1 --awsInitDir 2020
        upload_dir(localDir=args.localDir, bucketName=args.bucketName,
                awsInitDir=args.awsInitDir, tag=args.tag)