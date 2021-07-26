import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--retrain", help="train the model again with bag files in ./data/train/ directory",action="store_true")
parser.add_argument("--noextract", help="if train/test_extracted.mat or extracted img already exists, we can skip MATLAB extraction step with this command",action="store_true")
args = parser.parse_args()

if not args.noextract:
    curdir=os.getcwd();
    run_readbag="matlab -nosplash -nodesktop -wait -r \"cd('"+curdir+"\\anomaly_explainer\'); read_bag('test'); exit\""
    
    print("Waiting for MATLAB to finish extracting images and IMU data from all .bag files in ./data/test/ directory")
    #os.system(run_readbag)
    subprocess.call(run_readbag)
    print("Extraction complete!!")
    
    if args.retrain:
        tr_readbag="matlab -nosplash -nodesktop -wait -r \"cd('"+curdir+"\\anomaly_explainer\'); read_bag('train'); exit\""
        print("Waiting for MATLAB to finish extracting images and IMU data from all .bag files in ./data/train/ directory")
        subprocess.call(tr_readbag)
        print("Extraction complete!!")

run_commentator="python anomaly_explainer/commentator.py"
if args.retrain:
    run_commentator=run_commentator+" --retrain"

os.system(run_commentator)

