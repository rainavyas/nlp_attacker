import sys
import os
import argparse
import json

def accuracy(data, target='pred_label'):
    total = 0
    correct = 0
    for d in data:
        if d['label'] == d[target]:
            correct+=1
        total +=1
    return correct/total

def fool_rate(data):
    total = 0
    fooled = 0
    for d in data:
        if d['label'] == d['pred_label']:
            total+=1
            if d['att_pred_label'] != d['label']:
                fooled+=1
    return fooled/total
 
if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--json_path', type=str, required=True, help='saved .json file')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # load the json object
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    print(f'Accuracy of original predictions\t{accuracy(data)}')
    print(f"Accuracy of attacked predictions\t{accuracy(data, target='att_pred_label')}")
    print(f'Fooling Rate\t{fool_rate(data)}')
    
