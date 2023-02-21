import sys
import os
import argparse
import json
import torch
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import entropy
import numpy as np

from src.models.model_selector import select_model

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
    commandLineParser.add_argument('--fool', action='store_true', help='calulate fooling rate')
    commandLineParser.add_argument('--entropy', action='store_true', help='calulate fooling rate as retention curved of samples ranked by entropy')
    commandLineParser.add_argument('--plot_base', type=str, required=False, help='path to save output file')
    commandLineParser.add_argument('--model_path', type=str, required=False, help='trained model path')
    commandLineParser.add_argument('--model_name', type=str, required=False, help='e.g. roberta-large')
    commandLineParser.add_argument('--prompt_finetune', action='store_true', help='whether to use prompt finetuning model')
    commandLineParser.add_argument('--num_classes', type=int, default=2, help="Specify number of classes in data")
    commandLineParser.add_argument('--example', action='store_true', help='return/save the ten lowest entropy examples that were fooled')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # load the json object
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    if args.fool:
        print(f'Accuracy of original predictions\t{accuracy(data)}')
        print(f"Accuracy of attacked predictions\t{accuracy(data, target='att_pred_label')}")
        print(f'Fooling Rate\t{fool_rate(data)}')
    
    if args.entropy:
        # check to see if output predictions have been cached already
        basename = args.json_path
        basename = f'{basename.split(".")[0]}_probs.json'
        if 'probs' in args.json_path:
            print('Using cached probability predictions')
        elif os.path.exists(basename):
            print('Found cached probability predictions')
            with open(basename, 'r') as f:
                data = json.load(f)
        else:
            # need to generate predictions of original text and cache
            sf = torch.nn.Softmax(dim=0)
            print("Generating predictions")
            model = select_model(args.model_name, model_path=args.model_path, num_labels=args.num_classes, prompt_finetune=args.prompt_finetune)
            for d in tqdm(data):
                with torch.no_grad():
                    logits = model.predict([d['text']])[0].squeeze()
                    probs = sf(logits).detach().cpu().tolist()
                    d['prob'] = probs

            # cache predictions
            with open(basename, 'w') as f:
                json.dump(data, f)
        
        # keep only correctly classified samples
        data = [d for d in data if d['label']==d['pred_label']]
        
        # calculate and plot retention plot

        # rank by entropy
        for d in data:
            d['entropy'] = entropy(d['prob'])
        data = sorted(data, key=lambda x: x['entropy'], reverse=True)

        if args.example:
            fooled_data = [d for d in data[::-1] if d['pred_label']!=d['att_pred_label']][:10]
            with open('examples.json', 'w') as f:
                json.dump(fooled_data, f)
        else:

            # cumulative fooled samples (of correctly classified samples) vs retention
            retention = np.linspace(0,1,len(data)+1)
            fooled = [0]
            for d in data:
                if d['att_pred_label'] != d['pred_label']:
                    fooled.append(fooled[-1]+1)
                else:
                    fooled.append(fooled[-1])
                        
            plt.plot(retention, fooled, label='sst')
            plt.plot([0,1.0], [0, fooled[-1]], label='random', linewidth=0.5)
            plt.xlabel("Retention Fraction of correctly classified samples (high->low entropy)")
            plt.ylabel('Fooled Samples')
            plt.legend()
            plt.savefig(f'{args.plot_base}_cum.png', bbox_inches='tight')
            plt.clf()
        
            # fool rate vs entropy (binned), with underlying entropy histogram
            
            num_bins = 8
            fools = []
            centres = []
            for i in range(num_bins):
                select_data = data[int(i*(len(data)/num_bins)):int((i+1)*(len(data)/num_bins))]
                fools.append(fool_rate(select_data))
                centres.append((select_data[0]['entropy']+select_data[-1]['entropy'])/2)
            
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Entropy (of correctly classified samples)')
            ax1.set_ylabel('Fooling Rate')
            ax1.plot(centres, fools, color='k', linewidth=2)

            # underlying histogram
            entropies = [d['entropy'] for d in data]

            color = 'tab:red'
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('Density', color=color)  # we already handled the x-label with ax1
            ax2.hist(entropies, bins=2*num_bins, density=True, color=color, alpha=0.1)
            ax2.tick_params(axis='y', labelcolor=color)
            plt.savefig(f'{args.plot_base}_hist.png', bbox_inches='tight')

        


    
