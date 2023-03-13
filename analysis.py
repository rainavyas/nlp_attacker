'''
Analyze attack outputs

1) For input json attack file (e.g. textfooler), check what fraction satisfy the constraints of attack (e.g. bae)
'''
import sys
import os
import argparse
import json
from tqdm import tqdm

from src.attack.constraints import Constraint

def constraints_satisfied(data, attack_method='bae'):
    checker = Constraint(attack_method)
    satisfied = 0
    for d in tqdm(data):
        if checker.check_constraint(d['text'], d['att_text']):
            satisfied +=1
    return satisifed/len(data)

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--json_path', type=str, nargs='+', required=True, help='saved .json file(s)')
    commandLineParser.add_argument('--cross_constraint', action='store_true', help='check constraint against attack method constraints')
    commandLineParser.add_argument('--attack_constraint', type=str, default='bae', help='which attack method constraints to check')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analysis.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # load the json object
    datas = []
    for json_path in args.json_path:
        with open(json_path, 'r') as f:
            datas.append(json.load(f))
    if len(args.json_path) == 1:
        data = datas[0]

    if args.cross_constraint:
        # check what fraction of correctly classified samples satisfied constraint
        # split successful and unsuccessful attacks

        success = [d for d in data if (d['label'] == d['pred_label']) and d['att_pred_label'] != d['label']]
        unsuccess = [d for d in data if (d['label'] == d['pred_label']) and d['att_pred_label'] == d['label']]

        print(f'Successful Attack, {args.attack_constraint} satisfied:\t {constraints_satisfied(success, args.attack_constraint)*100}% of {len(success)} samples')
        print(f'Unsuccessful Attack, {args.attack_constraint} satisfied:\t {constraints_satisfied(unsuccess, args.attack_constraint)*100}% of {len(unsuccess)} samples')

