'''
Analyze attack outputs

1) For input json attack file (e.g. textfooler), check what fraction satisfy the constraints of attack (e.g. bae)
2) Find the worst good-bad word pair for eval of original text (test sensitivity)
3) Find the highest (on average) rank of sent words and check what fraction of predicted mask tokens change
'''
import sys
import os
import argparse
import json
from tqdm import tqdm
import torch
from statistics import mean, stdev

from src.attack.constraints import Constraint
from src.models.model_selector import select_model

def constraints_satisfied(data, attack_method='bae'):
    checker = Constraint(attack_method)
    satisfied = 0
    for d in tqdm(data):
        if checker.check_constraint(d['text'], d['att_text']):
            satisfied +=1
    return satisfied/len(data)

def get_highest_rank(logits, ids):
    '''Get highest rank for any of word id in ids'''
    ranked_inds = torch.argsort(logits, descending=True).detach().cpu().tolist()
    for ind in ranked_inds:
        if ind in ids:
            return ind, ranked_inds
    
def max_vote(mask_logits, id_pairs):
    summed = 0
    for id_pair in id_pairs:
        with torch.no_grad():
            logits = mask_logits[id_pair]
            pred_ind = torch.argmax(logits).detach().cpu().item()
        summed += pred_ind
    if summed == int(len(id_pairs)/2):
        return -1
    else:
        return int(summed/len(id_pairs))


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--json_path', type=str, nargs='+', required=True, help='saved .json file(s)')

    commandLineParser.add_argument('--cross_constraint', action='store_true', help='check constraint against attack method constraints')
    commandLineParser.add_argument('--attack_constraint', type=str, default='bae', help='which attack method constraints to check')

    commandLineParser.add_argument('--worst_pair', action='store_true', help='Eval the original text accuracy with worst performing good/bad masked token per sample')
    commandLineParser.add_argument('--best_pair', action='store_true', help='Eval the adv text accuracy with best performing good/bad masked token per sample')
    commandLineParser.add_argument('--model_path', type=str, required=False, help='trained prompt model path')
    commandLineParser.add_argument('--model_name', type=str, required=False, help='e.g. roberta-large')

    commandLineParser.add_argument('--task_mismatch', action='store_true', help='does task of mask token pred match sent prediction')
    # commandLineParser.add_argument('--model_path', type=str, required=False, help='trained prompt model path')
    # commandLineParser.add_argument('--model_name', type=str, required=False, help='e.g. roberta-large')

    commandLineParser.add_argument('--max_vote', action='store_true', help='vote after all good-bad pairs selected')
    # commandLineParser.add_argument('--model_path', type=str, required=False, help='trained prompt model path')
    # commandLineParser.add_argument('--model_name', type=str, required=False, help='e.g. roberta-large')

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
    
    if args.worst_pair or args.best_pair:
        # worst_pair: For each original text, find worst good-bad pair to harm the performance the most
        # best_pair: For each adv text, find best good-bad pair to boost the performance the most

        # load model
        model = select_model(args.model_name, model_path=args.model_path, num_labels=2, prompt_finetune=True)

        neg_words = ['terrible', 'horrible', 'poor', 'bad']
        pos_words = ['great', 'good', 'amazing', 'fantastic']

        neg_ids = [model.tokenizer(word).input_ids[1] for word in neg_words]
        pos_ids = [model.tokenizer(word).input_ids[1] for word in pos_words]

        id_pairs = []
        for neg in neg_ids:
            for pos in pos_ids:
                id_pairs.append([neg, pos])

        correct = 0
        total = 0

        if args.worst_pair:

            for d in tqdm(data):
                total += 1
                with torch.no_grad():
                    mask_pos_logits = model.predict([d['text']], all_logits=True).squeeze()
                    failed = False
                    for id_pair in id_pairs:
                        logits = mask_pos_logits[id_pair]
                        pred_ind = torch.argmax(logits).detach().cpu().item()
                        if pred_ind != d['label']:
                            failed = True
                            break
                if not failed:
                    correct += 1
            print(f'Worst pair accuracy\t{100*correct/total}%')

        if args.best_pair:

            for d in tqdm(data):
                total += 1
                with torch.no_grad():
                    mask_pos_logits = model.predict([d['att_text']], all_logits=True).squeeze()
                    for id_pair in id_pairs:
                        logits = mask_pos_logits[id_pair]
                        pred_ind = torch.argmax(logits).detach().cpu().item()
                        if pred_ind == d['label']:
                            correct += 1
                            break

            print(f'Best pair accuracy\t{100*correct/total}%')
    
    if args.task_mismatch:
        # a) Highest Rank of any sentiment word (positive or negative) in masked token prediction, averaged over the data samples
        # b) Fraction of samples with changed predicted masked token

        # Consider only originally correctly classified samples and for the adversarial attacks consider only successfully attacked samples.

        # load model
        model = select_model(args.model_name, model_path=args.model_path, num_labels=2, prompt_finetune=True)

        words = ['terrible', 'horrible', 'poor', 'bad', 'great', 'good', 'amazing', 'fantastic']
        ids = [model.tokenizer(word).input_ids[1] for word in words]

        pred_tkn_top_k = {'1':0, '10':0, '100':0} # is the original predicted token in the top-k of adv prediction for the mask token
        pred_tkn_count = 0

        sent_rank_orig = []
        sent_rank_adv = []

        for d in tqdm(data):
            if d['label'] != d['pred_label']:
                continue
            with torch.no_grad():
                mask_pos_logits_orig = model.predict([d['text']], all_logits=True).squeeze()
                rank, _ = get_highest_rank(mask_pos_logits_orig, ids)
                sent_rank_orig.append(rank)

                if d['att_pred_label'] != d['pred_label']:
                    mask_pos_logits_adv = model.predict([d['att_text']], all_logits=True).squeeze()
                    rank, ranked_adv_inds = get_highest_rank(mask_pos_logits_adv, ids)
                    sent_rank_adv.append(rank)

                    # check if predicted token changed
                    pred_tkn_count += 1
                    orig_pred = torch.argmax(mask_pos_logits_orig).cpu().detach().item()
                    if orig_pred in ranked_adv_inds[:100]:
                        pred_tkn_top_k['100'] += 1
                        if orig_pred in ranked_adv_inds[:10]:
                            pred_tkn_top_k['10'] += 1
                            if orig_pred in ranked_adv_inds[:1]:
                                pred_tkn_top_k['1'] += 1
        
        print('Rank of highest sentiment words')
        print(f'Original\t{mean(sent_rank_orig)}+-{stdev(sent_rank_orig)}')
        print(f'Successful Adversarial\t{mean(sent_rank_adv)}+-{stdev(sent_rank_adv)}')
        print()

        print('Fraction of samples with original predicted token in the top-k of adv prediction for the mask token')
        print(f'k=1\t{pred_tkn_top_k["1"]/pred_tkn_count}')
        print(f'k=10\t{pred_tkn_top_k["10"]/pred_tkn_count}')
        print(f'k=100\t{pred_tkn_top_k["100"]/pred_tkn_count}')
    
    if args.max_vote:
        # Report accuracy after using max-voting for class prediction

        # load model
        model = select_model(args.model_name, model_path=args.model_path, num_labels=2, prompt_finetune=True)

        neg_words = ['terrible', 'horrible', 'poor', 'bad']
        pos_words = ['great', 'good', 'amazing', 'fantastic']

        neg_ids = [model.tokenizer(word).input_ids[1] for word in neg_words]
        pos_ids = [model.tokenizer(word).input_ids[1] for word in pos_words]

        id_pairs = []
        for neg in neg_ids:
            for pos in pos_ids:
                id_pairs.append([neg, pos])

        correct_orig = 0
        correct_adv = 0
        total = 0

        for d in tqdm(data):
            total += 1
            with torch.no_grad():
                mask_pos_logits_orig = model.predict([d['text']], all_logits=True).squeeze()
                mask_pos_logits_adv = model.predict([d['att_text']], all_logits=True).squeeze()

            orig_pred = max_vote(mask_pos_logits_orig, id_pairs)
            if orig_pred == d['label']:
                correct_orig += 1
            
            adv_pred = max_vote(mask_pos_logits_adv, id_pairs)
            if adv_pred == d['label']:
                correct_adv += 1
        
        print('Max Voter Accuracy')
        print(f'Original\t{100*correct_orig/total}%')
        print(f'Adversarial\t{100*correct_adv/total}%')


                

