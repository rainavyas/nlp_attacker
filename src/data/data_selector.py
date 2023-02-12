from .data_utils import load_data


def select_data(args, train=True):
    train_data, val_data, test_data = load_data(args.data_name)

    if args.prompt_finetune:
        train_data = add_punct(train_data)
        val_data = add_punct(val_data)
        test_data = add_punct(test_data)

    if not train:
        return test_data
    return val_data, train_data

def add_punct(data):
    # NO LONGER ADD PROMPT HERE AS done in forward function of model
    for d in data:
        text = d['text'].strip()
        text = text + '.' if text[-1] not in ('.', '!', '?') else text
        d['text'] = text
    return data
