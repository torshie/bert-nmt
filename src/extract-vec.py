#!/usr/bin/env python3

import argparse

import torch
import pytorch_pretrained_bert as bert


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--bert-name', required=True)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    cmdline = parse_cmdline()
    print('Loading pretrained bert model ...')
    model = bert.BertModel.from_pretrained(cmdline.bert_name)
    print('Loading pretrained bert tokenizer ...')
    do_lower_case = (cmdline.bert_name == 'bert-base-chinese'
        or cmdline.bert_name.find('uncased') >= 0)
    tokenizer = bert.BertTokenizer.from_pretrained(cmdline.bert_name,
        do_lower_case=do_lower_case)

    vocab = [(v, k) for k, v in tokenizer.vocab.items()]
    embeddings = model.embeddings
    type_vector = embeddings.token_type_embeddings(torch.LongTensor([0]))

    print("Calculating word embeddings ...")
    final_weights = embeddings.word_embeddings.weight + type_vector

    print("Dumping vectors ...")
    with open(cmdline.output, 'w') as f:
        f.write('%d %d\n' % final_weights.shape)
        for i, (offset, word) in enumerate(vocab):
            assert i == offset
            vec = [str(x.item()) for x in final_weights[i]]
            f.write(word + ' ')
            f.write(' '.join(vec))
            f.write('\n')


if __name__ == '__main__':
    main()
