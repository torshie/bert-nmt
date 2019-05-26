import argparse
import os.path

import torch

import fairseq.tokenizer
from fairseq import options
from fairseq.models import (
    transformer, register_model, register_model_architecture,
    FairseqModel, FairseqEncoder
)
from fairseq.data import (
    data_utils, Dictionary
)
from fairseq.tasks import (
    register_task, translation
)
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


class BertCompatibleDictionary(Dictionary):
    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>'):
        # Parent constructor is omitted intentionally.
        #super().__init__()

        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_index = self.add_symbol(pad)
        for i in range(1, 100):
            self.add_symbol('[unused%d]' % i)
        self.unk_index = self.add_symbol(unk)
        self.add_symbol('<bos>')
        self.eos_index = self.add_symbol(eos)
        self.nspecial = len(self.symbols)


class WrapBertTokenizer:
    def __init__(self, tokenizer: BertTokenizer):
        self.__tokenizer = tokenizer
        self.__pad, self.__unk, self.__bos, self.__eos = \
            tokenizer.convert_tokens_to_ids(['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
        self.unk_word = '<unk>'

    def __len__(self):
        return len(self.__tokenizer.vocab)

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.eos() + 2, len(self)).long()
        t[-1] = self.eos()
        return t

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        else:
            return self.unk_word

    def encode_line(self, line, reverse_order=False, **kwargs):
        words = self.__tokenizer.tokenize(line)
        if len(words) > 512:
            print(line)
        if reverse_order:
            words = list(reversed(words))
        words.insert(0, '[CLS]')
        words.append('[SEP]')
        ids = self.__tokenizer.convert_tokens_to_ids(words)
        return torch.IntTensor(ids)

    def pad(self):
        return self.__pad

    def unk(self):
        return self.__unk

    def bos(self):
        return self.__bos

    def eos(self):
        return self.__eos


class WrapBertModel(FairseqEncoder):
    def __init__(self, dictionary: WrapBertTokenizer, bert: BertModel):
        super().__init__(dictionary)

        self.__bert = bert

    def forward(self, src_tokens, src_lengths):
        batch_size, max_length = src_tokens.shape
        tmp = torch.arange(max_length, dtype=torch.long).repeat([batch_size, 1]).cuda()
        masks = tmp < src_lengths.view(batch_size, 1)
        paddings = masks ^ 1

        with torch.no_grad():
            encoder_out = self.__bert(src_tokens, torch.zeros_like(src_tokens),
                masks.long(), False)

        return {
            'encoder_out': encoder_out[0],
            'encoder_padding_mask': paddings
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        return transformer.TransformerEncoder.reorder_encoder_out(self,
            encoder_out, new_order)

    def max_positions(self) -> int:
        return 512


@register_model('bert_nmt')
class BertNMT(FairseqModel):
    def __init__(self, args: argparse.Namespace, encoder: WrapBertModel,
            decoder: transformer.TransformerDecoder):
        super(BertNMT, self).__init__(encoder, decoder)

        self.__args = args

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--bert-name',
            choices=('bert-base-cased', 'bert-base-uncased',
                'bert-large-cased', 'bert-large-uncased',
                'bert-base-multilingual-uncased',
                'bert-base-multilingual-cased', 'bert-base-chinese'))

    @classmethod
    def build_model(cls, args: argparse.Namespace,
            task: translation.FairseqTask):
        if args.bert_name.find('large') >= 0:
            cls.__large_model(args)
        else:
            cls.__base_model(args)

        bert_wrapper = cls.__build_bert_wrapper(args)
        decoder = cls.__build_transformer_decoder(args, task.target_dictionary)
        return BertNMT(args, bert_wrapper, decoder)

    @staticmethod
    def __build_bert_wrapper(args: argparse.Namespace):
        bert_name = args.bert_name
        tokenizer = BertTokenizer.from_pretrained(bert_name)
        tokenizer_wrapper = WrapBertTokenizer(tokenizer)
        bert_model = BertModel.from_pretrained(bert_name)
        return WrapBertModel(tokenizer_wrapper, bert_model)

    @classmethod
    def __build_transformer_decoder(cls, args: argparse.Namespace,
            tgt_dict: Dictionary):
        decoder_embed_tokens = cls.__build_embedding(tgt_dict,
            args.decoder_embed_dim)
        decoder = transformer.TransformerDecoder(args, tgt_dict,
            decoder_embed_tokens)
        return decoder

    @classmethod
    def __large_model(cls, args: argparse.Namespace):
        pass

    @classmethod
    def __base_model(cls, args: argparse.Namespace):
        pass

    @staticmethod
    def __build_embedding(dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        emb = transformer.Embedding(num_embeddings, embed_dim, padding_idx)
        return emb


@register_task('bert_translation')
class BertTranslation(translation.TranslationTask):
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        src_dict = WrapBertTokenizer(BertTokenizer.from_pretrained(args.bert_name))
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad(), "%d != %d" % (src_dict.pad(), tgt_dict.pad())
        assert src_dict.eos() == tgt_dict.eos(), "%d != %d" % (src_dict.eos(), tgt_dict.eos())
        assert src_dict.unk() == tgt_dict.unk(), "%d != %d" % (src_dict.unk(), tgt_dict.unk())
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return BertCompatibleDictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = BertCompatibleDictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, fairseq.tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d


@register_model_architecture('bert_nmt', 'bert_nmt')
def base_bert_nmt(args: argparse.Namespace):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = 768  #getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
