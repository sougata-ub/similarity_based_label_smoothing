import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
import Models
import copy
import numpy as np
import math
import re
import subprocess
import os
import tempfile
import time
import itertools
import datetime
import pandas as pd


def get_encoder(config):
	embedding_layer, pos_layer = None, None
	if config['pretrained_encoder'] == 'roberta':
		config = RobertaConfig.from_pretrained('roberta-base')
		config.output_hidden_states = True
		config.return_dict = True
		mdl = RobertaConfig.from_pretrained('roberta-base', config=config)
		enc = Models.PretrainedEncoder(mdl, trainable=config['encoder_trainable'], typ=config['pretrained_encoder']).to(config['device'])

	elif config['pretrained_encoder'] == 'gpt2':
		config = GPT2Config.from_pretrained('gpt2')
		config.output_hidden_states = True
		config.return_dict = True
		config.use_cache=False
		mdl = GPT2Model.from_pretrained('gpt2', config=config)
		enc = Models.PretrainedEncoder(mdl, trainable=config['encoder_trainable'], typ=config['pretrained_encoder']).to(config['device'])
	else:
		embedding_layer = make_embedding_layer(config)
		pos_layer = make_pos_layer(config)
		enc = Models.Encoder(hid_dim=config['hid_dim'], embedding_layer=embedding_layer,
			pos_layer=pos_layer, n_layers=config['n_layers'], n_heads=config['n_heads'],
			pf_dim=config['pf_dim'], dropout=config['dropout'], 
			device=config['device']).to(config['device'])
	return enc, embedding_layer, pos_layer


def make_embedding_layer(config):
	embedding_layer = nn.Embedding(config['out_dim'], config['hid_dim'])
	if config['embedding_pretrained']:
		embedding_layer.weight.data.copy_(torch.from_numpy(config['weights_matrix']))
		embedding_layer.requires_grad_(True)
	return embedding_layer


def make_pos_layer(config):
	pe = torch.zeros(config['max_tgt_len'], config['hid_dim'])
	for pos in range(config['max_tgt_len']):
		for i in range(0, config['hid_dim'], 2):
			pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / config['hid_dim'])))
			pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / config['hid_dim'])))

	pos_layer = nn.Embedding(config['max_tgt_len'], config['hid_dim'])
	pos_layer.weight.data.copy_(pe)
	pos_layer.requires_grad_(False)
	return pos_layer


def get_decoder(config, embedding_layer, pos_layer):
	if embedding_layer is not None and pos_layer is not None:
		dec = Models.Decoder(embedding_layer=embedding_layer, pos_embedding=pos_layer, hid_dim=config['hid_dim'], n_layers=config['n_layers'], n_heads=config['n_heads'], pf_dim=config['pf_dim'], dropout=config['dropout'], device=config['device']).to(config['device'])
	else:
		embedding_layer = make_embedding_layer(config)
		pos_layer = make_pos_layer(config)
		dec = Models.Decoder(embedding_layer=embedding_layer, pos_embedding=pos_layer, hid_dim=config['hid_dim'],n_layers=config['n_layers'], n_heads=config['n_heads'], pf_dim=config['pf_dim'], dropout=config['dropout'], device=config['device']).to(config['device'])
	return dec


def get_output_layer(config):

	if config['loss'] in ['bce_ce', 'bce_ce_smooth', 'bce_ce_cosine', 'bce_kl', 'bce_kl_smooth', 'bce_kl_cosine', 'kl_cosine_ce', 'kl_cosine_ce_smooth', 'kl_ce_cosine']:
		return Models.DoubleHeadClassification(config['hid_dim'],config['out_dim']).to(config['device'])
	else:
		return Models.SingleHeadClassification(config['hid_dim'],config['out_dim']).to(config['device'])


def encode_custom(lst, word2index, max_len=None, add_special=True):
	# lst: list of sentences
	encoded = []
	for sent in lst:
		enc = [word2index[word] for word in sent.split()]
		if max_len is not None:
			enc = enc[-max_len:]
		if add_special:
			enc = [word2index['<s>']] + enc + [word2index['</s>']]
		encoded.append(enc)
	return encoded


def decode_custom(lst, index2word, remove_special=True, special_toks=[0,1,2]):
	# lst: list of encoded sentences
	decoded = []
	for enc in lst:
		if remove_special:
			dec = [index2word[tok] for tok in enc if tok not in special_toks]
		else:
			dec = [index2word[tok] for tok in enc]
		decoded.append(" ".join(dec))
	return decoded


def encode_pretrained(lst, tokenizer, max_len=None, add_special=True):
	encoded = []
	for sent in lst:
		enc = tokenizer.encode(sent, add_special_tokens=False)
		if max_len is not None:
			enc = enc[-max_len:]
		if add_special:
			enc = [tokenizer.bos_token_id] + enc + [tokenizer.eos_token_id]
		encoded.append(enc)
	return encoded


def decode_pretrained(lst, tokenizer, skip_special_tokens=True):
	tmp = tokenizer.batch_decode(lst,skip_special_tokens)
	return tmp


def pad_sequence(pad, batch):
	tmp = []
	maxlen = max([len(i) for i in batch])
	for i in batch:
		tmp.append(i + [pad] * (maxlen - len(i)))
	return tmp


def transformer_list(obj):
	# transformer [batch, turns, lengths] into [turns, batch, lengths]
	# turns are all the same for each batch
	turns = []
	batch_size, turn_size = len(obj), len(obj[0])
	for i in range(turn_size):
		turns.append([obj[j][i] for j in range(batch_size)])    # [batch, lengths]
	return turns


def moses_multi_bleu(hypotheses, references, lowercase=False):
	"""Calculate the bleu score for hypotheses and references
	using the MOSES ulti-bleu.perl script.
	Args:
	hypotheses: A numpy array of strings where each string is a single example.
	references: A numpy array of strings where each string is a single example.
	lowercase: If true, pass the "-lc" flag to the multi-bleu script
	Returns:
	The BLEU score as a float32 value.
	"""

	if np.size(hypotheses) == 0:
		return np.float32(0.0)

	multi_bleu_path = "./multi-bleu.perl"
	os.chmod(multi_bleu_path, 0o755)

	# Dump hypotheses and references to tempfiles
	hypothesis_file = tempfile.NamedTemporaryFile()
	hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
	hypothesis_file.write(b"\n")
	hypothesis_file.flush()
	reference_file = tempfile.NamedTemporaryFile()
	reference_file.write("\n".join(references).encode("utf-8"))
	reference_file.write(b"\n")
	reference_file.flush()

	# Calculate BLEU using multi-bleu script
	with open(hypothesis_file.name, "r") as read_pred:
		bleu_cmd = [multi_bleu_path]
		if lowercase:
			bleu_cmd += ["-lc"]
		bleu_cmd += [reference_file.name]
		try:
			bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
			bleu_out = bleu_out.decode("utf-8")
			bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
			bleu_score = float(bleu_score)
		except subprocess.CalledProcessError as error:
			if error.output is not None:
				print("multi-bleu.perl script returned non-zero exit code")
				print(error.output)
				bleu_score = np.float32(0.0)

	# Close temp files
	hypothesis_file.close()
	reference_file.close()
	return bleu_score


def get_BLEU_score(hypothesis, references):
	references = [i[0] for i in references]
	references = np.asarray(references)
	hypothesis = np.asarray(hypothesis)
	bleu = moses_multi_bleu(hypothesis, references, "-lc")
	print("Moses BLEU : ",bleu)
	return bleu


def get_batch_data(dct, typ, config, plus=0):
	# typ in ['train_dict', 'valid_dict', 'test_dict']
	X_chat = copy.deepcopy(dct[typ]['X_chat'])
	Y_chat = copy.deepcopy(dct[typ]['Y_chat'])
	device = config['device']

	if config['pretrained_encoder'] == 'roberta':
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
		x_chat_encoded = encode_pretrained(X_chat, tokenizer, max_len=config['max_len'])
		enc_pad_token_id = tokenizer.pad_token_id

	elif config['pretrained_encoder'] == 'gpt2':
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		x_chat_encoded = encode_pretrained(X_chat, tokenizer, max_len=config['max_len'])
		enc_pad_token_id = tokenizer.bos_token_id

	else:
		tokenizer = dct['word2index']
		x_chat_encoded = encode_custom(X_chat, tokenizer, max_len=config['max_len'])
		enc_pad_token_id = tokenizer['<pad>']

	tokenizer = dct['word2index']
	y_chat_encoded = encode_custom(Y_chat, tokenizer)
	dec_pad_token_id = tokenizer['<pad>']

	turns = [len(dialog) for dialog in x_chat_encoded]
	turnidx = np.argsort(turns)

	x_chat_encoded = [x_chat_encoded[idx] for idx in turnidx]
	y_chat_encoded = [y_chat_encoded[idx] for idx in turnidx]

	turns = [len(dialog) for dialog in x_chat_encoded]
	fidx, bidx = 0, 0

	while fidx < len(x_chat_encoded):
		bidx = fidx + config['batch_size']
		head = turns[fidx]
		cidx = 10000
		for p, i in enumerate(turns[fidx:bidx]):
			if i != head:
				cidx = p
				break
		cidx = fidx + cidx
		bidx = min(bidx, cidx)

		s_batch, t_batch = x_chat_encoded[fidx:bidx], y_chat_encoded[fidx:bidx]

		if len(s_batch[0]) <= plus:
			fidx = bidx
			continue
		shuffleidx = np.arange(0, len(s_batch))
		np.random.shuffle(shuffleidx)
		s_batch = [s_batch[idx] for idx in shuffleidx]
		t_batch = [t_batch[idx] for idx in shuffleidx]

		s_batch = pad_sequence(enc_pad_token_id, s_batch)

		tbatch_no_eos = []
		for ix, i in enumerate(t_batch):
			tbatch_no_eos.append(i[:-1])
		t_batch = pad_sequence(dec_pad_token_id, t_batch)
		tbatch_no_eos = pad_sequence(dec_pad_token_id, tbatch_no_eos)

		s_batch = torch.tensor(s_batch, dtype=torch.long)#.to(device)

		t_batch = torch.tensor(t_batch, dtype=torch.long)#.to(device)
		tbatch_no_eos = torch.tensor(tbatch_no_eos, dtype=torch.long)#.to(device)
		tbatch_no_sos = t_batch[:, 1:]
		tgt_mask = (tbatch_no_sos != dec_pad_token_id).long()#.to(device)

		fidx = bidx

		if config['ce']:
			if config['ce_smoothen']:
				y_ce = tbatch_no_sos.new_ones(tbatch_no_sos.size() + tuple([config['out_dim']])) * config['smoothing'] / (config['out_dim'] - 1.)
				# y_ce = y_ce.to(device)
				y_ce.scatter_(-1, tbatch_no_sos.unsqueeze(-1), (1. - config['smoothing']))

			elif config['ce_cosine']:
				# y_ce = dct['similarity'][tbatch_no_sos.cpu().numpy()]
				y_ce = dct['similarity'][tbatch_no_sos.numpy()]
				# y_ce = F.softmax(torch.tensor(y_ce), -1).to(device)
				y_ce = torch.tensor(y_ce)#.to(device)

			else:
				y_ce = torch.zeros(tbatch_no_sos.shape[0], tbatch_no_sos.shape[1], len(tokenizer))#.to(device)
				y_ce.scatter_(-1, tbatch_no_sos.unsqueeze(-1), 1)

		if config['kl']:
			if config['kl_smoothen']:
				y_kl = tbatch_no_sos.new_ones(tbatch_no_sos.size() + tuple([config['out_dim']])) * config['smoothing'] / (config['out_dim'] - 1.)
				# y_kl = y_kl.to(device)
				y_kl.scatter_(-1, tbatch_no_sos.unsqueeze(-1), (1. - config['smoothing']))

			elif config['kl_cosine']:
				# y_kl = dct['similarity'][tbatch_no_sos.cpu().numpy()]
				y_kl = dct['similarity'][tbatch_no_sos.numpy()]
				# y_kl = F.softmax(torch.tensor(y_kl), -1).to(device)
				y_kl = torch.tensor(y_kl)#.to(device)

			else:
				y_kl = torch.zeros(tbatch_no_sos.shape[0], tbatch_no_sos.shape[1], len(tokenizer))#.to(device)
				y_kl.scatter_(-1, tbatch_no_sos.unsqueeze(-1), 1)

		if config['bce']:
			# y_bce = dct['word_similarity_labels'][tbatch_no_sos.cpu().numpy()]
			y_bce = dct['word_similarity_labels'][tbatch_no_sos.numpy()]
			y_bce = torch.tensor(y_bce)#.to(device)

		s_batch = s_batch.to(device)
		t_batch = t_batch.to(device)
		tbatch_no_eos = tbatch_no_eos.to(device)
		tgt_mask = tgt_mask.to(device)

		if config['loss'] in ['ce', 'ce_smooth', 'ce_cosine', 'kl', 'kl_smooth', 'kl_cosine']:
			if config['ce']:
				yield s_batch, t_batch, tbatch_no_eos, [y_ce.to(device)], tgt_mask
			else:
				yield s_batch, t_batch, tbatch_no_eos, [y_kl.to(device)], tgt_mask

		elif config['loss'] in ['bce_ce', 'bce_ce_smooth', 'bce_ce_cosine', 'bce_kl', 'bce_kl_smooth', 'bce_kl_cosine']:
			if config['ce']:
				yield s_batch, t_batch, tbatch_no_eos, [y_ce.to(device), y_bce.to(device)], tgt_mask
			else:
				yield s_batch, t_batch, tbatch_no_eos, [y_kl.to(device), y_bce.to(device)], tgt_mask

		elif config['loss'] in ['kl_cosine_ce', 'kl_cosine_ce_smooth', 'kl_ce_cosine']:
			yield s_batch, t_batch, tbatch_no_eos, [y_kl.to(device), y_ce.to(device)], tgt_mask

		else:
			yield s_batch, t_batch, tbatch_no_eos, [y_bce.to(device)], tgt_mask


def get_word_vectors(dct):
	return np.asarray(list(dct.values()))


class CELoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, y_ce, ce_op, tgt_mask):
		ce_loss = ((y_ce * F.log_softmax(ce_op, dim=-1) * tgt_mask.unsqueeze(-1)).sum(dim=-1) * -1).sum()/tgt_mask.sum()
		return ce_loss


class BCELoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, y_bce, bce_op, tgt_mask):
		bce_loss = ((y_bce*torch.log(torch.sigmoid(bce_op)) + (1-y_bce)*torch.log(1-torch.sigmoid(bce_op))) * tgt_mask.unsqueeze(-1) * -1).sum()/tgt_mask.sum()
		return bce_loss


class KLDLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, target, input, mask):
		log_prob = F.log_softmax(input, dim=-1)
		loss = F.kl_div(log_prob, target, reduction='none') * mask.unsqueeze(-1)
		return loss.sum()/mask.sum()


def get_criterion(config):
	if config['loss'] in ['ce', 'ce_smooth', 'ce_cosine', 'kl', 'kl_smooth', 'kl_cosine']:
		if config['ce']:
			criterion = [CELoss()]
		else:
			criterion = [KLDLoss()]

	elif config['loss'] in ['bce_ce', 'bce_ce_smooth', 'bce_ce_cosine', 'bce_kl', 'bce_kl_smooth', 'bce_kl_cosine']:
		if config['ce']:
			criterion = [CELoss(), BCELoss()]
		else:
			criterion = [KLDLoss(), BCELoss()]

	elif config['loss'] in ['kl_cosine_ce', 'kl_cosine_ce_smooth', 'kl_ce_cosine']:
		criterion = [CELoss(), KLDLoss()]

	else:
		criterion = [BCELoss()]

	return criterion


def get_optimizer(model, config):
	if config['optimizer'] == 'AdamW':
		optim = torch.optim.AdamW(model.parameters(), lr=config['lr'])
	elif config['optimizer'] == 'Adam':
		optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
	else:
		optim = torch.optim.RMSprop(model.parameters(), lr=config['lr'])
	return optim


def train(model, iterator, optimizer, criterion, clip):
	model.train()

	ep_t_loss = 0
	batch_num = 0

	for i, batch in enumerate(iterator):
		if i % 20 == 0:
			print("Processed batches till ", i + 1)

		src, tgt, tgt_no_eos, y, tgt_mask = batch

		optimizer.zero_grad()

		op = model(src, tgt_no_eos)
		loss_list = [criteria(y[ix].float(), op[ix].float(), tgt_mask.float()) for ix, criteria in enumerate(criterion)]
		loss = torch.stack(loss_list).mean()
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()

		ep_t_loss += loss.item()
		batch_num += 1

	return ep_t_loss / batch_num


def evaluate(model, iterator, criterion, index2word, config, test=False):
	model.eval()

	ep_t_loss = 0
	batch_num = 0
	hypothesis, corpus = [], []

	for i, batch in enumerate(iterator):
		if i % 20 == 0:
			print("Processed batches till ", i + 1)

		src, tgt, tgt_no_eos, y, tgt_mask = batch
		with torch.no_grad():
			if test:
				op = model.predict(src, tgt_no_eos)
			else:
				op = model(src, tgt_no_eos)

		# For calculating BLEU score
		if config['sample_type'] == 'argmax':
			a = decode_custom(op[config['sample_index']].argmax(-1).tolist(), index2word, remove_special=False)
		else:
			_, tmp = torch.sigmoid(op).max(-1)
			a = decode_custom(tmp.tolist(), index2word, remove_special=False)
		b = decode_custom(tgt.tolist(), index2word, remove_special=True)
		# hypothesis.extend([i for i in a])
		# corpus.extend([[i] for i in b])
		hypothesis.extend(a)
		corpus.extend(b)

		if test:
			criteria = criterion[config['sample_index']]
			loss_list = [criteria(y[config['sample_index']].float(), op[0].float(), tgt_mask.float())]
		else:
			loss_list = [criteria(y[ix].float(), op[ix].float(), tgt_mask.float()) for ix, criteria in enumerate(criterion)]
		loss = torch.stack(loss_list).mean()

		ep_t_loss += loss.item()
		batch_num += 1
	# print("\nSample Output \n", op[0].argmax(-1))
	hypothesis = [i.split('</s>')[0] for i in hypothesis]
	return ep_t_loss / batch_num, hypothesis, corpus


def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs


def init_model(config):
	model = Models.Driver(config).to(config['device'])
	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'The model has {n_params:,} trainable parameters')

	return model


def init_all(config):
	model = init_model(config)
	criterion = get_criterion(config)
	optimizer = get_optimizer(model, config)

	return model, criterion, optimizer


def get_iterators(data_dct, config):
	train = get_batch_data(data_dct, 'train_dict', config)
	valid = get_batch_data(data_dct, 'valid_dict', config)
	test = get_batch_data(data_dct, 'test_dict', config)
	return train, valid, test


def get_loss_settings(loss):
	ce_smoothen, ce_cosine = False, False
	kl_smoothen, kl_cosine = False, False
	ce, kl, bce = False, False, False

	all_terms = loss.split("_")
	if 'ce_smooth' in loss:
		ce_smoothen = True
	if 'ce_cosine' in loss:
		ce_cosine = True
	if 'kl_smooth' in loss:
		kl_smoothen = True
	if 'kl_cosine' in loss:
		kl_cosine = True

	for term in all_terms:
		if term == 'ce':
			ce = True
		elif term == 'bce':
			bce = True
		else:
			kl = True
	if bce and not ce and not kl:
		sample_type = 'random'
	else:
		sample_type = 'argmax'

	config = {'ce_smoothen': ce_smoothen, 'ce_cosine':ce_cosine, 'kl_smoothen':kl_smoothen, 'kl_cosine':kl_cosine, 'ce':ce, 'kl':kl, 'bce':bce, 'sample_type':sample_type}
	return config


def format_similarity_matrix(similarity, threshold, power):
	similarity = similarity * (similarity >= threshold)
	tmp = similarity.round(1) ** power
	similarity = tmp / tmp.sum(-1).round(1)
	return similarity


def format_similarity_matrix_smoothen(similarity_mat, threshold, smoothen):
	similarity = similarity_mat * ((similarity_mat >= threshold) & (similarity_mat < 0.99))
	residual_prob = (similarity / similarity.sum(-1).reshape(-1, 1)) * smoothen
	np.nan_to_num(residual_prob, 0)
	residual_prob[np.where(similarity_mat >= 0.999)] = (1 - residual_prob.sum(-1))
	print(float(residual_prob.shape[0]), residual_prob.sum())
	assert float(residual_prob.shape[0]) == residual_prob.sum().round(1)
	return residual_prob


def format_similarity_matrix_wordnet_smoothen(similarity_mat, wordnet_mat, threshold, smoothen):
	similarity = similarity_mat * ((similarity_mat >= threshold) & (similarity_mat < 0.99))
	assert similarity.shape == wordnet_mat.shape
	similarity = similarity * wordnet_mat
	residual_prob = (similarity / similarity.sum(-1).reshape(-1, 1)) * smoothen
	np.nan_to_num(residual_prob, 0)
	residual_prob[np.where(similarity_mat >= 0.999)] = (1 - residual_prob.sum(-1))
	print(float(residual_prob.shape[0]), residual_prob.sum())
	assert float(residual_prob.shape[0]) == residual_prob.sum().round(1)
	return residual_prob


def make_name(smoothen, dataset, similarity, wordnet, loss, ct):
	name = "ds-"+dataset+"-ls-"+loss+"-sm-"+str(smoothen)+"-sim-"+str(similarity)+"-wn-"+str(wordnet)+"-dt-"+ct.strftime("%m_%d_%Y")
	return name


def get_experiment_df():
	ct = datetime.datetime.now()
	smoothen = [0.1, 0.2]
	dataset = ['empatheticdialogues', 'dailydialog']
	similarity = [0.8, 0.5, 0]
	wordnet = [0, 1]
	losses = ['ce_smooth', 'kl_smooth', 'ce_cosine', 'kl_cosine']
	s = [smoothen, dataset, similarity, wordnet, losses]
	s2 = [[0.0], ['empatheticdialogues', 'dailydialog'], [0.0], [0], ['ce', 'kl']]

	df = pd.DataFrame(list(itertools.product(*s)), columns = ['smoothen', 'dataset', 'similarity', 'wordnet', 'loss'])
	df['name'] = df.apply(lambda x: make_name(x['smoothen'], x['dataset'], x['similarity'], x['wordnet'], x['loss'], ct), 1)

	df2 = pd.DataFrame(list(itertools.product(*s2)), columns=['smoothen', 'dataset', 'similarity', 'wordnet', 'loss'])
	df2['name'] = df2.apply(lambda x: make_name(x['smoothen'], x['dataset'], x['similarity'], x['wordnet'], x['loss'], ct), 1)

	df = pd.concat([df, df2]).reset_index(drop=True)
	return df


def make_results_dict(tot_t_loss, tot_v_loss, bert_score_list, bleu_score_list, bleurt_score_list, bert_score_hash_list, best_valid_loss, model_name, config):
	results_dict = {}
	results_dict['tot_t_loss'] = tot_t_loss
	results_dict['tot_v_loss'] = tot_v_loss
	results_dict['bert_score_list'] = bert_score_list
	results_dict['bleu_score_list'] = bleu_score_list
	results_dict['bleurt_score_list'] = bleurt_score_list
	results_dict['bert_score_hash_list'] = bert_score_hash_list
	results_dict['best_valid_loss'] = best_valid_loss
	results_dict['model_name'] = "Best-"+model_name
	config = {k: config[k] for k in config.keys() if k not in ['weights_matrix']}
	results_dict['config'] = config
	return results_dict
