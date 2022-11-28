import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class PositionwiseFeedforwardLayer(nn.Module):
	def __init__(self, hid_dim, pf_dim, dropout):
		super().__init__()

		self.fc_1 = nn.Linear(hid_dim, pf_dim)
		self.fc_2 = nn.Linear(pf_dim, hid_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		#x = [batch size, seq len, hid dim]

		x = self.dropout(torch.relu(self.fc_1(x)))
		#x = [batch size, seq len, pf dim]

		x = self.fc_2(x)
		#x = [batch size, seq len, hid dim]

		return x


class MultiHeadAttentionLayer(nn.Module):
	def __init__(self, hid_dim, n_heads, dropout, device):
		super().__init__()

		assert hid_dim % n_heads == 0

		self.hid_dim = hid_dim
		self.n_heads = n_heads
		self.head_dim = hid_dim // n_heads

		self.fc_q = nn.Linear(hid_dim, hid_dim)
		self.fc_k = nn.Linear(hid_dim, hid_dim)
		self.fc_v = nn.Linear(hid_dim, hid_dim)

		self.fc_o = nn.Linear(hid_dim, hid_dim)

		self.dropout = nn.Dropout(dropout)

		self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

	def forward(self, query, key, value, mask = None):

		batch_size = query.shape[0]  
		# query = [batch size, query len, hid dim]
		# key = [batch size, key len, hid dim]
		# value = [batch size, value len, hid dim]
		        
		Q = self.fc_q(query)
		K = self.fc_k(key)
		V = self.fc_v(value)  
		# Q = [batch size, query len, hid dim]
		# K = [batch size, key len, hid dim]
		# V = [batch size, value len, hid dim]

		Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) 
		# Q = [batch size, n heads, query len, head dim]
		# K = [batch size, n heads, key len, head dim]
		# V = [batch size, n heads, value len, head dim]
		        
		energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale    
		# energy = [batch size, n heads, query len, key len]

		if mask is not None:
			energy = energy.masked_fill(mask == 0, -1e10)

		attention = torch.softmax(energy, dim = -1)         
		# attention = [batch size, n heads, query len, key len]

		x = torch.matmul(self.dropout(attention), V)
		# x = [batch size, n heads, query len, head dim]

		x = x.permute(0, 2, 1, 3).contiguous()
		# x = [batch size, query len, n heads, head dim]

		x = x.view(batch_size, -1, self.hid_dim)
		# x = [batch size, query len, hid dim]

		x = self.fc_o(x)  
		# x = [batch size, query len, hid dim]

		return x, attention


class Norm(nn.Module):
	def __init__(self, d_model, eps = 1e-6):
		super().__init__()

		self.size = d_model
		# create two learnable parameters to calibrate normalisation
		self.alpha = nn.Parameter(torch.ones(self.size))
		self.bias = nn.Parameter(torch.zeros(self.size))
		self.eps = eps

	def forward(self, x):
		norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
		/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
		return norm


class EncoderLayer(nn.Module):
	def __init__(self, hid_dim, n_heads, pf_dim,  dropout, device):
		super().__init__()

		self.self_attn_layer_norm = Norm(hid_dim)
		self.ff_layer_norm = Norm(hid_dim)
		self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
		self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, src, src_mask):

		#src = [batch size, src len, hid dim]
		#src_mask = [batch size, 1, 1, src len]

		_src = self.self_attn_layer_norm(src)
		#self attention
		_src, _ = self.self_attention(_src, _src, _src, src_mask)
		#dropout, residual connection and layer norm
		src = src + self.dropout1(_src)
		#src = [batch size, src len, hid dim]

		_src = self.ff_layer_norm(src)
		#self attention
		_src = self.positionwise_feedforward(_src)
		#dropout, residual connection and layer norm
		src = src + self.dropout2(_src)

		return src


class Encoder(nn.Module):
	def __init__(self, hid_dim, embedding_layer, pos_layer, n_layers, n_heads, pf_dim, dropout, device):
		super().__init__()

		self.device = device

		self.tok_embedding = embedding_layer #nn.Embedding(input_dim, hid_dim)
		self.pos_embedding = pos_layer#nn.Embedding(max_length, hid_dim).to(device)

		self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

		self.dropout = nn.Dropout(dropout)

		self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
		self.norm = Norm(hid_dim)

	def forward(self, src, src_mask):

		#src = [batch size, src len]
		#src_mask = [batch size, 1, 1, src len]

		batch_size = src.shape[0]
		src_len = src.shape[1]

		pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
		#pos = [batch size, src len]

		# src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
		src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
		#src = [batch size, src len, hid dim]

		for layer in self.layers:
			src = layer(src, src_mask)
		#src = [batch size, src len, hid dim]

		return self.norm(src)


class DecoderLayer(nn.Module):
	def __init__(self,  hid_dim, n_heads, pf_dim, dropout, device):
		super().__init__()

		self.self_attn_layer_norm1 = Norm(hid_dim)
		self.self_attn_layer_norm2 = Norm(hid_dim)
		self.ff_layer_norm = Norm(hid_dim)
		self.self_attention1 = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
		self.self_attention2 = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
		self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,pf_dim, dropout)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)


	def forward(self, trg, enc_src, trg_mask, src_mask):

		#trg = [batch size, trg len, hid dim]
		#enc_src = list of #turns, [batch size, src len, hid dim]
		#trg_mask = [batch size, trg len]
		#src_mask = list of #turns, [batch size, src len]

		#SUB LAYER 1 -> MH SELF ATTENTION
		#self attention
		_trg = self.self_attn_layer_norm1(trg)
		_trg, _ = self.self_attention1(_trg, _trg, _trg, trg_mask)
		# [batch size, trg len, hid dim]
		trg = trg + self.dropout1(_trg)
		#trg = [batch size, trg len, hid dim]

		_trg = self.self_attn_layer_norm2(trg)
		_trg, _ = self.self_attention2(_trg, enc_src, enc_src, src_mask)
		trg = trg + self.dropout2(_trg)

		_trg = self.ff_layer_norm(trg)
		_trg = self.positionwise_feedforward(_trg)
		trg = trg + self.dropout3(_trg)

		return trg


class Decoder(nn.Module):
	def __init__(self, embedding_layer, pos_embedding, hid_dim,  n_layers, n_heads, pf_dim, dropout, device):
		super().__init__()

		self.device = device

		self.tok_embedding = embedding_layer
		self.pos_embedding = pos_embedding

		self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

		# self.fc_out = nn.Linear(hid_dim, output_dim)

		self.dropout = nn.Dropout(dropout)
		self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
		self.norm = Norm(hid_dim)


	def forward(self, trg, enc_src, trg_mask, src_mask):

		batch_size = trg.shape[0]
		trg_len = trg.shape[1]

		pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

		#pos = [batch size, trg len]

		trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
		# trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
		#trg = [batch size, trg len, hid dim]

		for layer in self.layers:
			trg = layer(trg, enc_src, trg_mask, src_mask)

		return self.norm(trg)


class EncoderPytorch(nn.Module):
	def __init__(self, hid_dim, n_heads, pf_dim, dropout, num_layers, embedding_layer, pos_layer, device):
		super().__init__()

		self.encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
		self.device = device

		self.tok_embedding = embedding_layer
		self.pos_embedding = pos_layer

		self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
		self.dropout = nn.Dropout(dropout)

	def forward(self, src, src_mask):
		# src = [batch size, src len]
		# src_mask = [batch size, src len]
		batch_size = src.shape[0]
		src_len = src.shape[1]

		pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
		# pos = [batch size, src len]

		src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
		src = src.permute(1, 0, 2)
		# src = [src len, batch size, hid dim]

		op = self.encoder(src, src_key_padding_mask=src_mask)
		return op


class DecoderPytorch(nn.Module):
	def __init__(self, hid_dim, n_heads, pf_dim, dropout, num_layers, embedding_layer, pos_layer, device):
		super().__init__()

		self.decoder_layer = nn.TransformerDecoderLayer(d_model=hid_dim, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout)
		self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
		self.device = device

		self.tok_embedding = embedding_layer
		self.pos_embedding = pos_layer

		self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
		self.dropout = nn.Dropout(dropout)

	def forward(self, trg, enc_src, trg_mask, src_mask):
		batch_size = trg.shape[0]
		trg_len = trg.shape[1]
		pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
		# pos = [batch size, trg len]

		trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
		trg = trg.permute(1, 0, 2)
		# trg = [trg len, batch size, hid dim]

		op = self.decoder(trg, memory=enc_src, tgt_key_padding_mask=trg_mask, memory_key_padding_mask=src_mask)
		return op.permute(1, 0, 2)


class PretrainedEncoder(nn.Module):
	def __init__(self, transformer, trainable, typ):
		super().__init__()

		self.transformer = transformer
		self.trainable = trainable
		self.type = typ

	def forward(self, input_ids, attn_mask):
		if not self.trainable:
			with torch.no_grad():
				op = self.transformer(input_ids=input_ids, attention_mask=attn_mask)
		else:
			op = self.transformer(input_ids=input_ids, attention_mask=attn_mask)

		hidden = torch.sum(torch.stack(op['hidden_states'][-4:]), 0)
		return hidden


class DoubleHeadClassification(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.out1 = nn.Linear(input_dim, output_dim)
		self.out2 = nn.Linear(input_dim, output_dim)

	def forward(self, inpt):
		op1 = self.out1(inpt)
		op2 = self.out2(inpt)
		return [op1, op2]


class SingleHeadClassification(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.out1 = nn.Linear(input_dim, output_dim)

	def forward(self, inpt):
		op1 = self.out1(inpt)
		return [op1]


class Driver(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.src_pad_idx = config['src_pad_idx']
		self.tgt_pad_idx = config['tgt_pad_idx']
		self.device = config['device']
		self.config = config
		self.encoder, embedding_layer, pos_layer = utils.get_encoder(self.config)
		self.decoder = utils.get_decoder(self.config, embedding_layer, pos_layer)
		self.out = utils.get_output_layer(self.config)

	def make_src_mask(self, src):
		# src = [batch size, src len]

		src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
		# src_mask = [batch size, 1, 1, src len]
		return src_mask

	def make_trg_mask(self, trg):
		# trg = [batch size, trg len]

		trg_pad_mask = (trg != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
		# trg_pad_mask = [batch size, 1, 1, trg len]

		trg_len = trg.shape[1]
		trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
		# trg_sub_mask = [trg len, trg len]

		trg_mask = trg_pad_mask & trg_sub_mask
		# trg_mask = [batch size, 1, trg len, trg len]
		return trg_mask

	def forward(self, src_inpt_ids, tgt_inpt_ids):

		src_attn_mask = self.make_src_mask(src_inpt_ids)
		tgt_attn_mask = self.make_trg_mask(tgt_inpt_ids)

		enc_src = self.encoder(src_inpt_ids, src_attn_mask)
		hidden = self.decoder(tgt_inpt_ids, enc_src, tgt_attn_mask, src_attn_mask)

		return self.out(hidden)

	def predict(self, src_inpt_ids, tgt_inpt_ids):
		src_attn_mask = self.make_src_mask(src_inpt_ids)
		enc_src = self.encoder(src_inpt_ids, src_attn_mask)
		# batch = src_inpt_ids.shape[0]
		trg_tensor = tgt_inpt_ids[:, :1]  # torch.tensor(self.config['tgt_sos'] * batch).unsqueeze(-1).to(self.device)
		output_trg = []
		for i in range(tgt_inpt_ids.shape[1]):
			trg_mask = self.make_trg_mask(trg_tensor)
			output_hidden = self.decoder(trg_tensor, enc_src, trg_mask, src_attn_mask)[:, -1:, :]
			outputs = self.out(output_hidden)[self.config['sample_index']]
			output_trg.append(outputs)
			if self.config['sample_type'] == 'argmax':
				pred_token = outputs.argmax(-1)
			else:
				_, pred_token = torch.sigmoid(outputs).max(-1)
			pred_token = pred_token.detach()
			trg_tensor = torch.cat([trg_tensor, pred_token], -1)

		output_trg = torch.cat(output_trg, 1)
		return [output_trg]

# class Driver(nn.Module):
# 	def __init__(self, config):
# 		super().__init__()
# 		self.src_pad_idx = config['src_pad_idx']
# 		self.tgt_pad_idx = config['tgt_pad_idx']
# 		self.device = config['device']
# 		self.config = config
# 		self.embedding_layer = utils.make_embedding_layer(config)
# 		self.pos_layer = utils.make_pos_layer(config)
# 		self.encoder = EncoderPytorch(config['hid_dim'], config['n_heads'], config['pf_dim'], config['dropout'], config['n_layers'], self.embedding_layer, self.pos_layer, self.device)
# 		self.decoder = DecoderPytorch(config['hid_dim'], config['n_heads'], config['pf_dim'], config['dropout'], config['n_layers'], self.embedding_layer, self.pos_layer, self.device)
# 		self.out = utils.get_output_layer(self.config)
#
# 	def forward(self, src_inpt_ids, tgt_inpt_ids):
#
# 		src_attn_mask = (src_inpt_ids != self.src_pad_idx)
# 		tgt_attn_mask = (tgt_inpt_ids != self.tgt_pad_idx)
#
# 		enc_src = self.encoder(src_inpt_ids, src_attn_mask)
# 		hidden = self.decoder(tgt_inpt_ids, enc_src, tgt_attn_mask, src_attn_mask)
#
# 		return self.out(hidden)
#
# 	def predict(self, src_inpt_ids, tgt_inpt_ids):
# 		src_attn_mask = (src_inpt_ids != self.src_pad_idx)
# 		enc_src = self.encoder(src_inpt_ids, src_attn_mask)
#
# 		trg_tensor = tgt_inpt_ids[:, :1]  # torch.tensor(self.config['tgt_sos'] * batch).unsqueeze(-1).to(self.device)
# 		output_trg = []
# 		for i in range(tgt_inpt_ids.shape[1]):
# 			trg_mask = (trg_tensor != self.tgt_pad_idx)
# 			output_hidden = self.decoder(trg_tensor, enc_src, trg_mask, src_attn_mask)[:, -1:, :]
# 			outputs = self.out(output_hidden)[self.config['sample_index']]
# 			output_trg.append(outputs)
# 			if self.config['sample_type'] == 'argmax':
# 				pred_token = outputs.argmax(-1)
# 			else:
# 				_, pred_token = torch.sigmoid(outputs).max(-1)
# 			pred_token = pred_token.detach()
# 			trg_tensor = torch.cat([trg_tensor, pred_token], -1)
#
# 		output_trg = torch.cat(output_trg, 1)
# 		return [output_trg]
#
