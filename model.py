from data import *
import torch
import numpy as np
import torch.nn as nn


# 用来表示一个词的向量长度
d_model = 512
# FFN的隐藏层神经元个数
d_ff = 2048
# 分头后的q、k、v词向量长度，依照原文我们都设为64
# 原文：queries and kes of dimention d_k,and values of dimension d_v .所以q和k的长度都用d_k来表示
d_k = d_v = 64
# Encoder Layer 和 Decoder Layer的个数
n_layers = 6
# 多头注意力中head的个数，原文：we employ h = 8 parallel attention layers, or heads
n_heads = 8


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # dropout是原文的0.1，max_len原文没找到
        '''max_len是假设的一个句子最多包含5000个token,即max_seq_len'''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 开始位置编码部分,先生成一个max_len * d_model 的tensor，即5000 * 512
        # 5000是一个句子中最多的token数，512是一个token用多长的向量来表示，5000*512这个矩阵用于表示一个句子的信息
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # pos的shape为[max_len,1],即[5000,1]
        # 先把括号内的分式求出来,pos是[5000,1],分母是[256],通过广播机制相乘后是[5000,256]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        # 再取正余弦
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        # 一个句子要做一次pe，一个batch中会有多个句子，所以增加一维来用和输入的一个batch的数据相加时做广播
        pe = pe.unsqueeze(0) # [5000,512] -> [1,5000,512]
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        ''' x: [batch_size, seq_len, d_model] '''
        # 5000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        x = x + self.pe[:, :x.size(1), :] # 加的时候应该也广播了，第一维 1 -> batch_size
        return self.dropout(x) # return: [batch_size, seq_len, d_model], 和输入的形状相同



# 将输入序列中的占位符P的token（就是0） mask掉，用于计算注意力
# 返回一个[batch_size, len_q, len_k]大小的布尔张量，True是需要mask掉的位置
def get_attn_pad_mask(seq_q, seq_k):
    # len_q、len_k其实是q的length和k的length，q和k都是一个序列即一个句子，长度即句子中包含的词的数量
    batch_size, len_q = seq_q.size() # 获取作为q的序列（句子）长度
    batch_size, len_k = seq_k.size() # 获取作为k的序列长度
    # seq_k.data.eq(0)返回一个和seq_k等大的布尔张量，seq_k元素等于0的位置为True,否则为False
    # 然后扩维以保证后续操作的兼容(广播)
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # pad_attn_mask: [batch_size,1,len_k]
    # 要为每一个q提供一份k，所以把第二维度扩展了q次，这样只有最后一列是True，正常来说最后一行也需要是True，但是由于作为padding的token对其他词的注意力不重要，所以可以这样写
    res = pad_attn_mask.expand(batch_size, len_q, len_k)
    return res # return: [batch_size, len_q, len_k]
    # 返回的是batch_size个 len_q * len_k的矩阵，内容是True和False，第i行第j列表示的是query的第i个词对key的第j个词的注意力是否无意义，若无意义则为True，有意义的为False（即被padding的位置是True）



# 用于获取对后续位置的掩码，防止在预测过程中看到未来时刻的输入
# 原文：to prevent positions from attending to subsequent positions
def get_attn_subsequence_mask(seq):
    """seq: [batch_size, tgt_len]"""
    # batch_size个 tgt_len * tgt_len的mask矩阵
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # np.triu 是生成一个 upper triangular matrix 上三角矩阵，k是相对于主对角线的偏移量
    # k=1意为不包含主对角线（从主对角线向上偏移1开始）
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte() # 因为只有0、1所以用byte节省内存
    return subsequence_mask
    # return: [batch_size, tgt_len, tgt_len]


class ScaledDotProductionAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductionAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v] 全文两处用到注意力，一处是self attention，另一处是co attention，前者不必说，后者的k和v都是encoder的输出，所以k和v的形状总是相同的
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # 1) 计算注意力分数QK^T/sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores: [batch_size, n_heads, len_q, len_k]
        # 2)  进行 mask 和 softmax
        # mask为True的位置会被设为-1e9
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # attn: [batch_size, n_heads, len_q, len_k]
        # 3) 乘V得到最终的加权和
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        '''
        得出的context是每个维度(d_1-d_v)都考虑了在当前维度(这一列)当前token对所有token的注意力后更新的新的值，
        换言之每个维度d是相互独立的，每个维度考虑自己的所有token的注意力，所以可以理解成1列扩展到多列

        返回的context: [batch_size, n_heads, len_q, d_v]本质上还是batch_size个句子，
        只不过每个句子中词向量维度512被分成了8个部分，分别由8个头各自看一部分，每个头算的是整个句子(一列)的512/8=64个维度，最后按列拼接起来
        '''
        return context # context: [batch_size, n_heads, len_q, d_v]



class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model] len_q是作为query的句子的长度，比如enc_inputs（2,5,512）作为输入，那句子长度5就是len_q
        input_K: [batch_size, len_k, d_model]
        input_K: [batch_size, len_v(len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)

        # 1）linear projection [batch_size, seq_len, d_model] ->  [batch_size, n_heads, seq_len, d_k/d_v]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2) # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # 2）计算注意力
        # 自我复制n_heads次，为每个头准备一份mask
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductionAttention()(Q, K, V, attn_mask) # context: [batch_size, n_heads, len_q, d_v]

        # 3）concat部分
        context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
        output = self.concat(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual)  # output: [batch_size, len_q, d_model]

        '''        
        最后的concat部分，网上的大部分实现都采用的是下面这种方式（也是哈佛NLP团队的写法），但是我发现下面这种方式拼回去会使原来的位置乱序，于是并未采用这种写法，实验效果是相近的
        context = context.transpose(1, 2).reshape(batch_size, -1, d_model)
        output = self.linear(context)
        '''


# 这部分代码很简单，对应模型图中的 Feed Forward和 Add & Norm
class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        # 就是一个MLP
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, inputs):
        '''inputs: [batch_size, seq_len, d_model]'''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual) # return： [batch_size, seq_len, d_model] 形状不变


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # Q、K、V均为 enc_inputs
        enc_ouputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_ouputs: [batch_size, src_len, d_model]
        enc_ouputs = self.pos_ffn(enc_ouputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_ouputs  # enc_outputs: [batch_size, src_len, d_model]


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 直接调的现成接口完成词向量的编码，输入是类别数和每一个类别要映射成的向量长度
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''enc_inputs: [batch_size, src_len]'''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len] -> [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        # Encoder中是self attention，所以传入的Q、K都是enc_inputs
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # enc_self_attn_mask: [batch_size, src_len, src_len]
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs  # enc_outputs: [batch_size, src_len, d_model]


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len] 前者是Q后者是K
        '''
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs # dec_outputs: [batch_size, tgt_len, d_model]




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])


    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        这三个参数对应的不是Q、K、V，dec_inputs是Q，enc_outputs是K和V，enc_inputs是用来计算padding mask的
        dec_inputs: [batch_size, tgt_len]
        enc_inpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs).cuda()
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        # 将两个mask叠加，布尔值可以视为0和1，和大于0的位置是需要被mask掉的，赋为True，和为0的位置是有意义的为False
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).cuda()
        # 这是co-attention部分，为啥传入的是enc_inputs而不是enc_outputs呢
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)

        return dec_outputs # dec_outputs: [batch_size, tgt_len, d_model]


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        # 解散batch，一个batch中有batch_size个句子，每个句子有tgt_len个词（即tgt_len行），现在让他们按行依次排布，如前tgt_len行是第一个句子的每个词的预测概率，再往下tgt_len行是第二个句子的，一直到batch_size * tgt_len行
        return dec_logits.view(-1, dec_logits.size(-1))  #  [batch_size * tgt_len, tgt_vocab_size]
        '''最后变形的原因是：nn.CrossEntropyLoss接收的输入的第二个维度必须是类别'''

