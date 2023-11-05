import torch
import torch.utils.data as Data


# S: decoding input 的起始符
# E: decoding output 的结束符
# P：意为padding，如果当前句子短于本batch的最长句子，那么用这个符号填补缺失的单词
sentence = [
    # enc_input   dec_input    dec_output
    ['ich mochte ein bier P','S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P','S i want a coke .', 'i want a coke . E'],
]

# 词典，padding用0来表示
# 源词典，本例中即德语词典
src_vocab = {'P':0, 'ich':1,'mochte':2,'ein':3,'bier':4,'cola':5}
src_vocab_size = len(src_vocab) # 6
# 目标词典，本例中即英语词典,相比源多了特殊符
tgt_vocab = {'P':0,'i':1,'want':2,'a':3,'beer':4,'coke':5,'S':6,'E':7,'.':8}
# 反向映射词典，idx —— word，原代码那个有点不好理解
idx2word = {v:k for k,v in tgt_vocab.items()}
tgt_vocab_size = len(tgt_vocab) # 9

src_len = 5 # 输入序列enc_input的最长序列长度，其实就是最长的那句话的token数，是指一个batch中最长呢还是所有输入数据最长呢
tgt_len = 6 # 输出序列dec_inut/dec_output的最长序列长度

# 构建模型输入的Tensor
def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentence)):
        enc_input = [src_vocab[word] for word in sentence[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentence[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentence[i][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    # LongTensor是专用于存储整型的，Tensor则可以存浮点、整数、bool等多种类型
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
    # 返回的形状为 enc_inputs：（2,5）、dec_inputs（2,6）、dec_outputs（2,6）


# 使用Dataset加载数据
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        # 我们前面的enc_inputs.shape = [2,5],所以这个返回的是2
        return self.enc_inputs.shape[0]

    # 根据idx返回的是一组 enc_input, dec_input, dec_output
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# 获取输入
enc_inputs, dec_inputs, dec_outputs = make_data(sentence)

# 构建DataLoader
loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)
