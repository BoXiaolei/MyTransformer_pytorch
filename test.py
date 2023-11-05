import torch

from model import *

# 原文使用的是大小为4的beam search，这里为简单起见使用更简单的greedy贪心策略生成预测，不考虑候选，每一步选择概率最大的作为输出
# 如果不使用greedy_decoder，那么我们之前实现的model只会进行一次预测得到['i']，并不会自回归，所以我们利用编写好的Encoder-Decoder来手动实现自回归（把上一次Decoder的输出作为下一次的输入，直到预测出终止符）
def greedy_decoder(model, enc_input, start_symbol):
    """enc_input: [1, seq_len] 对应一句话"""
    enc_outputs = model.encoder(enc_input) # enc_outputs: [1, seq_len, 512]
    # 生成一个1行0列的，和enc_inputs.data类型相同的空张量，待后续填充
    dec_input = torch.zeros(1, 0).type_as(enc_input.data) # .data避免影响梯度信息
    next_symbol = start_symbol
    flag = True
    while flag:
        # dec_input.detach() 创建 dec_input 的一个分离副本
        # 生成了一个 只含有next_symbol的（1,1）的张量
        # -1 表示在最后一个维度上进行拼接cat
        # 这行代码的作用是将next_symbol拼接到dec_input中，作为新一轮decoder的输入
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).cuda()], -1) # dec_input: [1,当前词数]
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs) # dec_outputs: [1, tgt_len, d_model]
        projected = model.projection(dec_outputs) # projected: [1, 当前生成的tgt_len, tgt_vocab_size]
        # max返回的是一个元组（最大值，最大值对应的索引），所以用[1]取到最大值对应的索引, 索引就是类别，即预测出的下一个词
        # keepdim为False会导致减少一维
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1] # prob: [1],
        # prob是一个一维的列表，包含目前为止依次生成的词的索引，最后一个是新生成的（即下一个词的类别）
        # 因为注意力是依照前面的词算出来的，所以后生成的不会改变之前生成的
        next_symbol = prob.data[-1]
        if next_symbol == tgt_vocab['.']:
            flag = False
        print(next_symbol)
    return dec_input  # dec_input: [1,tgt_len]


# 测试
model = torch.load('MyTransformer_temp.pth')
model.eval()
with torch.no_grad():
    # 手动从loader中取一个batch的数据
    enc_inputs, _, _ = next(iter(loader))
    enc_inputs = enc_inputs.cuda()
    for i in range(len(enc_inputs)):
        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab['S'])
        predict  = model(enc_inputs[i].view(1, -1), greedy_dec_input) # predict: [batch_size * tgt_len, tgt_vocab_size]
        predict = predict.data.max(dim=-1, keepdim=False)[1]
        '''greedy_dec_input是基于贪婪策略生成的，而贪婪解码的输出是基于当前时间步生成的假设的输出。这意味着它可能不是最优的输出，因为它仅考虑了每个时间步的最有可能的单词，而没有考虑全局上下文。
        因此，为了获得更好的性能评估，通常会将整个输入序列和之前的假设输出序列传递给模型，以考虑全局上下文并允许模型更准确地生成输出
        '''
        print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict])



