from torch import optim

from model import *


model = Transformer().cuda()
model.train()
# 损失函数,忽略为0的类别不对其计算loss（因为是padding无意义）
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# 训练开始
for epoch in range(1000):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        '''
        enc_inputs: [batch_size, src_len] [2,5]
        dec_inputs: [batch_size, tgt_len] [2,6]
        dec_outputs: [batch_size, tgt_len] [2,6]
        '''
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
        outputs = model(enc_inputs, dec_inputs) # outputs: [batch_size * tgt_len, tgt_vocab_size]
        # outputs: [batch_size * tgt_len, tgt_vocab_size], dec_outputs: [batch_size, tgt_len]
        loss = criterion(outputs, dec_outputs.view(-1))  # 将dec_outputs展平成一维张量

        # 更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item()}')

torch.save(model, f'MyTransformer_temp.pth')

