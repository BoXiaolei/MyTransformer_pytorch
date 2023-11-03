# MyTransformer_pytorch
关于Transformer模型的最简洁pytorch实现，包含详细注释

> 本实现版本相比参考代码删去了每个模块不必要的返回（如注意力矩阵），力求最精简明晰的实现，适用于初学者入门学习

- 参考代码有：
  1. https://wmathor.com/index.php/archives/1455/
  2. http://nlp.seas.harvard.edu/annotated-transformer/ (哈佛NLP团队实现版本)


- file_list
  - data.py
    - 数据预处理
  - model.py
    - 模型文件
  - train.py
    - 训练模型
  - test.py
    - 读入训练好的pth模型文件，测试模型，完成一个翻译任务
  - .pth
    - My_Transformer.pth  
      - 是按照原concat写法训练1000次后得到的模型，Loss约为3e-6
    - My_Transformer_concat.pth
      - 是按照我修改后的concat写法训练1000次后得到的模型，Loss也为3e-6
    - MyTransformer_fault.pth
      - 只训练了5个epoch的模型，用于验证所做的测试是有意义的（用此模型预测会出错）
