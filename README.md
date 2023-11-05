# MyTransformer_pytorch
关于Transformer模型的最简洁pytorch实现，包含详细注释

> 本实现版本相比参考代码删去了每个模块不必要的返回（如注意力矩阵），力求最精简明晰的实现，适用于初学者入门学习

- 参考代码有：
  1. https://wmathor.com/index.php/archives/1455/
  2. http://nlp.seas.harvard.edu/annotated-transformer/ (哈佛NLP团队实现版本)


- file_list
  - MyTransformer.ipynb
    - jupyter notebook中的实现，与.py文件相比，添加了更多的说明文字
  - images
    - 为方便理解绘制的一些图，在 MyTransformer.ipynb 中被用到
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
     
- 训练好的模型文件链接：
  - 链接：https://pan.baidu.com/s/133Ud8f0yHV3kFnawdZGsQA 
  - 提取码：2022
- 下载后解压到项目根目录下即可

- 我的邮箱：crazystone_lei@163.com
  - 欢迎来信交流
  
- 以上内容均为原创，参考的代码已列出，如要转载请注明出处，best wishes.
