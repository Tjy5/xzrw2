# 导入一些必要的库
import pyaudio
import wave
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import jiwer
import streamlit as st

# 定义一些参数
CHUNK = 1024 # 每次读取的语音帧数
FORMAT = pyaudio.paInt16 # 语音格式
CHANNELS = 1 # 语音通道数
RATE = 16000 # 语音采样率
RECORD_SECONDS = 5 # 录音时长
WAVE_FILE = "1.wav" # 保存录音的文件名
MODEL_FILE = "model.pth" # 保存模型的文件名
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 使用的设备
VOCAB = ["<sos>", "<eos>", "<pad>", "<blank>"] + [chr(i) for i in range(ord('a'), ord('z')+1)] # 词汇表
VOCAB_SIZE = len(VOCAB) # 词汇表大小
HIDDEN_SIZE = 256 # 隐藏层大小
NUM_LAYERS = 3 # 层数
DROPOUT = 0.1 # Dropout概率
LEARNING_RATE = 0.001 # 学习率
BATCH_SIZE = 32 # 批大小
EPOCHS = 10 # 训练轮数
BEAM_WIDTH = 3 # Beam Search的宽度

# 定义一个函数，用于录制语音
def record_audio():
    # 创建一个PyAudio对象
    p = pyaudio.PyAudio()
    # 打开一个流，用于录制语音
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    # 提示用户开始录音
    print("* recording")
    # 创建一个空列表，用于存储语音帧
    frames = []
    # 循环读取语音帧，直到达到指定的录音时长
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # 读取一帧语音
        data = stream.read(CHUNK)
        # 将语音帧添加到列表中
        frames.append(data)
    # 提示用户结束录音
    print("* done recording")
    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    # 终止PyAudio对象
    p.terminate()
    # 将语音帧列表转换为字节串
    audio_data = b''.join(frames)
    # 返回录制的语音数据
    return audio_data

# 定义一个函数，用于保存语音到文件
def save_audio(audio_data, file_name):
    # 创建一个wave文件对象
    wf = wave.open(file_name, "wb")
    # 设置文件的参数
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    # 写入语音数据
    wf.writeframes(audio_data)
    # 关闭文件对象
    wf.close()

# 定义一个函数，用于从文件读取语音
def load_audio(file_name):
    # 打开一个wave文件对象
    wf = wave.open(file_name, "rb")
    # 读取语音数据
    audio_data = wf.readframes(wf.getnframes())
    # 关闭文件对象
    wf.close()
    # 返回读取的语音数据
    return audio_data

# 定义一个函数，用于播放语音
def play_audio(audio_data):
    # 创建一个PyAudio对象
    p = pyaudio.PyAudio()
    # 打开一个流，用于播放语音
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)
    # 提示用户开始播放
    print("* playing")
    # 循环播放语音帧
    for i in range(0, len(audio_data), CHUNK):
        # 播放一帧语音
        stream.write(audio_data[i:i+CHUNK])
    # 提示用户结束播放
    print("* done playing")
    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    # 终止PyAudio对象
    p.terminate()

# 定义一个函数，用于将语音数据转换为numpy数组
def audio_to_array(audio_data):
    # 将字节串转换为整数数组
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    # 返回语音数组
    return audio_array

# 定义一个函数，用于将numpy数组转换为语音数据
def array_to_audio(audio_array):
    # 将整数数组转换为字节串
    audio_data = audio_array.astype(np.int16).tobytes()
    # 返回语音数据
    return audio_data

# 定义一个函数，用于提取语音特征
def extract_feature(audio_array):
    # TODO: 实现你的语音特征提取方法，比如MFCC、FBANK等
    # 这里只是简单地将语音数组转换为浮点数，并归一化到[-1, 1]之间
    feature = audio_array.astype(np.float32) / 32768
    # 返回语音特征
    return feature

# 定义一个函数，用于将文本转换为索引序列
def text_to_index(text):
    # 初始化一个空列表，用于存储索引
    index = []
    # 在文本前后添加开始和结束标记
    text = "<sos>" + text + "<eos>"
    # 遍历文本中的每个字符
    for char in text:
        # 如果字符在词汇表中，将其对应的索引添加到列表中
        if char in VOCAB:
            index.append(VOCAB.index(char))
        # 否则，跳过该字符
        else:
            continue
    # 返回索引序列
    return index

# 定义一个函数，用于将索引序列转换为文本
def index_to_text(index):
    # 初始化一个空字符串，用于存储文本
    text = ""
    # 遍历索引序列中的每个索引
    for i in index:
        # 如果索引在词汇表的范围内，将其对应的字符添加到字符串中
        if i < VOCAB_SIZE:
            text += VOCAB[i]
        # 否则，跳过该索引
        else:
            continue
    # 返回文本
    return text

# 定义一个端到端的语音识别模型，这里使用了双向GRU作为编码器，注意力机制和GRU作为解码器
class ASRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(ASRModel, self).__init__()
        # 定义编码器，使用双向GRU
        self.encoder = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        # 定义解码器，使用单向GRU
        self.decoder = nn.GRU(input_size=hidden_size * 2 + vocab_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        # 定义注意力机制，使用点积注意力
        self.attention = nn.Linear(hidden_size * 3, 1)
        # 定义输出层，使用线性变换
        self.output = nn.Linear(hidden_size, vocab_size)
        # 定义softmax层，用于计算概率分布
        self.softmax = nn.Softmax(dim=-1)
        # 定义词嵌入层，用于将索引转换为向量
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        # 初始化词嵌入层的权重为单位矩阵，相当于one-hot编码
        self.embedding.weight.data = torch.eye(vocab_size)
        # 固定词嵌入层的权重，不参与训练
        self.embedding.weight.requires_grad = False

    def forward(self, encoder_input, decoder_input):
        # encoder_input: (batch_size, input_length, 1)
        # decoder_input: (batch_size, output_length)
        # 对编码器输入进行编码，得到编码器输出和最后一个隐藏状态
        # encoder_output: (batch_size, input_length, hidden_size * 2)
        # encoder_hidden: (num_layers * 2, batch_size, hidden_size)
        encoder_output, encoder_hidden = self.encoder(encoder_input)
        # 将编码器的最后一个隐藏状态的两个方向拼接起来，作为解码器的初始隐藏状态
        # decoder_hidden: (num_layers, batch_size, hidden_size)
        decoder_hidden = torch.cat([encoder_hidden[-2], encoder_hidden[-1]], dim=-1).unsqueeze(0)
        # 对解码器输入进行词嵌入，得到解码器输入向量
        # decoder_input: (batch_size, output_length, vocab_size)
        decoder_input = self.embedding(decoder_input)
        # 初始化一个空张量，用于存储解码器输出
        # decoder_output: (batch_size, output_length, vocab_size)
        decoder_output = torch.zeros(decoder_input.size(0), decoder_input.size(1), VOCAB_SIZE, device=DEVICE)
        # 遍历解码器的每个时间步
        for t in range(decoder_input.size(1)):
            # 取出当前时间步的解码器输入向量
            # current_input: (batch_size, 1, vocab_size)
            current_input = decoder_input[:, t, :].unsqueeze(1)
            # 将解码器的当前隐藏状态和编码器输出进行拼接
            # hidden_output: (batch_size, input_length, hidden_size * 3)
            hidden_output = torch.cat([decoder_hidden[-1].unsqueeze(1).expand(-1, encoder_output.size(1), -1), encoder_output], dim=-1)
            # 计算注意力权重，使用softmax归一化
            # attention_weight: (batch_size, input_length, 1)
            attention_weight = self.softmax(self.attention(hidden_output))
            # 根据注意力权重和编码器输出，计算上下文向量
            # context_vector: (batch_size, 1, hidden_size * 2)
            context_vector = torch.bmm(attention_weight.transpose(1, 2), encoder_output)
            # 将当前输入向量和上下文向量进行拼接，作为解码器的当前输入
            # current_input: (batch_size, 1, hidden_size * 2 + vocab_size)
            current_input = torch.cat([current_input, context_vector], dim=-1)
            # 对解码器的当前输入进行解码，得到解码器的当前输出和隐藏状态
            # current_output: (batch_size, 1, hidden_size)
            # decoder_hidden: (num_layers, batch_size, hidden_size)
            current_output, decoder_hidden = self.decoder(current_input, decoder_hidden)
            # 对解码器的当前输出进行线性变换，得到输出的概率分布
            # current_output: (batch_size, 1, vocab_size)
            current_output = self.output(current_output)
            # 将当前输出的概率分布存储到解码器输出张量中
            decoder_output[:, t, :] = current_output.squeeze(1)
        # 返回解码器输出
        return decoder_output

# 定义一个函数，用于训练模型，使用CTC损失函数和Adam优化器
def train_model(model, data_loader, epochs):
    # 将模型移动到指定的设备
    model.to(DEVICE)
    # 将模型设置为训练模式
    model.train()
    # 定义CTC损失函数，忽略填充索引
    criterion = nn.CTCLoss(blank=VOCAB.index("<blank>"), zero_infinity=True)
    # 定义Adam优化器，使用默认的超参数
    optimizer = optim.Adam(model.parameters())
    # 遍历训练轮数
    for epoch in range(epochs):
        # 初始化一个变量，用于存储累计的损失值
        total_loss = 0
        # 遍历数据加载器，获取每个批次的数据
        for batch in data_loader:
            # 获取特征和标签，分别对应语音和文本
            # feature: (batch_size, input_length, 1)
            # label: (batch_size, output_length)
            feature, label = batch
            # 将特征和标签移动到指定的设备
            feature = feature.to(DEVICE)
            label = label.to(DEVICE)
            # 清空优化器的梯度
            optimizer.zero_grad()
            # 用模型对特征进行预测，得到输出
            # output: (batch_size, output_length, vocab_size)
            output = model(feature, label)
            # 计算输出的对数概率，用于CTC损失函数
            # output_log_prob: (input_length, batch_size, vocab_size)
            output_log_prob = output.log_softmax(2).transpose(0, 1)
            # 计算标签的长度，用于CTC损失函数
            # label_length: (batch_size,)
            label_length = torch.sum(label != VOCAB.index("<pad>"), dim=-1)
            # 计算输出的长度，用于CTC损失函数，这里假设输出的长度等于输入的长度
            # output_length: (batch_size,)
            output_length = torch.full(label_length.size(), output_log_prob.size(0), dtype=torch.long, device=DEVICE)
            # 计算CTC损失函数
            loss = criterion(output_log_prob, label, output_length, label_length)
            # 反向传播，计算梯度
            loss.backward()
            # 使用优化器进行参数更新
            optimizer.step()
            # 累计损失值
            total_loss += loss.item()

        # 打印每个epoch的平均损失值
        avg_loss = total_loss / len(data_loader)


# 定义一个函数，用于解码模型的输出，使用贪心搜索算法
def greedy_decode(model, feature):
    # 将模型移动到指定的设备
    model.to(DEVICE)
    # 将模型设置为评估模式
    model.eval()
    # 将特征移动到指定的设备
    feature = feature.to(DEVICE)
    # 初始化一个空列表，用于存储索引
    index = []
    # 初始化一个开始标记，用于作为解码器的第一个输入
    # current_input: (1, 1)
    current_input = torch.LongTensor([[VOCAB.index("<sos>")]], device=DEVICE)
    # 初始化一个隐藏状态，用于作为解码器的初始隐藏状态
    # hidden_state: (num_layers, 1, hidden_size)
    hidden_state = None
    # 对特征进行编码，得到编码器输出和最后一个隐藏状态
    # encoder_output: (1, input_length, hidden_size * 2)
    # encoder_hidden: (num_layers * 2, 1, hidden_size)
    encoder_output, encoder_hidden = model.encoder(feature.unsqueeze(0))
    # 将编码器的最后一个隐藏状态的两个方向拼接起来，作为解码器的初始隐藏状态
    # hidden_state: (num_layers, 1, hidden_size)
    hidden_state = torch.cat([encoder_hidden[-2], encoder_hidden[-1]], dim=-1).unsqueeze(0)
    # 循环解码，直到遇到结束标记或达到最大长度
    while True:
        # 对当前输入进行词嵌入，得到当前输入向量
        # current_input: (1, 1, vocab_size)
        current_input = model.embedding(current_input)
        # 将解码器的当前隐藏状态和编码器输出进行拼接
        # hidden_output: (1, input_length, hidden_size * 3)
        hidden_output = torch.cat([hidden_state[-1].unsqueeze(1).expand(-1, encoder_output.size(1), -1), encoder_output], dim=-1)
        # 计算注意力权重，使用softmax归一化
        # attention_weight: (1, input_length, 1)
        attention_weight = model.softmax(model.attention(hidden_output))
        # 根据注意力权重和编码器输出，计算上下文向量
        # context_vector: (1, 1, hidden_size * 2)
        context_vector = torch.bmm(attention_weight.transpose(1, 2), encoder_output)
        # 将当前输入向量和上下文向量进行拼接，作为解码器的当前输入
        # current_input: (1, 1, hidden_size * 2 + vocab_size)
        current_input = torch.cat([current_input, context_vector], dim=-1)
        # 对解码器的当前输入进行解码，得到解码器的当前输出和隐藏状态
        # current_output: (1, 1, hidden_size)
        # hidden_state: (num_layers, 1, hidden_size)
        current_output, hidden_state = model.decoder(current_input, hidden_state)
        # 对解码器的当前输出进行线性变换，得到输出的概率分布
        # current_output: (1, 1, vocab_size)
        current_output = model.output(current_output)
        # 对输出的概率分布进行softmax归一化，得到概率最大的索引
        # current_output: (1, 1)
        current_output = current_output.softmax(-1).argmax(-1)
        # 将当前输出的索引添加到列表中
        index.append(current_output.item())
        # 如果当前输出的索引是结束标记，或者列表的长度达到了最大长度，停止解码
        if current_output.item() == VOCAB.index("<eos>") or len(index) >= 100:
            break
        # 将当前输出的索引作为下一个输入的索引
        current_input = current_output
    # 返回索引序列
    return index

# 定义一个函数，用于评估模型的性能，使用WER指标
def evaluate_model(model, data_loader):
    # 初始化一个变量，用于存储累计的WER值
    total_wer = 0
    # 初始化一个变量，用于存储样本的数量
    sample_count = 0
    # 遍历数据加载器，获取每个批次的数据
    for batch in data_loader:
        # 获取特征和标签，分别对应语音和文本
        # feature: (batch_size, input_length, 1)
        # label: (batch_size, output_length)
        feature, label = batch
        # 遍历每个样本
        for i in range(feature.size(0)):
            # 取出当前样本的特征和标签
            # current_feature: (input_length, 1)
            # current_label: (output_length,)
            current_feature = feature[i]
            current_label = label[i]
            # 用模型对当前特征进行解码，得到索引序列
            # current_index: (output_length,)
            current_index = greedy_decode(model, current_feature)
            # 将索引序列转换为文本
            # current_text: str
            current_text = index_to_text(current_index)
            # 将标签转换为文本
            # current_label: str
            current_label = index_to_text(current_label.tolist())
            # 计算当前文本和标签的WER值
            # current_wer: float
            current_wer = jiwer.wer(current_label, current_text)
            # 将当前的WER值累加到总的WER值中
            total_wer += current_wer
            # 将样本的数量加一
            sample_count += 1
    # 计算平均的WER值
    # average_wer: float
    average_wer = total_wer / sample_count
    # 返回平均的WER值
    return average_wer
# 定义一个函数，用于优化模型的性能，使用数据增强方法
def augment_data(feature, label):
    # TODO: 实现你的数据增强方法，比如添加噪声、变速、变调等
    # 这里只是简单地对语音特征进行随机裁剪，模拟不同长度的语音输入
    # feature: (input_length, 1)
    # label: (output_length,)
    # 生成一个随机的裁剪比例，介于0.8到1之间
    ratio = np.random.uniform(0.8, 1)
    # 根据裁剪比例，计算裁剪后的长度
    length = int(feature.size(0) * ratio)
    # 在语音特征的前后随机选择一个裁剪的起点
    start = np.random.randint(0, feature.size(0) - length)
    # 从起点开始，裁剪出指定长度的语音特征
    # feature: (length, 1)
    feature = feature[start:start+length]
    # 返回裁剪后的语音特征和原始的标签
    return feature, label
