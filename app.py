from flask import Flask, render_template,request, url_for, redirect, flash
import numpy as np
import os
from datetime import timedelta


# class NameForm(FlaskForm):
#     sentence=StringField("请输入英文句子：",validators=[DataRequired()])
#     submit=SubmitField("提交")

app = Flask(__name__)#实例化flask类
app.debug=True
app.config["SECRET_KEY"]=os.urandom(24)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

class HMM():
    def __init__(self):
        #词典，将每个词性和单词都赋予id，便于查表
        self.id2tag, self.tag2id,self.word2id, self.id2word={},{},{},{}

        self.tagNum=0 #词性数量
        self.wordNum=0  #单词数量
        self.A = None    #初始矩阵
        self.B = None   #转移矩阵
        self.C = None   #发射矩阵

    def train(self, file):
        #将矩阵初始化
        for line in open(file, 'r', encoding='utf-8'):
            line = line.split("/")
            if len(line) != 2:
                continue
            word, tag = line[0].lower(), line[1].rstrip()   #单词全部转为小写，tag去除换行符
            tag=tag.split("|")[0]   #标注了两个词性的话，取第一个

            # #合并相似功能的符号
            # if tag in [",", ".", ":", "``", "''"]:
            #     tag="END"#特殊的符号，视作一句话的终止，tag标为空。
            # elif tag in ["$","#","SYM","(", ")"]:
            #     tag="W"#句中符号，当做普通标点。

            if word not in self.word2id:#建立单词词典
                self.word2id[word] = len(self.word2id)
                self.id2word[len(self.id2word)] = word

            if tag not in self.tag2id:#建立词性词典
                self.tag2id[tag] = len(self.tag2id)
                self.id2tag[len(self.id2tag)] = tag

        self.tagNum = len(self.tag2id)#词性数量
        self.wordNum = len(self.word2id)#单词数量
        print("tag size: {}".format(self.tagNum))
        print("word size: {}".format(self.wordNum))

        self.A = np.zeros(self.tagNum)#初始矩阵初始化，以词性ID为索引
        self.B = np.zeros((self.tagNum, self.tagNum))#转移矩阵初始化，以词性ID为索引
        self.C = np.zeros((self.tagNum, self.wordNum))#发射矩阵初始化，以词性和单词的ID为索引

        #训练模型
        pre_tag = "END"#上一个单词的词性，以建立隐马尔科夫模型
        for line in open(file, 'r', encoding='utf-8'):
            line = line.split('/')
            if len(line) != 2:
                continue
            word, tag = line[0].lower(), line[1].rstrip()
            tag = tag.split("|")[0]

            # #合并相似功能符号
            # if tag in [",", ".", ":", "``", "''"]:
            #     tag="END"#特殊的符号，视作一句话的终止，tag标为空。
            # elif tag in ["$","#","SYM","(", ")"]:
            #     tag="W"#句中符号，当做普通标点。

            word_id, tag_id = self.word2id[word], self.tag2id[tag]#获取当前单词和词性

            if pre_tag == "END":#句首
                self.A[tag_id] += 1#统计初始矩阵
                self.C[tag_id][word_id] += 1#统计发射矩阵
            else:#句中
                self.B[self.tag2id[pre_tag]][tag_id] += 1#统计转移矩阵
                self.C[tag_id][word_id] += 1#统计发射矩阵
            pre_tag = tag#更新上一个单词的词性

        # 转成概率
        self.A = self.A / sum(self.A)   # 初始矩阵
        for i in range(self.tagNum):  # 转移矩阵、发射矩阵，按词性遍历
            if self.id2tag[i]!="END":   # 除去句末终止符号
                self.B[i] /= sum(self.B[i])
            self.C[i] /= sum(self.C[i])

    def viterbi(self, sentence):
        sentence=sentence.strip()
        print(sentence)
        words=[]
        for word in sentence.split(" "):
            if word.lower() in self.word2id:
                words.append(self.word2id[word.lower()])
            else:
                return False #包含生词 无法识别
        #words = [self.word2id[word.lower()] for word in sentence.split(" ")]
        length = len(words)

        dp = np.zeros((length, self.tagNum))#存储分数，行数=句子长度，列数=词性
        pre = np.zeros((length, self.tagNum), dtype=int)#存储上一层的选择

        #第一个单词
        for j in range(self.tagNum):#遍历词性
            dp[0][j] = self.A[j]*self.C[j][words[0]]#发射矩阵x转移矩阵
        #第二个单词起
        #i是单词
        for i in range(1, length):
            # j为当前层的词性
            for j in range(self.tagNum):
                dp[i][j] = -1
                # k为上一层的词性
                for k in range(self.tagNum):
                    #上一层该词性的得分*这一层对应转移矩阵的得分*这一层对应发射矩阵的得分
                    score = dp[i-1][k]*self.B[k][j]*self.C[j][words[i]]
                    #遍历完上一层后取得分最高的
                    if score > dp[i][j]:
                        dp[i][j] = score
                        pre[i][j] = k #存储上一层的选择

        result_id = [0] * length#存储最终结果的tagID
        result_id[length-1] = int(np.argmax(dp[length-1]))#返回最终得分最高的最后一个单词的词性

        # 向前遍历
        for i in range(length-2,-1,-1):#i为当前单词，从倒数第二个单词开始遍历
            result_id[i] = pre[i+1][result_id[i+1]]

        result = []
        for i in range(len(result_id)):#将tagID转化为tagName
            result.append(self.id2tag[result_id[i]])
        return result

@app.route('/', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':  # 判断是否是 POST 请求
        # 获取表单数据
        sentence = request.form.get('sentence')  # 传入表单对应输入字段的 name 值

        hmm = HMM()
        hmm.train('traindata.txt')
        tags = hmm.viterbi(sentence)
        if tags==False:
            print(0)
            return render_template('index.html',tags=[],words=[],length=0,flag=0)#有生词
        print(tags)
        return render_template('index.html',tags=tags,words=sentence.split(" "),length=len(tags),flag=1)#正常返回
    print(2)
    return render_template('index.html',tags=[],words=[],length=0,flag=2)





