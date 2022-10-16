<<<<<<< HEAD:hmm_vterbi.py
=======

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="https://pyscript.net/alpha/pyscript.css" />
    <script defer src="https://pyscript.net/alpha/pyscript.js"></script>
</head>
<body>
    <py-config type="toml">
        packages = ["numpy"]
        paths = ["./traindata.txt"]
    </py-config>
    <py-script>
>>>>>>> c4dc63751573b3509d6c13f16d78e0e43c500ee2:HMM/hmm.html
import numpy as np

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

            #合并相似功能的符号
            if tag in [",", ".", ":", "``", "''"]:
                tag="END"#特殊的符号，视作一句话的终止，tag标为空。
            elif tag in ["$","#","SYM","(", ")"]:
                tag="W"#句中符号，当做普通标点。

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

            #合并相似功能符号
            if tag in [",", ".", ":", "``", "''"]:
                tag="END"#特殊的符号，视作一句话的终止，tag标为空。
            elif tag in ["$","#","SYM","(", ")"]:
                tag="W"#句中符号，当做普通标点。

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
        words = [self.word2id[word.lower()] for word in sentence.split(" ")]#将单词转换为ID
        length = len(words)

        dp = np.zeros((length, self.tagNum))#存储分数，行数=句子长度，列数=词性
        pre = np.zeros((length, self.tagNum), dtype=int)#存储上一层的选择

        #第一个单词
        for j in range(self.tagNum):#遍历词性
            dp[0][j] = self.A[j]*self.C[j][words[0]]#发射矩阵x转移矩阵
        #第二个单词起
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

if __name__ == '__main__' :
    HMM = HMM()
    HMM.train('traindata.txt')
    sentence = "Today is a good day ;"
    print(sentence)
    tags = HMM.viterbi(sentence)
    print(tags)
<<<<<<< HEAD:hmm_vterbi.py
=======

    </py-script>
    <div>
        <input id="new-task-content" class="py-input" type="text">
        <button id="new-task-btn" class="py-button" type="submit" py-click="add_task()">
          RUN
        </button>
    </div>
</body>
</html>
>>>>>>> c4dc63751573b3509d6c13f16d78e0e43c500ee2:HMM/hmm.html
