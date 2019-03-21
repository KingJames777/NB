from numpy import *
from os import listdir
from sklearn.metrics import accuracy_score as acc

def string2words(string):
      import re
      words=re.split(r'\W+',string)
      return [word.lower() for word in words if len(word)>2]

def train_test_split():
      spam_test=random.choice(range(1,25),size=5,replace=False)  ##抽5个作为测试
      ham_test=random.choice(range(1,25),size=5,replace=False)
      spam_train=set(range(1,25))-set(spam_test)
      ham_train=set(range(1,25))-set(ham_test)
      return spam_test,spam_train,ham_test,ham_train

##words_list 和y的前19是spam训练的单词列表和标签，20-38是ham训练的.
##full_text是训练集的整个词汇表，测试集中未出现的不管

def split_words(spam_train,ham_train):  ##参数为各自索引
      words_list,full_text,y=[],[],[]
      for i in spam_train:
            words=string2words(open('spam/%d.txt'%i).read())  ## 单词列表
            words_list.append(words)
            full_text.extend(words)
            y.append(1)
      for i in ham_train:
            words=string2words(open('ham/%d.txt'%i).read())
            words_list.append(words)
            full_text.extend(words)
            y.append(0)
      return words_list,set(full_text),y

def vectorization(words_list,full_text):
      full_text=list(full_text)
      n=len(full_text) ##属性数
      m=len(words_list)  ##样本数
      X=zeros((m,n),dtype='int8')
            
      for i in range(m):
            stats=unique(words_list[i])
            for j in range(len(stats)):
                  try:
                        index=full_text.index(stats[j])  ##找到单词所在索引
                  except(ValueError):
                        continue
                  X[i][index]=1  ##对应位置一
      return X

def freq(num,X):
      count=0
      for x in X:
            if x==num:
                  count+=1
      return count
      
def nb(X_train,X):  ##  X是测试数据
      prob=0
      n=len(X_train)
      for i in range(len(X)):  ##每个属性
            X_temp=X_train[:,i]
            prob+=log((1+freq(X[i],X_temp))/(n+2))  ##这是个负值！
      return prob  

def predict(X_spam,X_ham,X_test,y_test):
      pred=[]
      for X in X_test:
            spam=nb(X_spam,X)
            ham=nb(X_ham,X)
            pred.append(1) if spam>ham else pred.append(0)
      print(pred,y_test)
      print('Accuracy:',acc(pred,y_test))

spam_test,spam_train,ham_test,ham_train=train_test_split()  ##索引
train_words_list,train_full_text,y_train=split_words(spam_train,ham_train)
test_words_list,test_full_text,y_test=split_words(spam_test,ham_test)

X_train=vectorization(train_words_list,train_full_text)
X_test=vectorization(test_words_list,train_full_text)

X_spam=X_train[:19]
X_ham=X_train[19:]

predict(X_spam,X_ham,X_test,y_test)

















