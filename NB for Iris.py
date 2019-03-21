from numpy import *
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

def Gaussian(x,ave,var):  ##对数化的高斯函数值
      return -(x-ave)*(x-ave)/2/var-0.5*log(var)

class NB:
      def __init__(self,X_train,y_train):
            self.X,self.y=X_train,y_train
            self.classes=len(unique(self.y))  ##类别数
            self.m,self.n=shape(self.X)
            self.prior, self.var, self.ave=self.var_ave()
            
      ##求出各类各属性的均值及方差
      def var_ave(self):
            var,ave=zeros((self.classes,self.n)),zeros((self.classes,self.n))
            class_labels, counts=unique(self.y,return_counts=True)  ##unique会自动排序
            prior=counts/sum(counts)
            for i in range(len(class_labels)):  ##注意i对应第i种类别
                  label=class_labels[i]  ##label=0,1,2.....
                  X_label=self.X[self.y==label]  ##取出各类对应的样本
                  
                  var[i],ave[i]=X_label.var(axis=0,ddof=True),X_label.mean(axis=0)
            return prior,var,ave

      def predict(self,X_test):
            pred=[]
            for X in X_test:
                  post=[]
                  for i in range(self.classes):  ##逐类计算后验概率
                        prob=log(self.prior[i])
                        for j in range(self.n):  ##各个属性
                              prob+=Gaussian(X[j],self.ave[i][j],self.var[i][j])
                        post.append(prob)
                  pred.append(argmax(post))
            return pred

                  
iris=datasets.load_wine()
X,y = iris.data,iris.target
X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901120, stratify=y)

nb=NB(X_train,y_train)
pred=nb.predict(X_test)

print('acc:',accuracy_score(y_test,pred))


##不宜用于手写数字识别，会遇到0方差。
digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901120, stratify=y)

nb=NB(X_train,y_train)
pred=nb.predict(X_test)

##print('acc:',accuracy_score(y_test,pred))


















