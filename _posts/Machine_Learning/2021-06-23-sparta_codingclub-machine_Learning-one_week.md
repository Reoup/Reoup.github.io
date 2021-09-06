---
layout: post
title:  "스파르타 코딩클럽 머신러닝 1주차"
author: Reo
date: '2021-06-24'
category: Machine-Learning
excerpt: "스파르타 코딩클럽 머신러닝 1주차"
thumbnail: /assets/img/posts/Machine_Learning.png
tags: [스파르타 1주차, 머신러닝, 딥러닝, 지도 학습, 비지도 학습, 강화 학습, 선형 회귀, 경사 하강법, Learning rate, 손실 함수,  데이터셋 분할, 알고리즘, Colab, Colaboratory, Kaggle, 캐글]
permalink: /blog/machine_learning-one_week/
usemathjax: true
---
<h1>01. 1주차 배울 것</h1>

<h2> 머신러닝 </h2>

<h2>선형 회귀(linear regression)</h2>

<p>컴퓨터가 풀 수 있는 문제 중에 가장 간단한 것이 두 데이터 간의 직선 관계를 찾아 내서 x 값이 주어졌을 때 y값을 예측하는 것</p>

![screenshot](/assets/img/one/linear_regression.png)



<h1>02. 필수 계정 가입</h1>

<p>Gmail 가입하기 <a href="http://gmail.com"><span>(가입 링크)</span></a></p>

<p>Kaggle 가입하기 <a href="http://kaggle.com"><span>(가입 링크)</span></a></p>

<h1>03. 머신러닝이란?</h1>

<h2>알고리즘</h2>

<p>관련 분야에서 어떠한 문제를 해결하기 위해 정해진 일련의 절차나 방법을 공식화한 형태로 표현한 것, 계산을 실행하기 위한 단계적 절차</p>

<h2>딥러닝</h2>

<p>
    머신러닝이라는 포괄적인 범위 안에 딥러닝이 포함되어 있음<br>
    연구 초반에는 MLP(Multi-Layer Perceptron)이라고 불렸으나 현재는 딥러닝으로 불림
</p>



<h2>머신러닝의 회귀와 분류</h2>

<h3>문제를 풀 때, 해답을 내는 방법을 크게 회귀 또는 분류로 나눌 수 있음</h3>

<h4>회귀(Regression)</h4>

<p>모든 문제를 풀기 위해서는 입력값(Input)과 출력값(Output)을 정의해야 함<br>
    출력값이 연속적인 소수점으로 예측하게 하도록 푸는 방법을 회귀 방법이라고 함
</p>

<h4>분류(Classification)</h4>

<p>분류 문제는 회귀 문제보다 더 쉬움<br>
    해당 과목의 이수 여부를(Pass or Fail)로 0, 1이라는 이진 클래스로 나눌 수 있다. 0이면 미이수(Fail), 1이면 이수(Pass) 이런식으로. 이런 경우를 이진 분류(Binary classification)라고 부름
</p>

<p>만약에 과목의 성적(A, B, C, D, E)일 때, 클래스를 5개의 클래스로 나누면 이 방법을 다중 분류(Multi-class classification, Multi-label classification)라고 부른다.</p>

<h2>지도 학습/ 비지도 학습/ 강화 학습</h2>

<h4>머신러닝은 크게 3가지 분류 (지도 학습/ 비지도 학습/ 강화 학습) </h4>

![screenshot](/assets/img/one/Machine_Learning.png)

출처: https://towardsdatascience.com/the-future-with-reinforcement-learning-877a17187d54

<ul>
	<li>Supervised Learning</li>
    <ul>
        <li>Classfication</li>
        <ul>
        	<li>Identify Fraud Detection</li>
            <li>Image Classfication</li>
            <li>Customer Retention</li>
        </ul>
        <li>Regression</li>
        <ul>
            <li>Advertising Popularity Prediction</li>
            <li>Weather Forecasting</li>
            <li>Market Forecasting</li>
            <li>Estimating life expectancy</li>
            <li>Popilation Growth Prediction</li>
        </ul>
    </ul>
    <li>Unsupervised Learning</li>
    <ul>
        <li>Clustering</li>
        <ul>
        	<li>Recommender Systems</li>
            <li>Targetted Marketing</li>
            <li>Customer Segmentation</li>
        </ul>
        <li>Dimensionlity Reduction</li>
        <ul>
            <li>Big data Visualistation</li>
            <li>Meaningful Compression</li>
            <li>Structure Discovery</li>
            <li>Feature Elicitation</li>
        </ul>
    </ul>
    <li>Reinforcement Learning</li>
    <ul>
        <li>Real-time decisions</li>
        <li>Robot Navigation</li>
        <li>Learning Tasks</li>
        <li>Skill Acquisition</li>
        <li>Game AI</li>
    </ul>
</ul>



<h3>지도 학습(Supervised Learning): 정답을 알려주면서 학습시키는 방법</h3>

![screenshot](/assets/img/one/Supervised_Learning.png)

<p>출처: https://www.researchgate.net/figure/Supervised-learning-and-unsupervised-learning-Supervised-learning-uses-annotation_fig1_329533120</p>

<p>지도 학습은 기계에게 입력값과 출력값을 전부 보여주면서 학습/ 대신 정답(출력값)이 없으면 이 방법으로 학습시킬 수 없음<br>
    입력값에 정답을 하나씩 입력해주는 작업을 하게 되는 경우가 있는데 이 과정을 <strike>노가다</strike> 라벨링(Labeling, 레이블링) 또는 어노테이션(Annotation)이라고 함
</p>

<h3>비지도 학습(Unsupervised Learning): 정답을 알려주지 않고 군집화(Clustering)하는 방법</h3>

![screenshot](/assets/img/one/Unsupervised_Learning.png)

<p>비지도 학습은 그룹핑 알고리즘의 성격을 띄우고 있음<br>
    비지도 학습은 라벨(Label 또는 Class)이 없는 데이터를 가지고 문제를 풀어야 할 때 큰 힘을 발휘함
</p>
<p>비지도 학습의 종류 (참고)</p>

![screenshot](/assets/img/one/Types_of_ Unsupervised_Learning.png)

<p style="line-height:3px;">군집(Clustering)<p>
<ul>
    <li>K-평균(K-Means)</li>
    <li>계측 군집 분석(HCA, Hierachical Cluster Analysis)</li>
    <li>기댓값 최대화(Expectation Maximization)</li>
</ul>

<p style="line-height:3px;">시각화(Visualization)와 차원 축소(Dimensionality Reduction)<p>
<ul>
    <li>주성분 분석(PCA, Principal Component Analysis)</li>
    <li>커널 PCA(Kernel PCA)</li>
    <li>지역적 선형 임베딩(LLE, Locally-Linear Embedding)</li>
    <li>t-SNE(t-distrubuted Stochastic Neighbor Embedding)</li>
</ul>

<p style="line-height:3px;">연관 규칙 학습(Association Rule Learning)<p>
<ul>
    <li>어프라이어리(Apriori)</li>
    <li>이클렛(Eclat)</li>
</ul>
<h3>강화 학습: 주어진 데이터 없이 실행과 오류를 반복하면서 학습하는 방법(알파고를 탄생시킨 머신러닝 방법)</h3>

<p>분류할 수 있는 데이터가 존재하지 않거나, 데이터가 있어도 정답이 따로 정해져 있지 않고, 자신이 한 행동에 대해 보상을 받으며 학습하는 것</p>

<p style="line-height:3px;">강화학습의 개념</p>
<ul>
	<li>에이전트(Agent)</li>
    <li>환경(Environment)</li>
    <li>상태(State)</li>
    <li>행동(Action)</li>
    <li>보상(Reward)</li>
</ul>

<p>강화 학습은 이전부터 존재했지만 좋은 결과를 내지 못했음. But 딥러닝의 등장으로 학습에 신경망을 적용하면서 바둑, 자율주행차와 같은 복잡한 문제에 적용할 수 있게 되었음</p>

<참고>

<a href="https://goo.gl/jrKrvf">딥러닝과 강화 학습으로 나보다 잘하는 쿠키런 AI 구현하기</a>

<h1>04. 선형 회귀</h1>
<h3>선형 회귀와 가설, 손실 함수 Hypothesis & Cost function (Loss function)</h3>

<img src="/assets/img/one/linear_regression.png" title="직선 그래프" alt="오류뜨지마"/>

<p>선형 모델은 수식으로 아래와 같이 표현할 수 있음(직선 = 1차 함수)</p>

$$
\begin{align*}
H(x,y)= Wx + b
\end{align*}
$$

<p> 정확한 시험 점수를 예측하기 위해 우리가 만든 임의의 직선(가설)과 점(정답)의 거리가 가까워지도록 해야 함(=mean squared error)</p>

$$
\begin{align*}
Cost = \frac{1}{N}\sum_{i=1}^N(H(x_i)-y_i)^2
\end{align*}
$$

<p> H(x)는 우리가 가정한 직선이고, y는 정답 포인트라고 했을 때 H(x)와 y의 거리(또는 차의 절대값)가 최소가 되어야 이 모델이 잘 학습되었다고 말할 수 있음<br>
    우리가 임의로 만든 직선 H(x)를 가설(Hypothesis)라고 하고, Cost를 손실 함수(Cost or Loss function)라고 한다.</p>

<dl>
    <dt><b>머신러닝 더 알아보기</b></dt>
    <dd>실무에서 사용하는 머신러닝 모델은 1차 함수보다 더 높은 고차원 함수이지만 원리는 똑같습니다.<br>
    우리는 데이터를 보고 "어떤 함수에 비슷한 모양일 것이다"라고 가설을 세우고 그에 맞는 손실 함수를 정의합니다. 여기에서 우리가 할 일은 끝이고, 이제는 우리가 정의한 손실 함수를 기계가 보고 우리의 가설에 맞출 수 있도록 열심히 계산하는 일을 하게 됩니다. 그래서 기계 학습의 이름을 "머신러닝"이라고 불리게 된 것입니다.</dd>
</dl>

<h3>다중 선형 회귀</h3>

<p>선형 회귀와 똑같지만 입력 변수가 여러개라고 생각하면 됨</p>

<p>만약 입력값이 2개 이상이 되는 문제를 선형 회귀로 풀고 싶을 때 다중 선형 회귀 방법을 사용합니다.</p>

<span>가설</span>
$$
\begin{align*}
H(x_1,x_2,...,x_n)= w_1x_1 + w_2x_2 + ... + w_n+x_n + b &
\end{align*}
$$

$$
\begin{align*}
Cost = \frac{1}{N}\sum_{i=1}^N(H(x_1,x_2,...,x_n)-y)^2
\end{align*}
$$

<h1>05. 경사 하강법</h1>

<h3> 경사 하강법이란?</h3>
<p>손실 함수를 최소화 하는 것이 목표임<br>
손실 함수를 대충 상상하고, 아래와 같은 모양을 가지고 있다고 가정</p>

<img src="/assets/img/one/Gradient_descent_method.png" title="경사 하강법 그래프" alt="오류뜨지마"/>

<p>Wx + b  W의 값과 b의 값을 바꿔가면서 Cost 값이 내려가는지 올라가는지 확인</p>

<img src="/assets/img/one/Gradient_descent_method.gif">

<p>점진적으로 문제를 풀어가며, 처음에 랜덤으로 한 점으로부터 시작합니다. 좌우로 조금씩 그리고 한번씩 움직이면서 Cost 값이 이전 값보다 작아지는지를 관찰합니다. 한칸씩 전진하는 단위를 Learning rate라고 부릅니다. 그리고 그래프의 최소점에 도달하게 되면 학습을 종료합니다.</p>

<p>출처: https://medium.com/hackernoon/life-is-gradient-descent-880c60ac1be8</p>

<h3> Learning rate</h3>

<p>우리가 만든 머신러닝 모델이 학습을 잘하기 위해서 적당한 Learning rate 값을 찾는 노가다가 필수적</p>

<p>만약 Learning rate값이 작으면 초기 위치로부터 최소점을 찾는데까지 많은 시간이 걸리고, 이것을 학습하는데 최소값에 수렴하기까지 많은 시간이 걸린다.</p>

<p>반대로 Learning rate가 지나치게 크면 우리가 찾으려는 최소값을 지나치고 계속 진동하다가 최악의 경우에는 ∞으로 발산하게 된다. 이런 상황을 Overshooting이라고 부릅니다.

<h3>실제로 손실 함수를 그릴 수 있을까?</h3>

<p>간단한 선형 회귀 문제의 경우에는 그래프를 그릴 수 있지만 복잡한 가설을 세울 경우에는 사람이 그릴 수도 없으며 상상도 할 수 없는 형태가 됩니다.

<img src="/assets/img/one/complication_graph.png" title="복잡한 그래프" alt="오류뜨지마"/>

<p>위의 사진처럼 그나마 복잡하게 그린 그래프지만 실제로는 저렇게 생기지도 않았으며, 2, 3차원도 아닌 몇십, 몇백차원으로 그릴 수도 없으며, 상상도 할 수 없음</p>

<p>그래서 우리의 목표는 이 손실 함수의 최소점인 Global cost minimum을 찾는 것입니다. 하지만 한 칸씩 움직이는 스탭(Learnig rate)를 잘못 설정할 경우 Local cost minimum에 빠질 가능성이 높다. Cost가 높다는 얘기는 우리가 만든 모델의 정확도가 낮다는 말과 같기 때문에 최대한 Global cost minimum을 찾기 위해 좋은 가설과 좋은 손실 함수를 만들어 기계가 잘 학습할 수 있도록 만들어야 하고, 그것이 바로 머신러닝 엔지니어의 핵심 역할이다.</p>

<h1>06. 데이터셋 분할</h1>

<h3>학습/검증/테스트 데이터</h3>

<h5>각각의 데이터가 어떤 용도로 사용되는지 확인해보자</h5>

<img src="/assets/img/one/Types_of_Data.png" title="데이터 종류들" alt="오류뜨지마"/>

<ol>
    <li>Traning set(학습 데이터셋, 트레이닝셋) = 교과서</li>
    머신러닝 모델을 학습시키는 용도로 사용합니다. 전체 데이터셋의 약 80% 정도를 차지함
    <li>Validation set(검증 데이터셋, 밸리데이션셋)= 모의고사</li>
    머신러닝 모델의 성능을 검증하고 튜닝하는 지표의 용도로 사용합니다. 이 데이터는 정답 라벨이 있고, 학습 단계에서 사용하기는 하지만, 모델에게 데이터를 직접 보여주지는 않으므로 모델의 성능에 영향을 미치진 않음<br>
    손실 함수, Optimizer 등을 바꾸면서 모델을 검증하는 용도로 사용 // 전체 데이터셋의 약 20% 정도 차지함
    <li>Test set(평가 데이터셋, 테스트셋) = 수능</li>
    정답 라벨이 없는 실제 환경에서의 평가 데이터셋입니다. 검증 데이터셋으로 평가된 모델이 아무리 정확도가 높더라도 사용자가 사용하는 제품에서 제대로 동작하지 않으면 쓸모가 없겠죠?
</ol>

<h1>07. 실습 환경 소개 (Colab)</h1>

<a href="https://colab.research.google.com/drive/1qfFR5qiAjaqpDuERWeNdlKJWpo0vY7ej?usp=sharing">Colab 소개</a>

<p>Gmail 계정과 Kaggle 계정이 있어야 합니당~</p>

드라이브를 복사한 후 진행

<h3>Colaboratory란?</h3>

<P>줄여서 Colab(콜랩)이라고도 하는 Colaboratory를 사용하면 브라우저에서 Python을 작성하고 실행할 수 있음.</P>
<ul>
    <li>개발 환경 구성이 필요하지 않음</li>
    <li>GPU 무료 액세스</li>
    <li>간편한 공유</li>
</ul>

<p><b>학생</b>이든,<b>데이터 과학자</b>든,<b>AI 연구원</b>이든 Colab으로 업무를 더욱 간편하게 처리할 수 있음.<br> <a href="https://www.youtube.com/watch?v=inN8seMm7UI">Colab 소개 영상</a>에서 자세한 내용 확인하거나 아래에서 시작

<p>정적 웹페이지가 아닌 코드를 작성하고 실행할 수 있는 대화형 환경인 <b>Colab 메모장</b><br>
    변수를 저장하고 결과를 출력하는 간단한 Python 스크립트가 포함된 <b>코드 셀</b><br>
    셀 실행은 셀을 클릭한 후 왼쪽 실행 버튼을 누르거나 Command/Ctrl+Enter 혹은 Command/Shift+Enter를 누르면 실행할 수 있다.
</p>

<h4>패키지 불러오기</h4>

<p>Colab을 통해 인기 있는 Python 라이브러리를 최대한 활용하여 데이터를 분석하고 시각화할 수 있음</p>

{% highlight python %}
import numpy as np
from matplotlib import pyplot as plt
{% endhighlight %}

<h4>그래프 그리기</h4>

<p>아래 코드셀에서는 <b>Numpy</b>를 사용하여 임의의 데이터를 생성하고 <b>매트플롯립</b>으로 이를 시각화한다.

{% highlight python %}
data = np.random.randint(-100, 100, 50)
print(data)
{% endhighlight %}

<p>랜덤으로 -100~99까지 50개를 출력<br>
    [-65   5 -78 -59  59  76  61 -29 -89  88 -78  46  64  69 -56  40 -64 -98
  -3 -85 -42  49 -48 -69  45  77 -65  61  -4 -59  76  -2  -4 -53  84  -7
    -68  -6  42  39 -36  84 -24  31  41  63  -8  15  63  45]</p>

{% highlight python %}
plt.plot(data.cumsum())
plt.show()
{% endhighlight %}

<img src="/assets/img/one/random_graph.png" title="랜덤 그래프" alt="오류뜨지마"/>

<h4>초간단 Linear Regression 실습(TensorFlow)</h4>

<p>tensorflow: 기본 머신러닝 프레임워크 (똑같은 코딩이 많아짐)<br>
    keras: tensorflow의 상위 API<br>
    현재 TensorFlow에서는 keras를 사용하도록 권장하고 있고 최신 트렌드에 맞게 keras를 사용할 예정
</p>

<h3>Kaggle 캐글</h3>

<a href="https://www.kaggle.com">https://www.kaggle.com</a>

<h4>Colab에서 Kaggle 데이터셋 다운로드 방법</h4>

<ol>
    <li>Kaggle 회원가입</li>
    <li>Account(계정)페이지 진입 (https://www.kaggle.com/[사옹자이름]/account)</li>
    <li>API - Create New API Token 클릭하고 kaggle.json 다운로드</li>
    <li>브라우저에서 json 파일을 열어 username 및 key 복사</li>
    <li>아래 코드에 자신의 username 및 key를 붙여넣어 환경변수 설정 실행</li>
</ol>

{% highlight python %}
import os
os.environ['KAGGLE_USERNAME'] = 'username'
os.environ['KAGGLE_KEY'] = 'key'
{% endhighlight %}

<h4>광고 데이터셋 다운로드</h4>

<ol>
    <li>Kaggle에서 원하는 데이터셋을 검색 (예: <a href="https://www.kaggle.com/ashydv/advertising-dataset">https://www.kaggle.com/ashydv/advertising-dataset</a>) </li>
    <li>[Copy API command] 버튼 클릭 (New Notebook 옆에 ... 버튼 클릭)</li>
    <li>코드 셀에 붙여넣고 실행! (! 필수)</li>
</ol>

{% highlight python %}
from google.colab import drive
drive.mount('/content/drive')
{% endhighlight %}

<p>!kaggle datasets download -d ashydv/advertising-dataset #다운로드</p>
<p>!unzip /content/advertising-dataset.zip #데이터셋 압축 해제</p>

<h4>광고 데이터 예측하기 (Multi-variable linear regression) #다중선형회귀

<p>TV, Newspaper, Radio 광고 금액으로 Sales 예측</p>

{% highlight python %}
from tensorflow.keras.models import Sequential #모델을 정의할 때 사용
from tensorflow.keras.layers import Dense #가설을 구현할 때 사용
from tensorflow.keras.optimizers import Adam, SGD #optimizers의 Adam과 SGD 사용
import numpy as np
import pandas as pd #csv 파일을 읽을 때 사용
import matplotlib.pyplot as plt #그래프를 그릴 때 사용
import seaborn as sns #그래프를 그릴 때 사용
from sklearn.model_selection import train_test_split #머신러닝을 도와주는 패키지 // training set 과 test set을 분리해주는 기능

df = pd.read_csv('advertising.csv') # advertising.csv 파일 읽기

#데이터셋 가공
x_data = np.array(df[['TV', 'Newspaper', 'Radio']], dtype=np.float32) 
y_data = np.array(df['Sales'], dtype=np.float32)

x_data = x_data.reshape((-1, 3)) # TV, Newspaper, Radio가 3개이므로 뒤에는 3으로 적어야 함 만약 5개면 5개겠죠? ㅎㅎ
y_data = y_data.reshape((-1, 1))

print(x_data.shape) #(200, 3)
print(y_data.shape) #(200, 1)

#데이터셋 분할
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=10) #train_test_split을 이용해서 training set 80퍼 validation set 20퍼

print(x_train.shape, x_val.shape) #(160, 1) (40, 1)
print(y_train.shape, y_val.shape) #(160, 1) (40, 1)

#학습
model = Sequential([
 Dense(1) #출력 1개
])

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.1))

model.fit(
  x_train,
  y_train,
  validation_data=(x_val, y_val), # 검증 데이터를 넣어주면 한 epoch이 끝날때마다 자동으로 검증
  epochs=100 # epochs 복수형으로 쓰기!
)
{% endhighlight %}
<img src="/assets/img/one/last_result.png" title="마지막 결과" alt="오류뜨지마"/>
{% highlight python %}
y_pred = model.predict(x_val)
print(y_pred.shape) #검증 셋
{% endhighlight %}
<p>(40, 1)</p>

<h4>TV 데이터</h4>
{% highlight python %}
plt.scatter(x_val[:, 0], y_val)
plt.scatter(x_val[:, 0], y_pred, color='r')
plt.show()
{% endhighlight %}
<img src="/assets/img/one/TV_data.png" title="마지막 결과" alt="오류뜨지마"/>
<h4>Newspaper 데이터</h4>
{% highlight python %}
plt.scatter(x_val[:, 1], y_val)
plt.scatter(x_val[:, 1], y_pred, color='r')
plt.show()
{% endhighlight %}
<img src="/assets/img/one/Newspaper_data.png" title="마지막 결과" alt="오류뜨지마"/>
<h4>Radio 데이터</h4>
{% highlight python %}
plt.scatter(x_val[:, 2], y_val)
plt.scatter(x_val[:, 2], y_pred, color='r')
plt.show()
{% endhighlight %}
<img src="/assets/img/one/Radio_data.png" title="마지막 결과" alt="오류뜨지마"/>

<h3>1주차 숙제</h3>
<h4>스스로 선형회귀를 구현하기</h4>
<ul>
    <li>연차-연봉 데이터셋 살펴보기 <a href="https://www.kaggle.com/rsadiq/salary">https://www.kaggle.com/rsadiq/salary</a></li>
    <li>Learning rate(lr)를 바꾸면서 실험</li>
    <li>Optimizer를 바꾸면서 실험</li>
    <li>손실함수(loss)를 mean_absolute_error로 바꿔서 실험</li>
</ul>
<a href="https://colab.research.google.com/drive/1aSHh6uL8FUyHVOB26T3HgR3K2Nez2rpn?usp=sharing">숙제</a>