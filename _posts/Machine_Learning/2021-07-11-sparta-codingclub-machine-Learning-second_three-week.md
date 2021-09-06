---
layout: post
title: "스파르타 코딩클럽 머신러닝 3주차"
author: Reo
date: '2021-07-11'
category: Machine-Learning
excerpt: "스파르타 코딩클럽 머신러닝 3주차"
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 3주차, 딥러닝, Deep learning, 선형 회귀, 비선형,Deep neural networks, Multilayer Perceptron, MLP, 역전파, Backpropagation, batch, iteration, epoch, 활성화 함수, Activation function, 과적합, 과소적합, Overfitting, Underfitting, 데이터 증강기법, Data augmentation, 드랍아웃, Dropout, 앙상블, Ensemble, Learning rate decay, Learning rate schedules]
permalink: /blog/machine_learning-three_week/
usemathjax: true
---

<h2> 3주차 오늘 배울 것</h2>
<h4>딥러닝이란?</h4>
<p>딥러닝은 머신 러닝 한 분야임.</p>
<img src="/assets/img/three/3_1.png" title="딥러닝의 필요성"/>
<p>출처: https://neocarus.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9D%98-%ED%95%84%EC%9A%94%EC%84%B1</p>

<p>복잡한 문제들을 풀기 위해 선형회귀를 여러번 반복</p>
<img src="/assets/img/three/3_2.png" title="선형 회귀 반복"/>
<p>선형회귀를 여러번 반복한다 해서 비선형(선형이 아닌 것)이 되는 것은 아니였음.</p>
$$
\begin{align*}
y= W_2(W_1x + b_1) + b_2 = (W_1W_2)x + (W_1b_1 + b_2)
\end{align*}
$$
<p>그래서 선형 회귀 사이에 비선형의 무언가를 넣어야 한다고 판단</p>
<img src="/assets/img/three/3_3.png" title="선형 회귀 +(비선형 추가) 반복"/>
<p>이렇게 층(layer)을 여러개 쌓기 시작했고 이 모델은 작동이 잘 되었음. 층을 깊게(Deep) 쌓는다고 해서 딥러닝이라고 부르게 된 것</p>
{% capture notice-2 %}  <!--notice-2 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
<h3>딥러닝의 다른 단어 표현</h3>
<ol>
  <li> 딥러닝(Deep learning)</li>
  <li> Deep neural networks</li>
  <li>Multilayer Perceptron(MLP)</li>
</ol>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-2 | markdownify }} <!--div 태그 사이에 notice-2 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>
<h4>딥러닝의 주요 개념과 기법</h4>
<p>신경망을 실제로 구성하는데 필요한 다양한 개념과 기법들에 대해 알아보기!</p>
<ul>
  <li>배치 사이즈와 에폭</li>
  <li>활성화 함수</li>
  <li>과적합과 과소적합</li>
  <li>데이터 증강</li>
  <li>드랍아웃</li>
  <li>앙상블</li>
  <li>학습률 조정</li>
</ul>

<h2>딥러닝의 역사</h2>
<h4>XOR 문제</h4>
<p>기존의 머신러닝은 AND, OR 문제로부터 시작</p>
<img src="/assets/img/three/3_4.png" title="OR 그래프와 AND 그래프"/>
<p>출처: https://donghwa96.tistory.com/11</p>
<ul>
  <li>위와 같은 문제를 풀기 위해서는 직선 하나만 있으면 되지만 그 직선은 논리회귀로 쉽게 만들 수 있었음</li>
  $$
\begin{align*}
y = w_0 + w_1x_1 + w_2x_2
\end{align*}
$$
  <li>이 수식을 아래와 같이 그림으로 나타낼 수 있었음. 이런 모양을 Perceptron(퍼셉트론)이라고 불렀음.</li>
<img src="/assets/img/three/3_5.png" title="퍼셉트론"/>
  <li>선형 회귀로는 AND, OR 문제를 잘 풀었지만 XOR 문제는 풀지 못했음</li>
<img src="/assets/img/three/3_6.png" title="XOR?"/>
  <li>Perceptron을 여러개 붙인 Multilayer Perceptrons (MLP)라는 개념을 도입해서 문제 시도</li>
<img src="/assets/img/three/3_7.png" title="MLP"/>
<h4>Backpropagation (역전파)</h4>
  <li>1974년에 발표된 Paul Werbos(폴)이라는  사람의 박사 논문의 주장</li>
<img src="/assets/img/three/3_8.png" title="폴의 주장"/>
  <ol>
    <li>우리는 W(weight)와 b(bias)를 이용해서 주어진 입력을 가지고 출력을 만들어 낼 수 있다.</li>
    <li>MLP가 만들어낸 출력이 정답값과 다를 경우 W와 b를 조절해야 한다.</li>
    <li>조절하는 가장 좋은 방법은 <b>출력에서 Error(오차)를 발견하여 뒤에서 앞으로 점차 조절하는 방법</b>이 필요하다.</li>
  </ol>
  <li>이 알고리즘은 관심받지 못하다가 1986년 Hinton 교수가 똑같은 방법을 독자적으로 발표하면서 알려짐</li>
  <li>XOR 문제는 MLP를 풀 수 있게 되어 해결될 수 있었고 그 핵심 방법은 바로 역전파 알고리즘의 발견이였다.</li>
<img src="/assets/img/three/3_9.png" title="역전파 알고리즘"/>
</ul>

<h2>Deep Neural Networks 구성 방법</h2>
<h4>Layer(층) 쌓기</h4>
<ul>
  <li>딥러닝에서 네트워크의 구조는 크게 3가지로 나누어짐</li>
<img src="/assets/img/three/3_10.png" title="네트워크 구조"/>
  <ul>
    <li>Input layer(입력층): 네트워크의 입력 부분. 우리가 학습시키고 싶은 x 값</li>
    <li>Output layer(출력층): 네트워크의 출력 부분. 우리가 예측한 값, 즉 y 값</li>
    <li>Hidden layers(은닉층): 입력층과 출력층을 제외한 중간층</li>
  </ul>
  <li>풀어야 하는 문제에 따라 입력층과 출력층의 모양은 정해져 있고, 우리가 신경써야할 부분은 은닉층임. 은닉층은 완전연결 계층(Fully connected layer = Dense layer)로 이루어짐.</li>
  <li>기본적인 뉴럴 네트워크(Deep neural networks)에서는 보통 은닉층에 중간 부분을 넓게 만드는 경우가 많음. 예를 들면 보편적으로 아래와 같이 노드의 개수가 점점 늘어나다가 줄어드는 방식으로 구성함.</li>
  <ul>
    <li>입력층의 노드 개수 4개</li>
    <li>첫 번째 은닉층 노드 개수 8개</li>
    <li>두 번쨰 은닉층 노드 개수 16개</li>
    <li>세 번쨰 은닉층 노드 개수 8개</li>
    <li>출력층 노드개수 3개</li>
  </ul>
  <li>활성화 함수를 어디다가 넣어야하는지도 중요함. 보편적인 경우 모든 은닉층 바로 뒤에 위치함</li>
<img src="/assets/img/three/3_11.png" title="은닉층과 활성화 함수"/>
</ul>
<h4>네트워크의 Width(너비)와 Depth(깊이)</h4>
<ul>
  <li>우리의 수많은 시간을 투자하여 완성한 적당한 연산량을 가진, 적당한 정확도의 딥러닝 모델이 있다고 가정. 그 모델은 Baseline model(베이스라인 모델)이라고 보편적으로 지칭함.
  예를 들어 우리가 만든 베이스라인 모델의 크기가 다음과 같음</li>
  <ul>
    <li>입력층: 4</li>
    <li>첫 번째 은닉층: 8</li>
    <li>두 번쨰 은닉층: 4</li>
    <li>출력층: 1</li>
  </ul>
  <li>이 베이스라인 모델을 가지고 여러가지 실험(튜닝)을 할 수 있는데, 간단하게 성능을 테스트할 수 있는 방법이 바로 모델의 너비와 깊이를 가지고 테스트 하는 것.</li>
  <ol>
    <li>네트워크의 너비를 늘리는 방법</li>
    <p>네트워크의 은닉층의 개수를 그대로 두고 은닉층의 노드 개수를 늘리는 방법.</p>
    <ul>
      <li>입력층: 4</li>
      <li>첫 번째 은닉층: 8*2 = 16</li>
      <li>두 번쨰 은닉층: 4*2 = 8</li>
      <li>출력층: 1</li>
    </ul>
    <li>네트워크의 깊이를 늘리는 방법</li>
    <p>은닉층의 개수를 늘리는 방법.</p>
    <ul>
      <li>입력층: 4</li>
      <li>첫 번째 은닉층: 4</li>
      <li>두 번쨰 은닉층: 8</li>
      <li>세 번쨰 은닉층: 8</li>
      <li>네 번쨰 은닉층: 4</li>
      <li>출력층: 1</li>
    </ul>
    <li>너비와 깊이를 전부 늘리는 방법</li>
    <p>위에서 했던 두 가지 방법 모두 사용함.</p>
    <ul>
      <li>입력층: 4</li>
      <li>첫 번째 은닉층: 8</li>
      <li>두 번쨰 은닉층: 16</li>
      <li>세 번쨰 은닉층: 16</li>
      <li>네 번쨰 은닉층: 8</li>
      <li>출력층: 1</li>
    </ul>
  </ol>
</ul>
{% capture notice-3 %}  <!--notice-3 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
<p>실무에서는 네트워크의 너비와 깊이를 바꾸면서 실험을 많이 한다. 그만큼 시간도 많이 들고 지루한 작업이다. 다음 파트에서 배울 과적합과 과소적합을 피하기 위해서는 꼭 필요한 노가다이다.</p>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-3 | markdownify }} <!--div 태그 사이에 notice-3 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>

<h2>딥러닝의 주요 개념</h2>
<h4>Batch size, Epoch(배치 사이즈, 에폭)</h4>
<p><b>batch와 iteration</b></p>
<ul>
  <li>만약 우리가 10,000,000개의 데이터셋을 가지고 있다고 가정. 10,000,000개의 데이터셋을 한꺼번에 메모리에 올리고 학습시키려면 엄청난 용량을 가진 메모리가 필요하고 그 메모리를 사는데 (메모리가 없다면 개발하는데) 천문학적인 비용이 들 것임.</li>
  <li>따라서 우리는 이 데이터셋을 작은 단위로 쪼개서 학습 시키는데 쪼개는 단위를 배치(Batch)라고 부름. 예를 들어 1,000만개의 데이터셋을 1,000개 씩 쪼개어 10,000번을 반복하는 것 이 과정을 iteration(이터레이션)이라고 부름.</li>
</ul>
<p><b>epoch</b></p>
<ul>
  <li>보통 머신러닝에서는 똑같은 데이터셋을 가지고 반복 학습을 함. 만약 100번 반복 학습을 한다면 100 epochs(에폭)을 반복한다고 말함.</li>
  <li>batch를 몇 개로 나눠놓았냐에 상관 없이 전체 데이터셋을 한 번 돌 때 한 epoch이 끝남.</li>
</ul>
<p>따라서 1천만개의 데이터셋을 1천개 단위의 배치로 쪼개면, 1만개의 배치가 되고, 이 1만개의 배치를 100 에폭을 돈다고 하면 1만 * 100 = 100만번의 이터레이션을 도는 것이 됨.</p>
<img src="/assets/img/three/3_12.png" title="glossary"/>
<p>출처: https://www.mauriciopoppe.com/notes/computer-science/artificial-intelligence/machine-learning/glossary/</p>
<h4>Activation functions(활성화 함수)</h4>
<ul>
  <li>우리가 앞서 배운 MLP의 연결 구조를 여러개의 뉴런이 연결된 모습과 비슷하다고 가정하고 생각!</li>
<img src="/assets/img/three/3_13.png" title="MLP"/>
  <li>수많은 뉴런들은 서로 서로 빠짐없이 연결되어 있음. 그런데 뉴런들은 전기 신호의 크기가 특정 임계치(Threshold)를 넘어야만 다음 뉴런으로 신호를 전달하도록 설계되어 있음. 연구자들은 뉴런의 신호전달 체계를 흉내내는 함수를 수학적으로 만들었고, 전기 신호의 임계치를 넘어야 다음 뉴런이 활성화 한다고 해서 활성화 함수라고 부르게 됨.</li>
  <li>활성화 함수는 비선형 함수여야 하고, 대표적인 예가 시그모이드 함수임. 따라서 비선형 함수 자리에 시그모이드를 넣으면 밑에 처럼 됨.</li>
<img src="/assets/img/three/3_14.png" title="선형회귀와 시그모이드"/>
  <li>이런 비선형의 활성화 함수를 사용하여 다음 뉴런을 활성화 시킬지를 결정할 수 있음.</li>
<img src="/assets/img/three/3_15.png" title="시그모이드 함수"/>
  <li>시그모이드 함수는 x가 -6보다 작을 때는 0에 가까운 값을 출력으로 내보내서 비활성 상태를 만듦. 반대로 x가 6보다 클 때는 1에 가까운 값을 출려으로 내보내서 활성화 상태로 만듦.</li>
  <li>이런 활성화 함수는 여러가지 종류가 있음 그래프로 보면 아래와 같음.</li>
<img src="/assets/img/three/3_16.png" title="활성화 함수 여러 그래프"/>
  <li>딥러닝에서 가장 많이 보편적으로 쓰이는 활성화 함수는 Relu(렐루)임. 이유는 다른 활성화 함수에 비해 학습이 빠르며, 연산 비용이 적고, 구현이 간단함.</li>
  {% capture notice-4 %}  <!--notice-4 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
  <p>대부분 딥러닝 모델을 설계할 때 ReLU를 기본적으로 많이 쓰고, 여러 활성화 함수를 교체하는 노가다를 거쳐 최종적으로 정확도를 높이는 작업을 동반한다. 이러한 노가다의 과정을 모델 튜닝이라고 부름!</p>
  {% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
  <div class="notice">
    {{ notice-4 | markdownify }} <!--div 태그 사이에 notice-4 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
  </div>
</ul>
<h4>Overfitting, Underfitting(과적합, 과소적합)</h4>
<ul>
  <li>딥러닝 모델을 설계/튜닝하고 학습시키다 보면 가끔씩 Training loss는 점점 낮아지는데 Validation loss가 높아지는 시점이 있다. 그래프로 표시하면 아래와 같은 현상임.</li>
<img src="/assets/img/three/3_17.png" title="과적합, 과소적합"/>
  <li>이런 현상을 과적합 현상이라고 한다. 우리가 풀어야하는 문제의 난이도에 비해 모델의 복잡도가 클 경우 가장 많이 발생하는 현상임.</li>
  <li>반대로 우리가 풀어야하는 문제의 난이도에 비해 모델의 복잡도가 낮을 경우 문제를 제대로 풀지 못하는 혁상을 과소적합이라고 함.</li>
  <li>따라서 우리는 적당한 복잡도를 가진 모델을 찾아야 하며, 수십번의 튜닝 과정을 거쳐 최적합(Best fit)의 모델을 찾아야 함!!</li>
  {% capture notice-5 %}  <!--notice-5 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
  <p>딥러닝 모델을 학습시키다보면 보통 과소적합보다는 과적합때문에 골치를 썩는 경우가 많음. 과적합을 해결하는 방법에는 여러가지 방법이 있지만 대표적인 방법으론 데이터를 더 모으기, Data augmenation, Dropout 등이 있음.</p>
  {% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
  <div class="notice">
    {{ notice-5 | markdownify }} <!--div 태그 사이에 notice-5 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
  </div>
</ul>

<h2>딥러닝의 주요 스킬</h2>
<h4>Data augmentation (데이터 증강기법)</h4>
<p>과적합을 해결할 가장 좋은 방법은 데이터의 개수를 늘리는 방법. 하지만 실무에서는 데이터가 부족한 경우가 엄청 많음. 부족한 데이터를 보충하기 위해 데이터 증강기법을 사용함. 데이터 증강기법은 이미지 처리 분야의 딥러닝에서 주로 사용하는 기법이다.</p>
<img src="/assets/img/three/3_18.png" title="Data augmentation (데이터 증강기법)"/>
<p>출처: https://www.mygreatlearning.com/blog/understanding-data-augmentation/</p>
<p>원본 이미지 한장을 여러가지 방법으로 복사를 함. 사람의 눈으로 보았을 때 위의 어떤 사진을 보아도 사자인 것처럼 딥러닝 모델도 똑같이 보도록 학습을 시킴. 이 방법을 통해 더욱 강건한 딥러닝 모델을 만들게 됨.</p>
{% capture notice-6 %}  <!--notice-6 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
<p>데이터 증강기법은 반드시 정해진 방법들을 사용해야 하는 것은 아님. 데이터가 부족할 때 이미 있는 데이터를 사용하여 증강시키는 개념으로 여러가지의 방법을 사용해서 증강 방법을 새로 만들어 낼 수 있음.</p>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-6 | markdownify }} <!--div 태그 사이에 notice-6 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>
<h4>Dropout(드랍아웃)</h4>
<p>과적합을 해결할 수 있는 가장 간단한 방법으로는 Dropout이 있음. Dropout은 단어에서도 의미를 유추할 수 있듯이 각 노드들이 이어진 선을 빼서 없애버린다는 의미임.</p>
<img src="/assets/img/three/3_19.png" title="Dropout (드랍아웃)"/>
<p>출처: https://www.researchgate.net/figure/Dropout-Strategy-a-A-standard-neural-network-b-Applying-dropout-to-the-neural_fig3_340700034</p>
<p>오른쪽 그림처럼 각 노드의 연결을 끊어버리는 작업을 하는데, 각 배치마다 랜덤한 노드를 끊어 버린다. 즉 다음 노드로 전달할 때 랜덤하게 출력을 0으로 만들어버리는 것과 같음.<br>
"사공이 많으면 배가 산으로 간다"라는 속담처럼 과적합이 발생했을 때 적당한 노드들을 탈락시켜서 더 좋은 효과를 낼 수 있음.</p>
<img src="/assets/img/three/3_20.png" title="Dropout (드랍아웃)"/>
<p>위와 같이 많은 노드들이 있다면, 이들 중 일부만 사용해도 충분히 결과를 낼 수 있다. 오히려 이들 중에서 충분할 만큼의 노드만 선출해서 반복적으로 결과를 낸다면, 오히려 균형 잡힌 훌륭한 결과가 나올 가능성도 큼</p>
{% capture notice-7 %}  <!--notice-7 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
<p>Dropout은 과적합 발생 시 생각보다 좋은 효과를 냄. (그리고 사용하기도 정말 간단)실무에서 과적합이 발생하면 한 번은 사용하는 걸 추천함!</p>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-7 | markdownify }} <!--div 태그 사이에 notice-7 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>
<h4>Ensemble(앙상블)</h4>
<p>앙상블 기법은 컴퓨팅 파워만 충분하다면 가장 시도해보기 쉬운 방법임. 여러개의 딥러닝 모델을 만들어 각각 학습시킨 후 각각의 모델에서 나온 출력을 기반으로 투표를 하는 방법. 저번 차시에서 설명했던 랜덤 포레스트의 기법과 비슷함.</p>
<img src="/assets/img/three/3_21.png" title="Ensemble(앙상블)"/>
<p>앙상블 또한 개념적으로 이해하는 것이 중요한데 여러개의 모델에서 나온 출력에서 다수결로 투표(Majority voting)를 하는 방법도 있고, 평균값을 구하는 방법도 있고, 마지막에 결정하는 레이어를 붙이는 경우 등 당야한 방법으로 응용이 가능함.<br>
 앙상블을 사용할 경우 최소 2% 이상의 성능 향상 효과를 볼 수 있다고 알려져 있음.</p>
<h4>Learning rate decay (Learning rate schedules)</h4>
<p>Learning rate decay 기법은 실무에서도 자주 쓰는 기법으로 Local minimum에 빠르게 도달하고 싶을 때 사용함.</p>
<img src="/assets/img/three/3_22.png" title="Learning rate decay (Learning rate schedules)"/>
<p>출처: https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/</p>
<p>위 사진에서 보면 왼쪽 그림은 학습의 앞부분에서는 큰 폭으로 건너뛰고 뒷부분으로 갈수록 점점 조금씩 움직엿서 효율적으로 Local minimum을 찾는 모습임. 오른쪽 그림은 Learning rate를 고정시켰을 때의 모습임.</p>
<p>또한 Learning rate decay 기법을 사용하면 Local minimum을 효과적으로 찾도록 도와줌. 아래 그림을 보면 Learning rate가 줄어들 때마다 Error 값이 한 번씩 큰 폭으로 떨어지는 현상을 볼 수 있음.</p>
<img src="/assets/img/three/3_23.png" title="Learning rate decay (Learning rate schedules)"/>
{% capture notice-8 %}  <!--notice-8 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
<p>Keras에서는 tf.keras.callbacks.LearningRateScheduler()와 tf.keras.callbacks.ReduceLROnPlateau() 를 사용하여 학습 중 Learning rate를 조절함.</p>
<a href="https://keras.io/api/callbacks/learning_rate_scheduler/">learning_rate_scheduler</a>
<a href="https://keras.io/api/callbacks/reduce_lr_on_plateau">reduce_lr_on_plateau</a>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-8 | markdownify }} <!--div 태그 사이에 notice-8 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>

<h2>XOR 실습</h2>
<h4>딥러닝으로 XOR 문제 풀어보기</h4>
<p>XOR 실습</p>
<a href = "https://colab.research.google.com/drive/1z8gLF2xwZHgSItojlIV6NE_qGkeAPG9N?usp=sharing">XOR 실습하기</a>

<h2>딥러닝 실습</h2>
<h4>Sign Language MNIST (수화 알파벳) 실습</h4>
<p>Sign Language MNIST 실습</p>
<a href="https://colab.research.google.com/drive/1PqpH9iNpDuJpdI-y8DX2o7r-Nm0MxMAt?usp=sharing">Sign Language MNIST 실습하기</a>

<h2>3주차 끝 & 숙제</h2>
<h4>숫자 MNIST</h4>
<p>머신러닝에서 가장 유명한 데이터셋 중 하나인 MNIST 데이터베이스를 직접 분석하기!</p>
<p>MNIST 데이터베이스(Modified National Institude of Standards and Technology database, 수정된 미국 국립표준기술연구소 데이터베이스)는 손으로 쓴 0~9까지의 숫자 이미지 모음임.</p>
<a href="https://colab.research.google.com/drive/1afOx9qcbta9dYbRXTDz-5FmTp1ESWA1Q?usp=sharing">3주차 숙제: 숫자 MNIST</a>
