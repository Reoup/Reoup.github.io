---
layout: post
title: "스파르타 코딩클럽 머신러닝 2주차"
author: Reo
date: '2021-07-01'
category: Machine-Learning
excerpt: "스파르타 코딩클럽 머신러닝 2주차"
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 2주차, 논리회귀, 전처리, Logistic regression, Sigmoid function, 선형 회귀, binary_crossentropy, 다항 논리 회귀, One-hot encoding, Softmax, categorical_crossentropy, Support Vector Machine, SVM, KNN, Decision tree, Random Forest, 정규화, 표준화]
permalink: /blog/machine_learning-two_week/
use_math: true
---

<h2>01. 2주차 오늘 배울 것</h2>
<h4>논리 회귀</h4>
<p>머신러닝에서, 입력값과 범주 사이의 관계를 구하는 것을 논리 회귀라고 함</p>

<h4>전처리</h4>
<p>실제 업무에서 얻는 데이터는 오기입하는 경우도 있고 단위와 분포가 제각각이기 때문에 정제 작업이 필수임</p>

<h2>02. 논리 회귀 (Logistic regression)</h2>
<p>입력값은 [공부한 시간] 그리고 출력값은 [이수 부여]가 됨 우리는 이수 여부를 0, 1 이라는 이진 클래스로 나눌 수 있음 이것을 선형 회귀를 사용하면</p>

<img src="/assets/img/two/2_1.png" title="선형 회귀"/>
<p><b>Reference</b> https://ursobad.tistory.com/44</p>

<p>이러한 그림이 나옵니다. 따라서 또 다른 방법이 필요했는데 바로 Logistic function(=Sigmoid function)</p>

<img src="/assets/img/two/2_2.png" title="논리 회귀"/>
<p><b>Reference</b> https://ursobad.tistory.com/44</p>

<p>로지스틱 함수는 입력값(x)으로 어떤 값이든 받을 수가 있지만 출력 결과(y)는 항상 0~1 사이 값이 된다</p>
<p>→ 자연, 사회현상에서 특정 변수에 대한 확률값이 선형이 아닌 S 커브 형태를 따르는 경우가 많기 때문에 s-커브 함수로 표현한 것이 바로 로지스틱 함수이다. 딥러닝에서는 이것을 시그모이드 함수(Sigmoid function)이라고 부른다.</p>

<h4>가설과 손실함수</h4>

<p>실질적인 계산은 선형 회귀와 비슷하지만, 출력에 시그모이드 함수를 붙이면서 0에서 1사이의 값을 가지도록 하게 됩니다.</p>

<img src="/assets/img/two/2_3.png" title="시그모이드 함수"/>

<h4>시그모이드 함수</h4>
<p>x(입력)가 음수 방향으로 갈 수록 y(출력)가 0에 가까워지며,<br>
x(입력)가 양수 방향으로 갈 수록 y(출력)가 1에 가까워진다.<br>
즉, 시그모이드 함수를 통과하면 0에서 1사이의 값이 나온다.</p>
{: .notice}

<p>선형 회귀에서의 가설은</p> 
$$
\begin{align*}
H(x,y)= Wx + b
\end{align*}
$$
<p>논리 회귀에서는 시그모이드 함수에 선형 회귀식을 넣어주면 됨</p>

<img src="/assets/img/two/2_4.png" title="선형 회귀+시그모이드"/>
$$
\begin{align*}
H(x,y)= \frac{1}{1+e^-(Wx+b)}
\end{align*}
$$
<p>논리 회귀에서 손실 함수는 아래와 같은 어려운 수식이 되지만, 수식보다는 개념을 이해하는 것이 목표</p>
$$
\begin{align*}
-\frac{1}{M}\sum_{i=1}^N[y^{(i)}log(h(z^{(i)}))+(1-y^{(i)})log(1-h(z^{(i)}))]
\end{align*}
$$
<img src="/assets/img/two/2_5.png" title="정답이 0인 경우와 정답이 1인 경우1"/>
<p>예측한 라벨이 0일 경우 확률이 1(=100%)이 되도록 해야하고, 예측한 라벨이 1일 경우 확률이 0(=0%)이 되도록 만들어야 한다.<br>반대로 정답 라벨  y가 1일 경우, 예측한 라벨 1일 때 확률이 1(=100%) 되도록 만들어야 함</p>
<p>위의 그래프를 실제적으로 그리면 아래의 그래프처럼 된다.</p>
<img src="/assets/img/two/2_6.gif" title="정답이 0인 경우와 정답이 1인 경우2"/>
<p><b>출처: </b>https://machinelearningknowledge.ai/cost-functions-in-machine-learning/</p>
<p>가로축을 라벨(클래스)로 표시하고 세로축을 확률로 표시한 그래프를 확률 분포 그래프라고 한다. 확률 분포 그래프의 차이를 비교할 때는 Crossentropy 라는 함수를 사용합니다.<br>
임의의 입력값에 대해 원하는 확률 분포 그래프를 만들도록 학습시키는 것이 손실 함수이다.</p>
<img src="/assets/img/two/2_7.png" title="정답이 0인 경우(y=0)"/>
<p>현재 학습 중인 입력값의 확률 분포가 파란색 그래프처럼 나왔다고 가정했을 때, crossentropy는 파란색 그래프를 빨간색 그래프처럼 만들어주기 위해 노력하는 함수이다.</p>
<span>Keras에서 이진 논리 회귀의 경우</span> **binary_crossentropy**<span> 손실 함수를 사용함! </span>
{: .notice}

<h2>03. 다항 논리 회귀(Multinomial logistic regression)</h2>
<h4>다항 논리 회귀와 One-hot encoding</h4>
<p>클래스를 5개의 클래스로 나누는 이 방법을 다중 논리 회귀라고 부른다.</p>
<img src="/assets/img/two/2_8.png" title="성적 클래스 분류"/>
<p>원핫 인코딩은 다항 분류 문제를 풀 때 출력값의 형태를 가장 예쁘게 표현할 수 있는 방법이다.</p>
<img src="/assets/img/two/2_9.png" title="성적 클래스 분류 - One-hot encoding"/>
<ol>원핫 인코딩을 만드는 방법은 아래와 같다.
  <li>클래스(라벨)의 개수만큼 배열을 0으로 채운다.</li>
  <li>각 클래스의 인덱스 위치를 정한다.</li>
  <li>각 클래스에 해당하는 인덱스에 1을 넣는다.</li>
</ol>
<h4>Softmax 함수와 손실함수</h4>
<p>Softmax는 선형 모델에서 나온 결과(Logit)를 모두가 더하면 1이 되도록 만들어주는 함수이다.<br>
이유는 예측의 결과를 확률(=Confidence)로 표현하기 위함인데, One-hot encoding을 할 때에도 라벨의 값을 전부 더하면 1(100%)가 되기 때문이다.</p>
<img src="/assets/img/two/2_10.png" title="Softmax 함수1"/>
<p><b>출처: </b>https://www.tutorialexample.com/implement-softmax-function-without-underflow-and-overflow-deep-learning-tutorial/</p>
<img src="/assets/img/two/2_11.png" title="Softmax 함수2"/>
<p><b>출처: </b>https://www.programmersought.com/article/62574848686/</p>
<p>다항 논리 회귀에서 Softmax 함수를 통과한 결과 값의 확률 분포 그래프를 그려서 아래 그래프의 모양이라고 가정을 하면, 단항 논리 회귀에서처럼 가로축은 클래스(라벨)이 되고 세로축은 확률이 된다.<br>
마찬가지로 확률 분포의 차이를 계산할 때 Crossentropy 함수를 쓰고 항이 여러개가 되었을 뿐 차이는 이진 논리 회귀와 차이가 없다. 데이터셋의 정답 라벨, 예측한 라벨의 확률 분포 그래프를 구해서 Crossentropy로 두 확률 분포의 차이를 구한 다음 그 차이를 최소화하는 방향으로 학습을 시킨다.</p>
<img src="/assets/img/two/2_12.jpg" title="Crossentropy로 최소화하는 방향 학습"/>
<span>Keras에서 이진 논리 회귀의 경우</span> **categorical_crossentropy**<span> 손실 함수를 사용함! </span>
{: .notice}

<h2>04. 다양한 머신러닝 모델</h2>
<h4>Support Vector Machine(SVM)</h4>
<p>이번에는 강아지, 고양이를 구분하는 문제를 푼다고 가정. 구분하는 문제를 푸는 것은 분류 문제이고 분류 문제를 푸는 모델을 분류기(Classifier)라고 부른다. 
<img src="/assets/img/two/2_13.png" title="강아지와 고양이 분류한 SVM"/>
<P>각 그래프의 축을 Feature(특징)이라고 부르고 각 고양이, 강아지와 빨간 벡터의 거리를 Support Vector라고 부른다. 그 벡터의 거리를 Margin이라고 부르고 Margin이 넓어지도록 이 모델을 학습시켜 좋은 Support Vector </P>

{% capture notice-2 %}  <!--notice-2 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
#### 예외 상황 발생!

만약 충성심이 강한 개냥이가 등장한다면 어떻게 할까요? 그럴 경우에는 Feature(특성)의 개수를 늘려서 학습시키는 것이 일반적입니다. 현재는 "귀의 길이", "충성심" 의 2개의 특성만 있지만, "목소리의 굵기"라는 특성을 추가시켜 3차원 그래프를 그려 구분할 수 있겠죠! 이것이 바로 분류 문제의 기초 개념이고 딥러닝을 더 낮은 차원에서 이해할 수 있는 방법입니다.
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->

<div class="notice">
  {{ notice-2 | markdownify }} <!--div 태그 사이에 notice-2 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>

<h4>기타 머신러닝 모델 간단한 소개</h4>
<p><b>k-Nearest neighbors (KNN)</b></p>
<img src="/assets/img/two/2_14.png" title="KNN"/>
<p>KNN은 비슷한 특성을 가진 개체끼리 군집화하는 알고리즘이다.</P>
<p><b>Decision tree (의사결정나무)</b></p>
<img src="/assets/img/two/2_15.png" title="의사결정나무1"/>
<p>스무고개와 같은 방식으로 예, 아니오를 반복하며 추론하는 방식(생각보다 성능이 좋아 간단한 문제를 풀 때 자주 사용함)</p>
<img src="/assets/img/two/2_16.png" title="의사결정나무2"/>
<p><b>Random forest</b></p>
<img src="/assets/img/two/2_17.png" title="랜덤 포레스트"/>
<p>의사결정나무를 여러개 합친 모델. 의사결정나무는 한 사람이 결정하는 것이라고 한다면 랜덤 포레스트는 각각의 의사결정나무들이 결정을 하고 마지막에 투표를 통해 최종 답을 결정한다.</p>

<h2>05. 머신러닝에서의 전처리</h2>
<h4>전처리(Preprocessing)란?</h4>
<p>전처리는 넓은 범위의 데이터 정제 작업을 뜻함. 필요없는 데이터를 지우고 필요한 데이터만을 취하는 것, null 값이 있는 행은 삭제하고, 정규화(Normalization), 표준화(Standardization) 등의 많은 작업들을 포함함<br>
또한 머신러닝 실무에서도 전처리가 80%를 차지할 정도로 중요한 작업이고 전처리 노가다에서 많은 시간과 실수가 많다.
<h4>정규화(Normalization)</h4>
<p>데이터를 0과 1사이의 범위를 가지도록 만듦. 같은 특성의 데이터 중에서 가장 작은 값을 0으로 만들고, 가장 큰 값을 1로 만듦

$$
\begin{align*}
H(x,y)= \frac{X-X_{최소}}{X_{최대}-X_{최소}}
\end{align*}
$$

<h4>표준화(Standardization)</h4>
<p>데이터의 분포를 정규분포로 바꿔줌. 즉 데이터의 평균이 0이 되도록하고 표준편차가 1이 되도록 만들어주는 형식</p>

$$
\begin{align*}
H(x,y)= \frac{X-X_{평균}}{X_{표준편차}}
\end{align*}
$$

<p>데이터의 평균을 0으로 만들어주면 데이터의 중심이 0에 맞춰지도록 (Zero-centered)가 된다.<br>
그리고 표준편차를 1로 만들어 주면 데이터가 예쁘게 정규화(Normalized)가 된다. 이렇게 표준화를 시키게 되면 일반적으로 학습 속도(최저점 수렴 속도)가 빨라지고, Local minima에 빠질 가능성도 적어진다.
<img src="/assets/img/two/2_18.png" title="표준화"/>
<p><b>출처: </b>http://cs231n.stanford.edu/2016/</p>
<p>아래의 그래프에서 정규화와 표준화의 차이를 대략적으로나마 느낄 수 있다.</p>
<img src="/assets/img/two/2_19.png" title="정규화와 표준화의 차이"/>
<p><b>출처: </b>https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/</p>

<h2>06. 이진 논리회귀 실습</h2>
{% highlight html %}
https://colab.research.google.com/drive/1ElLU8rpS1vcbq3U75dcyQJe7GHNad8Bb?usp=sharing
{% endhighlight %}
<p>[코드스니펫] - 타이타닉 생존자 예측하기</p>
<a href="https://colab.research.google.com/drive/1VJWKNcGBe1u-xu3M12O8MYwSBcNri4ni?usp=sharing">타이타닉 실습</a>

<h2>07. 다항 논리회귀 실습</h2>
{% highlight html %}
https://colab.research.google.com/drive/1FuzzhcnIOzUcZQ7soSfq0IOxZw5Uos5z?usp=sharing
{% endhighlight %}
<p>[코드스니펫] - 와인 종류 예측하기</p>
<a href="https://colab.research.google.com/drive/1WFD_E911hf5GpmnNsizfW4cUpRjN-EGj?usp=sharing">와인 실습</a>

<h2>08. 2주차 끝 & 숙제 설명</h2>
<p>이진 논리회귀 직접 해보기</p>
{% capture notice-2 %}  <!--notice-2 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
숙제로는 연령, 혈압, 인슐린 수치 등을 통해 당뇨병을 진단해보기!<br>
<a href="https://www.kaggle.com/kandij/diabetes-dataset">https://www.kaggle.com/kandij/diabetes-dataset</a>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-2 | markdownify }} <!--div 태그 사이에 notice-2 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>
<p>이진 논리회귀 실습</p>
<a href="https://colab.research.google.com/drive/16qEdJQ3GraXg22Tts9EvlNvDnrOu_cb5?usp=sharing">이진 논리회귀 숙제</a>
<p>다항 논리회귀 실습</p>
<a href="https://colab.research.google.com/drive/1weHzatlDcDuXfzlnRv1pS2NGKaTYBvOe?usp=sharing">다항 논리회귀 숙제</a>

