---
layout: post
title: "스파르타 코딩클럽 머신러닝 4주차"
author: Reo
date: '2021-07-18'
category: Machine-Learning
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 4주차, Convolution Neural Networks, 합성곱 신경망, 물체 인식, 이미지 분할, AlexNet, VGGNet, GoogLeNet, Inception V3, ResNet, Transfer Learning, 전이 학습, Recurrent Neural Networks, 순환 신경망, Generative Adversarial Network, 생성적 적대 신경망]
permalink: /blog/machine_learning-four_week/
usemathjax: true
---

<h2>이번 주차 배울 것</h2>
<h4>다양한 신경망 구조</h4>
<p>신경망을 구성하는 방법은 여러가지가 있다. 이 중 가장 많이 쓰는 합성곱 신경망(CNN), 순환 신경망(RNN), 생성적 적대 신경망(GAN)에 대해 알아볼 것이다.<br>
특히 이미지 처리에서 많이 쓰이는 CNN에 대해서 알아 볼 예정이다.</P>
<img src="/assets/img/four/4_1.png" title="CNN(합성곱 신경망)"/>
<p>출처: https://www.cnblogs.com/wangxiaocvpr/p/6247424.html</p>
<h4>전이학습</h4>
<p>이미 학습된 모델을 비슷한 문제를 푸는데 다시 사용하는 것이 바로 전이학습이라고 함<br>
더 적은 데이터로 더 빠르고 더 정확하게 학습시킬 수 있어 실무에서도 많이 쓰이는 방법이라고 합니다.</p>

<h2>Convolutional Neural Networks (합성곱 신경망)</h2>
<h4>합성곱과 합성곱 신경망</h4>
<p>합성곱(Convolution)은 예전부터 컴퓨터 비전(Computer Vision, CV) 분야에서 많이 쓰이는 이미지 처리 방식으로 계산하는 방식은 아래와 같다. 입력데이터와 필터의 각각의 요소를 서로 곱한 후 다 더하면 출력값이 된다.</P>
<img src="/assets/img/four/4_2.png" title="CNN(합성곱 신경망)"/>
<P>출처: https://ce-notepad.tistory.com/14</P>
<P>딥러닝 연구워들은 이 합성곱을 어떻게 딥러닝에 활용할 수 있을 지 고민하다가, 1998년 Yann LeCun 교수님이 대단한 <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf">논문</a>을 발표하게 된다.</p>
<img src="/assets/img/four/4_3.png" title="르쿤 교수님의 CNN"/>
<P>합성곱을 이용한 이 신경망 디자인을 합성곱 신경망(CNN)이라고 명칭하였고 이미지 처리에서 엄청난 성능을 보이는 것을 증명하였다. CNN의 발견 이후 딥러닝은 전성기를 이루었고 이후 CNN은 얼굴 인식, 사물 인식 등에 널리 사용되며 현재도 이미지 처리에서 가장 보편적으로 사용되는 네트워크 구조이다.</P>
<h4>Filter, Strides and Padding</h4>
<h5>합성곱 신경망에서 가장 중요한 합성곱 계층(Convolution layer)이다.</h5>
<p>아래와 같이 5x5의 크기의 입력이 주어졌을 때, 3x3짜리 필터를 사용하여 합성곱을 하면 3x3 크기의 특성맵(Feature map)을 뽑아낼 수 있다. 필터(Filter 또는 Kernel)를 한 칸씩 오른쪽으로 움직이며 합성곱 연산을 하는데, 이 때 이동하는 간격을 스트라이드(Stride)라고 한다.</p>
<img src="/assets/img/four/4_4.gif" title="Feature map1"/>
<p>출처: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1</p>
<p>하지만 이렇게 연산을 하게 된다면 합성곱 연산의 특성상 출력값인 특성 맵의 크기가 줄어드게 된다. 이런 현상을 방지하기 위해서 패딩(Padding 또는 Margin)을 주어, 스트라이드가 1일 때 입력값과 특성 맵의 크기를 같게 만들 수 있다.</p>
<img src="/assets/img/four/4_5.gif" title="Feature map2"/>
<p>출처: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1</p>
<p>위에서는 1개의 필터를 사용하여 연산을 하였지만 여러개의 필터를 이용하여 합성곱 신경망의 성능을 높일 수 있다. 그리고 이미지는 3차원(가로, 세로, 채널)이기 때문에 아래와 같은 모양이 된다. 이 그림에서 각각의 입력과 출력은 다음과 같다.</p>
<ul>
  <li>입력 이미지 크기: (10, 10, 3)</li>
  <li>필터의 크기: (4, 4, 3)</li>
  <li>필터의 개수: 2</li>
  <li>출력 특성 맵의 크기: (10, 10, 2)</li>
</ul>
<img src="/assets/img/four/4_6.gif" title="Feature map3"/>
<p>출처: https://stackoverflow.com/questions/42883547/intuitive-understanding-of-1d-2d-and-3d-convolutions-in-convolutional-neural-n</p>

<h2>CNN의 구성</h2>
<h4>CNN의 구성</h4>
<p>합성곱 신경망은 합성곱 계층(Convolution layer)과 완전연결 계층(Dense layer)을 함께 사용한다.</p>
<img src="/assets/img/four/4_7.png" title="Convolution-Neural-Networks"/>
<p>출처: https://teknoloji.org/cnn-convolutional-neural-networks-nedir/</p>
<p>합성곱 계층 + 활성화 함수 + 풀링을 반복하며 점점 작아지지만 핵심적인 특성들을 뽑아 내는데, 여기서 풀링 계층은 특성 맵의 중요부분을 추출하여 저장하는 역할을 한다.<br>
아래의 이미지는 Max pooling의 예시이다. 2x2 크기의 풀 사이즈(Pool size)로 스트라이드 2의 Max pooling 계층을 통과할 경우 2x2 크기의 특성 맵에서 가장 큰 값들을 추출한다.</p>
<img src="/assets/img/four/4_8.gif" title="Max pooling"/>
<p>출처: https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks</p>
<p>아래는 Average pooling의 예시이다. Max pooling에서는 2x2 크기의 특성 맵에서 최대 값을 추출했다면 Average pooling은 2x2 크기의 특성 맵에서 평균 값을 추출하는 방식이다.</p>
<img src="/assets/img/four/4_9.png" title="Average pooling"/>
<p>출처: https://www.kaggle.com/questions-and-answers/59502</p>
<p>Max pooling과 Average pooling의 결과 비교 ↓</p>
<img src="/assets/img/four/4_10.png" title="Max pooling과 Average pooling의 비교"/>
<p>출처: https://towardsdatascience.com/beginners-guide-to-understanding-convolutional-neural-networks-ae9ed58bb17d</p>
<p>두 번째 풀링 계층을 지나면 완전연결 계층과 연결이 되어야 하는데 풀링을 통과한 특성 맵은 2차원이ㅏ고 완전연결 계층은 1차원이므로 연산이 불가능하다.</p>
<img src="/assets/img/four/4_7.png" title="Convolution-Neural-Networks"/>
<p>출처: https://teknoloji.org/cnn-convolutional-neural-networks-nedir/</p>
<p>따라서 평탄화 계층(Flatten layer)를 사용해서 2차원을 1차원으로 펼치는 작업을 하게 된다. 아래에는 간단하게 평탄화 계층의 동작을 설명하는 그림이다.</p>
<img src="/assets/img/four/4_11.png" title="Flatten layer"/>
<p>출처: https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-3-flattening</p>
<p>평탄화 계층을 통과하게 되면 완전연결 계층에서 행렬 곱셉을 할 수 있게 되고 마찬가지로는 완전연결 계층(=Dense=Fully connected) + 활성화 함수의 반복을 통해 점점 노드의 개수를 축소시키다가 마지막에 Softmax 활성화 함수를 통과하고 출력층으로 결과를 출력하게 된다.</p>
<h3>CNN의 활용 예</h3>
<h4>물체 인식(Object Detection)</h4>
<img src="/assets/img/four/4_12.png" title="Object Detection"/>
<p>출처: https://arxiv.org/abs/1612.04402v1</p>
<p>Object Detection은 사진 이미지에서 정확히 물체를 인식하는 것을 뜻하며 컴퓨터 비전에서 가장 중요한 기술이다. 각각의 객체를 정확하게 인식하는 것부터 Computer Vision이 시작하기 때문이다.</p>
<h4>이미지 분할(Segmentation)</h4>
<img src="/assets/img/four/4_13.png" title="Segmentation1"/>
<ul>
<li>나누는 기준이 디테일 할수록 정교화된 성능을 가져야 하고 처리속도 또한 문제가 될 수 있다.<br>
Segmentation의 Class를 동물로 분리할 수 있고, 강아지와 고양이로 분리할 수 있다. 더욱 더 세분화해서 분류를 하게 되면 강아지 중에서도 다양한 종들로 분리할 수 있다.</li>
<li>인물과 배경을 Segmentation하여 배경은 흐릿하게 처리하고 인물을 Focus 할 수 있는 기술이다.</li>
<img src="/assets/img/four/4_14.png" title="Segmentation2"/>
<li>의료영상에서도, 양성/음성부분을 파악하고 악성인 부분을 Segmentation하여 인식할 수 있도록 도와준다.</li>
<img src="/assets/img/four/4_15.png" title="Segmentation3"/>
<h3>활용 예</h3>
<ul>
  <li>자율주행 물체 인식</li>
  <li>자세 인식</li>
  <li>화질 개선</li>
  <li>Style Transfer</li>
  <li>사진 색 복원</li>
</ul>

<h2>다양한 CNN 종류</h2>
<h4>다양한 CNN의 종류</h4>
<img src="/assets/img/four/4_16.png" title="Operation"/>
<p>출처: https://www.researchgate.net/figure/Results-shown-in-Canziani-et-al2016-that-compare-model-accuracy-vs-operation-count_fig5_339199431</p>
<ul>
  <li>AlexNet(2012)</li>
  <img src="/assets/img/four/4_17.png" title="AlexNet"/>
  <p>출처: https://paperswithcode.com/method/alexnet</p>
  <p>AlexNet은 의미있는 성능을 낸 첫 번째 합성곱 신경망이였고, Dropout과 Image augmentation 기법을 효과적으로 적용하여 딥러닝에 많은 기여를 했음</p>
  <li>VGGNet(2014)</li>
  <img src="/assets/img/four/4_18.png" title="VGGNet"/>
  <p>출처: https://medium.com/deep-learning-g/cnn-architectures-vggnet-e09d7fe79c45</p>
  <p>VGGNET은 큰 특징은 없는데 엄청 Deep한 모델(파라미터의 개수가 많고 모델의 깊이가 깊음)로 잘 알려져 있다. 또한 요즘에도 딥러닝 엔지니어들이 처음 모델을 설계할 때 전이 학습 등을 통해서 가장 먼저 테스트하는 모델이기도 하다. 간단한 방법론으로는 좋은 성적을 내서 유명해졌다.</p>
  <li>GoogLeNet(=Inception V3)(2015)</li>
  <p>구글에서 개발한 합성곱 신경망 구조이다. AlexNet 이후 층을 더 깊게 쌓아 성능을 높이려는 시도들이 계속되었고 그게 바로 VGGNet, GoogLeNet이 대표적인 사례이다. GoogLeNet은 VGGNet 보다 구조가 복잡해 널리 쓰이진 않지만 구조 면에서는 주목을 받았다.<br>
  GoogLeNet 연구진들은 한 가지의 필터를 적용한 합성곱 계층을 단순히 깊게 쌓는 방법도 있지만, 하나의 계층에서도 다양한 종류의 필터, 풀링을 도입함으로써 개별 계층을 두텁게 확장 시킬 수 있다는 창조적인 아이디어로 떠오른 구조가 Inception module(인셉션 모듈)이다.</p>
  <img src="/assets/img/four/4_19.png" title="Inception module"/>
  <p>출처: https://tariq-hasan.github.io/concepts/computer-vision-cnn-architectures/</p>
  <p>인셉션 모듈에서 주의깊게 보아야할 점은 차원(채널) 축소를 위한 1x1 합성곱 계층 아이디어이다. 또한 여러 계층을 사용하여 분할하고 합치는 아이디어는, 갈림길이 생김으로써 조금 더 다양한 특성을 모델이 찾을 수 있게 하고, 인공지능이 사람이 보는 것과 비슷한 구조로 볼 수 있게 한다. 이러한 구조로 인해 VGGNet 보다 신경망이 깊어졌음에도, 사용된 파라미터는 절반 이하로 줄어들었음</p>
  <li>ResNet(2015)</li>
  <p>AlexNet이 처음 제안된 이후로 합성곱 신경망의 계층은 점점 더 깊어져 갔다. AlexNet이 불과 5개 계층에 불과한 반면 VGGNet은 19계층, GoogLeNet은 22개 계층에 달한다. 하지만 층이 깊어질 수록 역전파의 기울기가 점점 사라져서 학습이 잘 되지 않는 문제(Gradient vanishing)가 발생했다. ResNet 저자들이 제시한 아래 학습 그래프를 보면 이같은 문제가 뚜렷이 나타나는 것을 볼 수 있다.
  <img src="/assets/img/four/4_20.png" title="ResNet"/>
  <p>출처: https://neurohive.io/en/popular-networks/resnet/</p>
  <p>따라서 ResNet 연구진은 Residual block을 제안한다. 그래디언트가 잘 흐를 수 있도록 일종의 지름길(Shortcut=Skip connection)을 만들어 주는 방법이다.</p>
  <img src="/assets/img/four/4_21.png" title="Shortcut=Skip connection"/>
  <p>출처: https://neurohive.io/en/popular-networks/resnet/</p>
  <p>위의 그림에서 알 수 있듯 y = F(x) + X 를 다시 쓰면  F(x) = y - x로 표현할 수 있고, Residual block은 입력과 출력 간의 차이를 학습하도록 설계되어 있다.<br>
  ResNet의 Residual block은 합성곱 신경망 역사에서 큰 영향을 끼쳤고 아직도 가장 많이 사용되는 구조 중에 하나이다. 많은 사람들이 Residual block을 사용하면 대부분의 경우 모델의 성능이 좋아진다라고 한다.</p>
</ul>

<h2>Transfer Learning(전이 학습)</h2>
<h4>전이 학습</h4>
<p>전이 학습이라는 개념은 인간이 학습하는 방법을 모사하여 만들어졌다. 과거에 문제를 해결하면서 축적된 경험을 토대로 그것과 유사한 문제를 해결하도록 신경망을 학습시키는 방법을 전이 학습이라고 한다. 전이 학습은 비교적 학습 속도가 빠르고(빠른 수렴), 더 정확하고, 상대적으로 적은 데이터셋으로 좋은 결과를 낼 수 있기 때문에 실무에서도 자주 사용하는 방법이다.</p>
<img src="/assets/img/four/4_22.png" title="전이 학습"/>
<p>출처: https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a</p>
{% capture notice-1 %}  <!--notice-1 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->  
<p>Transfer Learning will be the next driver of ML commerical success after Supervised Learning.<br>
전이 학습은 지도 학습 이후로 머신러닝의 상업적 성공에 가장 큰 기여를 할 것이다.</p>
<p><b>- Andrew Ng, Baidu Research</b></p>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-1 | markdownify }} <!--div 태그 사이에 notice-1 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>
<p>전이학습은 위에서 소개한 유명한 네트워쿼들과 같이 미리 학습시킨 모델(pretrained models)을 가져와 새로운 데이터셋에 대해 다시 학습시키는 방법이다. 예를 들어, 1000개의 동물/사물을 분류하는 ImageNet이라는 대회에서 학습한 모델들을 가져와 얼굴 인식 데이터셋에 학습시켜도 좋은 결과를 얻을 수 있다. 이런 특징 덕분에 전이 학습은 딥러닝에서 더욱 중요하게 되었다.</p>

<h2>Recurrent Neural Networks(순환 신경망)</h2>
<h4>Recurrent Neural Networks (RNN)</h4>
<p>RNN은 은닉층이 순차적으로 연결되어 순환구조를 이루는 인공신경망의 한 종류이다. 음성, 문자 등 순차적으로 등장하는 데이터 처리에 적합한 모델로 알려져 있으며, 합성곱 신경망과 더불어 최근 들어서는 각광 받고 있는 신경망 구조이다.<br>
길이에 관계없이 입력과 출력을 받아들일 수 있는 구조이기 떄문에 필요에 따라 다양하고 유연하게 구조를 만들 수 있다는 점이 RNN의 큰 장점이다.</p>
<img src="/assets/img/four/4_23.png" title="RNN1"/>
<p>출처: https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/</p>
<p>우리가 <a href="https://www.youtube.com/watch?v=6eGyNifjcvg">소설을 지어내는 인공지능</a>을 만든다고 할 떄, hell 이라는 입력을 받으면 ello라는 출력을 만들어내게 해서 결과적으로 hello 라는 순차적인 문자열을 만들어 낼 수 있게 하는 아주 좋은 구조이다.</p>
<img src="/assets/img/four/4_24.png" title="RNN2"/>
<p>출처: https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/</p>
<p>이 외에도 <a href="https://youtu.be/sG_WeGbZ9A4">주식이나 암호화폐의 시세를 예측</a>한다던지, 사람과 대화하는 챗봇을 만드는 등의 다양한 모델을 만들 수 있다.</p>

<h2>Generative Adversarial Network(생성적 적대 신경망)</h2>
<h4>Generative Adversarial Network(GAN)</h4>
{% capture notice-2 %}  <!--notice-2 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->
<p>서로 적대(Adversarial)하는 관계의 2가지 모델(생성 모델과 판별 모델)을 동싱 ㅔ사용하는 기술이다. 최근 딥러닝 학계에서 굉장히 핫한 분야이다.</p>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-2 | markdownify }} <!--div 태그 사이에 notice-2 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>
<img src="/assets/img/four/4_25.png" title="GAN1"/>
<p>GAN은 위조지폐범과 이를 보고 적발하는 경찰의 관계로 설명할 수 있다.</p>
<ul>
  <li>생성모델(위조지폐범): 경찰도 구분 못하는 진짜같은 위조지폐 만들기</li>
  <li>판별모델(경찰): 진짜 지폐와 위조 지폐를 잘 구분하기</li>
  <p>이와 같이 계속 진행되면 위조지폐범은 더욱 더 정교하게, 경찰은 더욱 더 판별을 잘하면서 서로 발전의 관계가 되어 원본과 구별이 어려운 가짜 이미지가 만들어지게 된다.</p>
</ul>
<h5>GAN이 어떻게 작용하는지 살펴보기</h5>
{% capture notice-3 %}  <!--notice-3 라는 변수에 다음 텍스트 문단을 문자열로 저장한다.-->
<p>GAN에 대해 Input data와 Output data를 파악하고 위조지폐범과 경찰 입장에서 한번 생각해보기</p>
{% endcapture %}  <!--캡처 끝! 여기까지의 텍스트를 변수에 저장-->
<div class="notice">
  {{ notice-3 | markdownify }} <!--div 태그 사이에 notice-3 객체를 출력하되 markdownify 한다. 즉 마크다운 화-->
</div>
<img src="/assets/img/four/4_26.png" title="GAN2"/>
<p>AnimalGAN이라는 머신이 어떻게 잡음으로부터 동물 이미지를 만드는지 봄으로써 GAN의 작동방식을 이해합시다. 이 머신에게 주어진 문제는 다음과 같습니다.</p>
<ul>
  <li>Input Data: 랜덤으로 생성된 잡음</li>
  <li>Output Data: 0~1 사이의 값(0은 가짜, 1은 진짜)</li>
</ul>
<p>이때 대립하는 두 모델은 다음과 같다.</p>
<ul>
  <li>Generator(위조지폐범): 이미지가 진짜(1)로 판별될 수 있도록 보다 정교하게 모델을 만들기 위해 노력하며 Target은 1로 나오도록 해야한다. 가짜를 진짜인 1처럼 만들기 위해선 타깃인 1과 예측의 차이인 손실을 줄이기 위해 Backpropagation을 이용한 weight를 조정할 것이다.</li>
  <li>Discriminator(경찰): 진짜 이미지는 1로, 가짜 이미지는 0으로 판별될 수 있어야 한다. 생성된 모델에서 Fake와 Real 이미지 둘다를 학습하여 예측과 타깃의 차이인 손실을 줄여야 한다.</li>
</ul>
<p>이렇게 두 모델이 대립하면서(Adversarial) 발전해 에폭(Epoch)이 지날 때마다 랜덤 이미지가 점점 동물을 정교하게 생성해 내는 것(Generative)을 볼 수 있다.</p>
<img src="/assets/img/four/4_27.png" title="GAN3"/>
<h5>GAN을 사용한 예시들</h5>
<ul>
  <li>CycleGAN</li>
  <li>StarGAN</li>
  <li>CartoonGAN</li>
  <li>DeepFake</li>
  <li>BeautyGAN</li>
  <li>Toonify Yourself</li>
</ul>

<h2>CNN 실습</h2>
<h4>수화 MNIST CNN으로 학습해보기</h4>
<a href="https://colab.research.google.com/drive/1x2SRHEAdRqNHTMKvVn8oUSwF9KV9Vi4C?usp=sharing#scrollTo=I_strLH75R_x">실습</a>

<h2>전이학습 실습</h2>
<h4>과일 분류에 전이학습 적용해보기</h4>
<a href="https://colab.research.google.com/drive/1hHPsoG6F80ff6WmnFsqfzKLZTSxWZ3fQ?usp=sharing#scrollTo=NoVKiZa7z7U2">실습</a>

<h2>숙제</h2>
<h4>전이학습 기법을 이용해 RestNet 모델을 가져와 빌딩, 숲, 빙하, 산 등의 사진을 분류해보기</h4>
<a href="https://colab.research.google.com/drive/1liuyqHkZSOvEYYup40TWUZkEc8YjWp0u?usp=sharing">숙제</a>