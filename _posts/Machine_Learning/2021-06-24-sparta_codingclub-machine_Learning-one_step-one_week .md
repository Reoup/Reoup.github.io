---
layout: post
title:  "스파르타 코딩클럽 머신러닝 1주차"
author: Reo
date: '2021-06-24'
category: Machine-Learning
excerpt: "한걸음 더 알아보기"
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 1주차, Optimizer, keras, 머신러닝, mean_squared_error, mean_absolute_error]
permalink: /blog/machine_learning-one_step-one_week/
usemathjax: true
---
# [한걸음 더] 다양한 Optimizers 살펴보기
<p>참고: <a href="https://keras.io/ko/optimizers/">Optimizers 사용법</a><br>
keras는 기본적으로 다양한 Optimizers를 제공한다. 생성한 모델이나 데이터셋에 따라 다른 Optimizer를 사용해야 하는데, 보통 머신러닝 엔지니어들은 이 과정을 여러번의 실험 <strike>노가다</strike>를 통해 알아낸다.</p>
<img src="/assets/img/one/Optimizers.gif" title="Optimizers" alt="오류뜨지마">

# [한걸음 더] keras기본 손실 함수 살펴보기
<p>참고: <a href="https://keras.io/ko/losses/">kerass 기본 손실 함수</a><br>
가설에 따라 다양한 손실 함수를 사용하는데 1주차 강의에서는 mean_squared_error를 사용했음. 선형회귀 문제에서는 mean_squared_error 대신에 mean_absolute_error를 사용할 수 있습니다.</p>