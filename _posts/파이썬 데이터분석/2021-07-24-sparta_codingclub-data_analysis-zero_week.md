---
layout: post
title:  "스파르타 코딩클럽 파이썬 데이터분석 1주차"
author: Reo
date: '2021-07-24'
category: Data-Analysis
excerpt: "스파르타 코딩클럽 파이썬 데이터분석 1주차"
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 1주차, list, dictionary, set, 조건문, 반복문, 함수, Pandas, Matplotlib]
permalink: /blog/data_analysis-one_week/
usemathjax: true
---

{% highlight python %}
a = 3 # 3을 a에 넣는다.
b = a # a를 b에 넣는다.
a = a+1 # a+1을 다시 a에 넣는다.
num1 = a*b # a*b의 값을 num1이라는 변수에 넣는다.
num2 = 99 # 99의 값을 num2이라는 변수에 넣는다.
{% endhighlight %}

{% highlight python %}
# 숫자, 문자열, 참거짓
num = 12 # 숫자가 들어갈 수도 있고,
name = 'Harry' # 변수에는 문자열이 들어갈 수도 있고,
number_status = True # True 또는 False -> "Boole" 형이 들어갈 수도 있다.

# 리스트 형
waiting_list = [] # 비어있는 리스트 만들기
waiting_list.append('원영준') # 리스트에 문자열 데이터를 넣는다.
waiting_list.append('김진원') # 리스트에 '김진원'라는 문자열을 하나 더 넣는다.
waiting_list.append(['이시연','장선진']) # 리스트에는 또 다른 리스트가 추가될 수 있다.

# Dictionary 형
eng_kor_dict = {} # 비어있는 딕셔너리 만들기
eng_kor_dict = {'apple': '사과', 'pear': '배'}
eng_kor_dict['banana'] = '바나나' # 딕셔너리에 추가하기

# Set 형
group1 = set([1, 2, 3, 4, 2, 1]) # 1, 2, 3, 4
group2 = set([1, 2, ,3 ,1, 6]) # 1, 2, 3, 6
group1 & group2 # 1, 2, 3
group1 | group2 # 1, 2, 3, 4, 6
{% endhighlight %}
<p>리시트에 있는 데이터에 접근할 때 list_name[0]와 같은 방법으로 접근한다.<br>
딕셔너리에 있는 데이터에 접근할 때는 dictionary_name["키값"]의 방법을 이용한다.<br>
셋은 리스트를 ()로 감싸주어 사용한다.</p>

{% highlight python %}
# 조건문
age = 20
if age >= 20:
  print('성인') # 조건이 참이면 성인 출력
else:
  print('청소년') # 조건이 거짓이면 청소년 출력

age = 65
if age > 80:
  print('아직 정정')

elif age > 60:
  print('인생은 60부터')
else:
  print('아직')

# 반복문
fruits = ['사괴', '배', '감', '귤']
for fruit in fruits: # fruit은 우리가 임의로 지어준 이름
  print(fruit) # 사과, 배, 감, 귤 출력

fruits = ['사과', '배', '배', '감', '수박', '귤', '딸기', '사과', '배', '수박']
count = 0
for fruit in fruits:
  if fruit == '사과':
    count = count + 1

print(count) # 사과의 갯수를 출력

# 함수
def sum(a, b):
  return a + b

print(sum(3, 5)) # 3 + 5 = 8

def print_name(name):
  print('반갑습니다. ' + name + '님')

print_name('원영준')
{% endhighlight %}

<h4>Pandas란?</h4>
<img src="/assets/img/data_analysis/one/1.png" title="Pandas" alt="오류뜨지마"/>
<p>파이썬에서 사용되는 데이터 분석 라이브러이이다. 관계형 데이터를 행과 열로 구성된 객체로 만들어 준다. 데이터를 쉽게 도와주는 도구이다. </p>

{% highlight python %}
import pandas as pd #  pandas 임포트
chicken07 = pd.read_csv('경로/chicken_07.csv') # 경로의 chicken_07.csv 파일 읽기
chicken07.tail(5) # 마지막 5개 보여주기
chicken07.describe() # 데이터의 기본 통계치
# count: 갯수
# mean: 평균
# std: 표준편차
# min: 최솟값
# max: 최댓값

# 성별 데이터 살펴보기
gender_range = set(chicken07['성별'])
print(gender_range, len(gender_range)) #{'남', '여'} 2 출력

# 연령대 데이터 살펴보기
age_range = set(chicken07['연령대'])
print(age_range, len(age_range)) # {'50대', '60대이상', '40대', '30대', '10대', '20대'} 6

# 치킨 데이터 합치기
chicken_07 = pd.read_csv('경로/chicken_07.csv')
chicken_08 = pd.read_csv('경로/chicken_08.csv')
chicken_09 = pd.read_csv('경로/chicken_09.csv')

# 3분기 데이터
chicken_data = pd.concat(chicken_07, chicken_08, chicken_09) # 07, 08, 09 데이터 합치기
chicken_data = chicken_data.reset_index(drop = True) # 인덱스 다시 생성
{% endhighlight %}

<h4>Matplotlib 이란?</h4>
<img src="/assets/img/data_analysis/one/2.png" title="Matplotlib" alt="오류뜨지마"/>
<p>파이썬에서 사용되는 시각화 라이브러리이다. 판다스가 관계형 데이터를 다루는데 사용된다면, Matplotlib은 그 데이터들을 시각화 하는데 사용한다.
{% highlight python %}
import pandas as pd #  pandas 임포트
import matplotlib.pyplot as plt # matplotlib의 pyplot 임포트

plt.rcParams['font_family'] = 'Malgun Gothic' # 폰트 맑은 고딕으로 변경

# 치킨 데이터 합치기
chicken_07 = pd.read_csv('경로/chicken_07.csv')
chicken_08 = pd.read_csv('경로/chicken_08.csv')
chicken_09 = pd.read_csv('경로/chicken_09.csv')

# 3분기 데이터
chicken_data = pd.concat(chicken_07, chicken_08, chicken_09) # 07, 08, 09 데이터 합치기
chicken_data = chicken_data.reset_index(drop = True) # 인덱스 다시 생성

# 방법 1
sum_of_calls_by_week = chicken_data.groupby('요일')['통화건수'].sum() # 요일 별로 그룹화하고 통화건수를 모두 더한 데이터

# 방법 2
groupdata = chicken_data.groupby('요알')
call_data = groupdata['통화건수']
sum_of_calls_by_week = call_data.sum()

plt.figure(figsize=(8,5)) # 그래프의 사이즈
plt.bar(sum_of_calls_by_week.index, sum_of_calls_by_week) # bar 그래프에 x축, y축 값을 넣어줍니다.
plt.title('요일에 따른 치킨 주문량 합계') # 그래프의 제목
plt.show() # 그래프 그리기
{% endhighlight %}
<img src="/assets/img/data_analysis/one/3.png" title="그래프" alt="오류뜨지마"/>

{% highlight python %}
import pandas as pd #  pandas 임포트
import matplotlib.pyplot as plt # matplotlib의 pyplot 임포트

plt.rcParams['font_family'] = 'Malgun Gothic' # 폰트 맑은 고딕으로 변경

# 치킨 데이터 합치기
chicken_07 = pd.read_csv('경로/chicken_07.csv')
chicken_08 = pd.read_csv('경로/chicken_08.csv')
chicken_09 = pd.read_csv('경로/chicken_09.csv')

# 3분기 데이터
chicken_data = pd.concat(chicken_07, chicken_08, chicken_09) # 07, 08, 09 데이터 합치기
chicken_data = chicken_data.reset_index(drop = True) # 인덱스 다시 생성

# 요일 별로 모아주기
groupdata = chicken_data.groupby('요일')
# '통화건수' 열만 떼어보기
call_data = groupdata['통화건수']
# 요일 별로 더해주기
sum_of_calls_by_week = call_data.sum()
sorted_sum_of_calls_by_week = sum_of_calls_by_week.sort_values(ascending=True)

plt.figure(figsize=(8,5)) # 그림의 사이즈
plt.bar(sorted_sum_of_calls_by_week.index, sorted_sum_of_calls_by_week) # 바 그래프
plt.title('요일에 따른 치킨 주문량 합계') # 그래프의 제목
plt.show() # 그래프 그리기
{% endhighlight %}
<img src="/assets/img/data_analysis/one/4.png" title="그래프" alt="오류뜨지마"/>

<h2>숙제</h2>
<p>피자를 가장 많이 시켜먹는 요일은 언제인가?</p>
{% highlight python %}
import pandas as pd #  pandas 임포트
import matplotlib.pyplot as plt # matplotlib의 pyplot 임포트

plt.rcParams['font_family'] = 'Malgun Gothic' # 폰트 맑은 고딕으로 변경

pizza_data = pd.read_csv('경로/pizza_09.csv')

# 3분기 데이터
pizza_data = pizza_data.reset_index(drop = True) # 인덱스 다시 생성

# 요일 별로 모아주기
groupdata = pizza_data.groupby('요일')
# '통화건수' 열만 떼어보기
call_data = groupdata['통화건수']
# 요일 별로 더해주기
sum_of_calls_by_week = call_data.sum()
sorted_sum_of_calls_by_week = sum_of_calls_by_week.sort_values(ascending=True)

plt.figure(figsize=(8,5)) # 그림의 사이즈
plt.bar(sorted_sum_of_calls_by_week.index, sorted_sum_of_calls_by_week) # 바 그래프
plt.title('요일에 따른 피자 주문량 합계') # 그래프의 제목
plt.show() # 그래프 그리기
{% endhighlight %}
<img src="/assets/img/data_analysis/one/5.png" title="그래프" alt="오류뜨지마"/>

<p>피자를 가장 많이 시켜먹는 구는 어디인가?</p>
{% highlight python %}
import pandas as pd #  pandas 임포트
import matplotlib.pyplot as plt # matplotlib의 pyplot 임포트

plt.rcParams['font_family'] = 'Malgun Gothic' # 폰트 맑은 고딕으로 변경

pizza_data = pd.read_csv('경로/pizza_09.csv')

# 3분기 데이터
pizza_data = pizza_data.reset_index(drop = True) # 인덱스 다시 생성

# 요일 별로 모아주기
groupdata = pizza_data.groupby('발신지_구')
# '통화건수' 열만 떼어보기
call_data = groupdata['통화건수']
# 요일 별로 더해주기
sum_of_calls_by_week = call_data.sum()
sorted_sum_of_calls_by_week = sum_of_calls_by_week.sort_values(ascending=True)

plt.figure(figsize=(8,5)) # 그림의 사이즈
plt.bar(sorted_sum_of_calls_by_week.index, sorted_sum_of_calls_by_week) # 바 그래프
plt.title('구에 따른 피자 주문량 합계') # 그래프의 제목
plt.show() # 그래프 그리기
{% endhighlight %}
<img src="/assets/img/data_analysis/one/6.png" title="그래프" alt="오류뜨지마"/>
<p>바 그래프 2개 그리기</p>
{% highlight python %}
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

chicken_data = pd.read_csv('./data/chicken_09.csv')
pizza_data = pd.read_csv('./data/pizza_09.csv')

chicken_week = chicken_data.groupby('요일')['통화건수'].sum()
pizza_week = pizza_data.groupby('요일')['통화건수'].sum()


chicken_week = chicken_week.sort_values(ascending=True)
pizza_week = pizza_week.sort_values(ascending=True)

plt.figure(figsize=(10, 5))
plt.bar(chicken_week.index, chicken_week)
plt.bar(pizza_week .index, pizza_week )
plt.title('요일에 따른 치킨과 피자 주문량 합계')
plt.show()
{% endhighlight %}
<img src="/assets/img/data_analysis/one/7.png" title="그래프" alt="오류뜨지마"/>
