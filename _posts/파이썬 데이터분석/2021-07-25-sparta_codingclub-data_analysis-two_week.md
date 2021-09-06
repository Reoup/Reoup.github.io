---
layout: post
title:  "스파르타 코딩클럽 파이썬 데이터분석 2주차"
author: Reo
date: '2021-07-25'
category: Data-Analysis
excerpt: "스파르타 코딩클럽 파이썬 데이터분석 2주차"
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 2주차, 데이터 분석의 4단계, 바 그래프, 라인 그래프, 파이 차트, 히트맵, 지도, Pandas, Matplotlib, Folium, Json]
permalink: /blog/data_analysis-two_week/
usemathjax: true
---

<h4>데이터 분석의 4단계</h4>
<p>데이터 불러오기 → 데이터 살펴보기 → 데이터 가공하기 → 데이터 시각화</p>
<h4>어울리는 그래프 찾기</h4>
<ul>
  <li>바 그래프</li>
    <p>각 항목들의 수치와 순위를 볼 때 좋음</p>
    <img src="/assets/img/data_analysis/two/1.png" title="바 그래프" alt="오류뜨지마"/>
  <li>라인 그래프</li>
    <p>이전 항목들 혹은 흐름에 따라 데이터의 관계를 볼 떄 좋음</p>
    <img src="/assets/img/data_analysis/two/2.png" title="라인 그래프" alt="오류뜨지마"/>
  <li>파이 차트</li>
    <p>비율을 볼 때 좋음</p>
    <img src="/assets/img/data_analysis/two/3.png" title="파이 차트" alt="오류뜨지마"/>
  <li>히트맵</li>
    <p>두 개의 축의 수치를 한 눈에 보기 좋음</p>
    <img src="/assets/img/data_analysis/two/4.png" title="히트맵" alt="오류뜨지마"/>
  <li>지도</li>
    <p>지리 정보를 한 눈에 보기 좋음</p>
    <img src="/assets/img/data_analysis/two/5.png" title="지도" alt="오류뜨지마"/>
</ul>

<h2>[상권 데이터] 불러오기</h2>
{% highlight python %}
import pandas as pd # pandas 모듈 임포트해서 pd 이름으로 지정
import matplotlib.pyplot as plt # matplotlib의 pyplot 모듈 임포트해서 plt 이름으로 지정

commercial = pd.read_csv('경로/commercial.csv') # commercial.csv 파일 읽기

commercial.tail(5) # 맨 뒤 5개 출력

list(commercial), len(list(commercial)) # 컬럼, 컬럼의 갯수 확인

commercial.groupby('상가업소번호')['상권업종소분류명'].count().sort_values(ascending=False) # 상가업소번호 가게 중복 확인

category = set(commercial['상권업종소분류명']) # 가게를 분류하기 위한 이름
category, len(category) # 가게, 가게의 갯수

commercial[['시', '구', '상세주소']] = commercial['도로명'].str.split(' ', n=2, expand=True) # 도로명을 잘라 정리
commercial.tail(5) # 맨 뒤 5개 출력

seoul_data = commercial[commercial['시'] == '서울특별시'] # 서울특별시 데이터 저장
seoul_data # 서울특별시 데이터 출력

city_type = set(seoul_data['시']) # 서울특별시만 남았는지 확인

seoul_chicken_data = seoul_data[seoul_data['상권업종소분류명'] == '후라이드/양념치킨'] # 후라이드/양념치킨 데이터만 저장
seoul_chicken_data # 후라이드/양념치킨 데이터 출력

set(seoul_chicken_data['상권업종소분류명']) # 후라이드/양념치킨만 남았는지 확인

group_data = seoul_chicken_data.groupby('구') # 구로 그룹화
group_by_category = group_data['상권업종소분류명'] # 구의 상권업종소분류명(후라이드/양념치킨) 저장
chicken_count_gu = group_by_category.count() # 구의 치킨 가게 갯수 저장
sorted_chicken_count_gu = chicken_count_gu.sort_values(ascending=False) # 내림차순
sorted_chicken_count_gu # 출력
{% endhighlight %}
<img src="/assets/img/data_analysis/two/6.png" title="구의 치킨 갯수" alt="오류뜨지마"/>
<p>이어서</p>
{% highlight python %}
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
# plt.rcParams['font.family'] = 'AppleGothic' # Apple

plt.figure(figsize=(10, 5)) # 그래프의 사이즈 (10, 5)
plt.bar(sorted_chicken_count_gu.index, sorted_chicken_count_gu) 
plt.title('구에 따른 치킨가게 수의 합계') 
plt.xticks(rotation= 45)
plt.show()
{% endhighlight %}
<img src="/assets/img/data_analysis/two/7.png" title="구에 따른 치킨가게 수의 합계" alt="오류뜨지마"/>
<p>이어서</p>
{% highlight python %}
import folium # folium 모듈 임포트
import json # json 모듈 임포트

# https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json
seoul_state_geo = '경로/seoul_geo.json' #  seoul_geo.json 데이터 가져오기
geo_data = json.load(open(seoul_state_geo, encoding='utf-8')) # json 파일 열기, 인코딩 방식은 utf-8

map = folium.Map(location=[37.5502, 126.982], zoom_start=11) # 서울 위치로 zoom을 11로 설정
folium.Choropleth(geo_data=geo_data, # 지리 정보
                 data=chicken_count_gu, # 데이터
                 columns=[chicken_count_gu.index, chicken_count_gu], # 데이터의 인덱스와 데이터의 갯수
                 fill_color='PuRd', # 색은 퍼플레드
                 key_on='properties.name').add_to(map) # 핵심 키를 맵에 추가
map # 지도 출력
{% endhighlight %}
<img src="/assets/img/data_analysis/two/8.png" title="지도" alt="오류뜨지마"/>

<h2>[유동인구 데이터]</h2>
{% highlight python %}
import pandas as pd
import matplotlib.pyplot as plt

population = pd.read_csv('./data/population07.csv')
list(population), len(list(population)) # (['일자', '시간(1시간단위)', '연령대(10세단위)', '성별', '시', '군구', '유동인구수'], 7)
set(population['연령대(10세단위)']), len(set(population['연령대(10세단위)'])) # ({20, 30, 40, 50, 60, 70}, 6)
set(population['시']) # {'서울'}
set(population['군구']), len(set(population['군구'])) # 군구의 이름, 군구의 갯수

sum_of_population_by_gu = population.groupby('군구')['유동인구수'].sum() # 군구의 유동인구수 합

plt.rcParams['font.family'] = 'Malgun Gothic' # 맑은 고딕

sorted_sum_of_population_by_gu = sum_of_population_by_gu.sort_values(ascending=True) # 오름차순

plt.figure(figsize=(10, 5)) # 그래프 사이즈
plt.bar(sorted_sum_of_population_by_gu.index, sorted_sum_of_population_by_gu) # 군구, 군구의 유동인구수 합
plt.title('2020년 7월 서울 군구별 유동인구 수') 
plt.xlabel('군구')
plt.ylabel('유동인구 수(명)')
plt.xticks(rotation = -45)
plt.show()

population_gangnam = population[population['군구'] == '강남구'] # 강남구
set(population_gangnam['군구']) # 강남구만 있는지 확인

population_gangnam_daily = population_gangnam.groupby('일자')['유동인구수'].sum() # 강남구의 일자별 유동인구수 합
{% endhighlight %}
<p>이어서</p>
{% highlight python %}
plt.figure(figsize=(10, 5)) # 그래프 사이즈

date = [] # 비어있는 데이터
for day in population_gangnam_daily.index: # 강남의 일자별
    date.append(str(day)) # date에 문자열(str) 추가

plt.plot(date, population_gangnam_daily) # 일자별, 일자별 유동인구수 합
plt.title('2020년 7월 서울 강남구 날짜별 유동인구 수') 
plt.xlabel('날짜')
plt.ylabel('유동인구 수(천만명)')
plt.xticks(rotation=-90)
plt.show()
{% endhighlight %}
<img src="/assets/img/data_analysis/two/9.png" title="2020년 7월 서울 강남구 날짜별 유동인구 수" alt="오류뜨지마"/>
<p>이어서</p>
{% highlight python %}
import folium
import json

map = folium.Map(location=[37.5502, 126.982], zoom_start=11, tiles='stamentoner')

seoul_state_geo = './data/seoul_geo.json'
geo_data =json.load(open(seoul_state_geo, encoding='utf-8'))
folium.Choropleth(geo_data=geo_data,
                 data=sum_of_population_by_gu,
                 columns=[sum_of_population_by_gu.index, sum_of_population_by_gu],
                 fill_color='PuRd',
                 key_on='properties.name').add_to(map)
map
{% endhighlight %}
<img src="/assets/img/data_analysis/two/10.png" title="지도" alt="오류뜨지마"/>
<a href="https://python-visualization.github.io/folium/modules.html#module-folium.map">folium 참고 링크</a>

<h2>상권과 유동인구 동시 분석</h2>
{% highlight python %}
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'

population = pd.read_csv('./data/population07.csv')
commercial = pd.read_csv('./data/commercial.csv')

commercial[['시','구','상세주소']] = commercial['도로명'].str.split(' ', n=2, expand=True) 
seoul_data = commercial[commercial['시'] == '서울특별시']
seoul_chicken_data = seoul_data[seoul_data['상권업종소분류명'] == '후라이드/양념치킨']
chicken_count_gu = seoul_chicken_data.groupby('구')['상권업종소분류명'].count()
new_chicken_count_gu = pd.DataFrame(chicken_count_gu).reset_index() # index 초기화하고 다시 index 설정

sum_of_population_by_gu = population.groupby('군구')['유동인구수'].sum()
new_sum_of_population_by_gu = pd.DataFrame(sum_of_population_by_gu).reset_index()

gu_chicken = new_chicken_count_gu.join(new_sum_of_population_by_gu.set_index('군구'), on = '구')

gu_chicken['유동인구수/치킨집수'] = gu_chicken['유동인구수'] / gu_chicken['상권업종소분류명']
gu_chicken = gu_chicken.sort_values(by='유동인구수/치킨집수')

plt.figure(figsize=(10,5))
plt.bar(gu_chicken['구'], gu_chicken['유동인구수/치킨집수'])
plt.title('치킨집 당 유동인구수')
plt.xlabel('구')
plt.ylabel('유동인구수/치킨집수')
plt.xticks(rotation=90)
plt.show()
{% endhighlight %}
<img src="/assets/img/data_analysis/two/11.png" title="상권과 유동인구 같이 분석" alt="오류뜨지마"/>

<h2>숙제</h2>
<h4>Q1. 4월의 유동인구가 가장 많은 구는 어디인가?</h4>
{% highlight python %}
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'

population = pd.read_csv('./data/population04.csv')

sorted_population_daily = population.groupby('군구')['유동인구수'].sum().sort_values(ascending=True)

plt.figure(figsize=(10, 5))
plt.bar(sorted_population_daily.index, sorted_population_daily)
plt.xlabel('군구')
plt.xticks(rotation=-45)
plt.ylabel('유동인구수(명)')
plt.title('2020년 4월 서울 군구별 유동인구 수')
plt.show()
{% endhighlight %}
<img src="/assets/img/data_analysis/two/12.png" title="2020년 4월 서울 군구별 유동인구 수" alt="오류뜨지마"/>

<h4>Q2. 4월과 7월의 강남구의 일별 유동인구는 어떤가?</h4>
{% highlight python %}
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'

population04 = pd.read_csv('./data/population04.csv')
population07 = pd.read_csv('./data/population07.csv')

population04_gangnam = population04[population04['군구'] == '강남구'].groupby('일자')['유동인구수'].sum()

population07_gangnam = population07[population07['군구'] == '강남구'].groupby('일자')['유동인구수'].sum()

# 방법 1
date04_list = []
for day in population04_gangnam_data.index:
    date04_list.append(str(day))
date07_list = []
for day in population07_gangnam_data.index:
    date07_list.append(str(day))
    
# 방법 2
# date04_list = [str(day) for day in population04_gangnam.index]
# date07_list = [str(day) for day in population07_gangnam.index]
    
plt.figure(figsize=(20, 5))
plt.plot(date04_list, population04_gangnam_data)
plt.plot(date07_list, population07_gangnam_data)
plt.xlabel('일자')
plt.xticks(rotation=90)
plt.ylabel('유동인구수(명)')
plt.title('2020년 4월과 7월의 서울 강남구 날짜별 유동인구 수')
plt.show()
{% endhighlight %}
<img src="/assets/img/data_analysis/two/13.png" title="2020년 4월과 7월의 서울 강남구 날짜별 유동인구 수" alt="오류뜨지마"/>

<h2>더 알아보기</h2>
<h4>Pandas</h4>
<a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html">Pandas 참고 링크</a>

