---
layout: post
title: "Javascript (2)- 객체"
author: Reo
date: '2021-09-05'
category: React
excerpt: "스파르타 코딩클럽 리엑트 1주차"
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 1주차, 변수와 상수, var, let, const, TDZ, 자료형]
permalink: /blog/react_organize(1)/
usemathjax: true
---

<h2>객체</h2>

<ul>
  <li>오직 한 타입의 데이터만 담을 수 있는 원시형과 달리, 다양한 데이터를 담을 수 있다.</li>
  <li>key로 구분된 데이터 집합, 복잡한 개체를 저장할 수 있다.</li>
  <li>{...} ← 중괄호 안에 여러 쌍의 프로퍼티를 넣을 수 있다.</li>
  → 프로퍼티는 key: value로 구성되어 있다.
  → key에는 문자형, value에는 모든 자료 형이 들어갈 수 있다.
{% highlight javascript %}
// 객체 생성자로 만들기
let cat = new Object();

// 객체 리터럴로 만들기
// 중괄호로 객체를 선언하는 걸 리터럴이라고 하는데, 객체 선언할 때 주로 사용
let cat = {};
{% endhighlight %}
</ul>

<hr/>

<h2>상수는 재할당 X</h2>
<p>하지만 const로 선언된 객체는 수정될 수 있다.</p>
<ul>
  <li>const로 선언된 객체는 객체에 대한 참조를 변경하지 못한다는 것을 의미한다.</li>
  <li>즉, 객체에 프로퍼티는 보호되지 않는다.(중요!)</li>
{% highlight javascript %}
// my_cat이라는 상수를 만든다.
const my_cat = {
 name: "perl",
 status: "좀 언짢음"
}

my_cat.name = "펄이";
console.log(my_cat); // 고양이 이름이 바뀜

my_cat = {name: "perl2", status: "많이 언짢음"};
// 여기에서는 에러가 난다. 프로퍼티는 변경이 되지만,
// 객체 자체를 재할당 할 순 없기 때문이다.
{% endhighlight %}
</ul>