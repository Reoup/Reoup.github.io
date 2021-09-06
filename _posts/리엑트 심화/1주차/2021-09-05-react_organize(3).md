---
layout: post
title: "Javascript (3)- 함수"
author: Reo
date: '2021-09-06'
category: React
excerpt: "스파르타 코딩클럽 리엑트 1주차"
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 1주차, 변수와 상수, var, let, const, TDZ, 자료형]
permalink: /blog/react_organize(1)/
usemathjax: true
---
> 자바스크립트는 함수를 특별한 값으로 취급한다.
> 자바스크립트는 ()가 있으면 함수를 실행하고 없으면 함수를 문자형으로 바꿔 출력하기도 한다. (함수를 값으로 취급함)
> 이걸 응용하면, 함수를 복사할 수 있고, 또 매개변수처럼 전달할 수 있다.
> - 함수는 return으로 어떤 값을 넘겨주지 않으면 기본적으로 undefined를 반환한다. 

<h2>함수 선언문과 함수 표현식</h2>
<b>함수 선언문</b>
{% highlight javascript %}
// 이렇게 생긴 게 함수 선언문 방식으로 함수를 만든 것
function cat() {
  console.log('perl');
}
{% endhighlight %}

<b>함수 표현식</b>
{% highlight javascript %}
// 이렇게 생긴 게 함수 표현식을 사용해 함수를 만든 것
let cat = function() {
  console.log('perl');
}

// 물론 화살표 함수로 써도 된다.
// 다만 주의해야 하는 점은 화살표 함수는 함수 표현식의 단축형이라는 것이다.
let cat2 = () => {
  console.log('perl2');
}
{% endhighlight %}

<p>함수 선언문 vs 함수 표현식</p>
<ul>
  <li>함수 선언문으로 함수를 생성하면 독립된 구문으로 존재한다.</li>
  <li>함수 표현식으로 함수를 생성하면 함수가 표현식의 일부로 존재한다.</li>
  <li>함수 선언문은 코드 블록이 실행되기 전에 미리 처리되어 블록 내 어디서든 사용할 수 있다.</li>
  <li>함수 표현식은 실행 컨텍스트가 표현식에 닿으면 만들어진다. (변수처럼 처리된다.)</li>
</ul>

<hr/>

<h2>지역 변수와 외부 변수</h2>
<b>지역 변수</b>
<ul>
  <li>함수 내에서 선언한 변수이다.</li>
  <li>함수 내에서만 접근이 가능하다.</li>
</ul>
<b>외부 변수(global 변수라고도 함)</b>
<ul>
  <li>함수 외부에서 선언한 변수이다.</li>
  <li>함수 내에서도 접근이 가능하다.</li>
  <li>함수 내부에 같은 이름을 가진 지역 변수가 있으면 사용할 수 없다.</li>
</ul>
{% highlight javascript %}
let a = 'a';
let b = 'b';
let c = 'outter';
const abc = () => {
  let b = 'inner!';
  c = 'c';
  let d = 'd';
  console.log(a, b, c, d);
}

console.log(a, b, c, d); // a, b, outter, undefined

abc(); // a, inner, c, d

console.log(a ,b, c, d); // a, b, c, undefined 
{% endhighlight %}

<hr/>

<h2>콜백 함수</h2>
> 함수를 값처럼 전달할 때, 인수로 넘겨주는 함수를 콜백 함수라고 한다.
{% highlight javascript %}
const playWithCat = (cat, action) => {
  action(cat);
}

const useBall = (cat) => {
  alert(cat + "과 공으로 놀아줍니다.");
}

//playWithCat 함수에 넘겨주는 useBall 함수가 콜백 함수이다.
playWithCat("perl", useBall);
{% endhighlight %}