---
layout: post
title: "Javascript (1)- 기본"
author: Reo
date: '2021-09-05'
category: React
excerpt: "스파르타 코딩클럽 리엑트 1주차"
thumbnail: /assets/img/posts/code.jpg
tags: [스파르타 1주차, 변수와 상수, var, let, const, TDZ, 자료형]
permalink: /blog/react_organize(1)/
usemathjax: true
---

<h2>변수와 상수</h2>
<h3>변수 생성의 3단계</h3>
<p>선언 → 초기화 → 할당</p>
<ul>
  <li>선언: 실행 컨텍스트에 변수 객체를 등록한다. (스코프가 참조하는 대상이 되어야 하기 떄문이다.)</li>
  <li>초기화: 변수 객체에 등록된 변수를 위해 메모리에 공간을 확보한다. (여기에서 변수는 보통 undefined 로 초기화 됨)</li>
  <li>할당: undefined로 초기화된 변수에 실제 값을 할당한다.</li>
</ul>

<h4>var (가급적 사용을 자제한다.)</h4>
<ul>
  <li>블록 스코프가 아니라 함수 수준 스코프를 가진다.</li>
  <li>선언과 초기화를 한번에 한다.</li>
  <li>재선언이 가능하다.</li>
  <li>선언하기 전에도 사용이 가능하다.</li>
  <p> * <b>호이스팅</b>: 함수 안에 있는 선언들을 모두 끌어 올려서 해당 함수의 유효 범위의 최상단에 선언하는 것</p>
{% highlight javascript %}
// var는 이런 식의 사용도 가능
// var name은 선언! name = "perl"을 할당
function cat() {
 name = "perl";
 aler(name);
 var name;
}

cat();
{% endhighlight %}
 <li>코드 블럭을 무시한다. (var는 함수의 최상위로 호이스팅이 된다. 선언은 호이스팅이 되고 할당은 호이스팅이 되지 않는다.)</li>
{% highlight javascript %}
// var name은 함수의 최상위로 호이스팅되기 때문에,
// 실행될 일 없는 구문 속에 있어도 선언이 된다. 
// (자바스크립트가 동작하기 전에 코드를 한 번 다 보고 그 때,
// var로 선언된 코드를 전부 최상위로 끌어올려버린다.)

function cat() {
 name = "perl";
 if(false) {
  var name;
 }
 alert(name);
}

cat();
}
{% endhighlight %}
</ul>

<h4>let</h4>
<ul>
  <li>자바스크립트에서 변수를 생성할 때 쓰는 키워드이다.</li>
  <li>block-scope를 갖는다.</li>
  → {} 안에서 선언하면 {} 안에서만 쓰고 바깥에선 사용할 수 없다.
  <li>재선언이 불가능하지만, 재할당은 가능하다.</li>
{% highlight javascript %}
// 재할당은 가능
let cat_name = "perl";
cat name = "펄이";

// 재선언은 에러
let cat_name = "perl";
let cat_name = "펄이";
{% endhighlight %}
</ul>

<h4>const</h4>
<ul>
  <li>자바스크립트에서 상수를 생성할 때 쓰는 키워드이다.</li>
  <li>block-scope를 갖는다.</li>
  → {} 안에서 선언하면 {} 안에서만 쓰고 바깥에선 사용할 수 없다.
  <li>재선언, 재할당 둘 다 불가능하다. (⇒ 선언과 동시에 할당이 된다.)</li>
{% highlight javascript %}
// 재할당 에러
const cat_name = "perl";
cat_name = "펄이";

// 재선언도 에러
const cat_name = "perl";
const cat_name = "펄이";

// 선언과 동시에 할당이 되기 때문에 값을 안줘도 오류가 난다.
// declare
const cat_name;
{% endhighlight %}
</ul>

<h4>TDZ (Temporal Dead Zone) = 일시적 사각지대</h4>
<p>(var과 let, const의 차이점 중 하나는 변수가 선언이 되기 전에 호출하면 ReferenceError가 난다.)</p>
<p>(호이스팅(=선언 끌어 올리기)은 되나 선언 후, 초기화 단계에서 메모리에 공간을 확보하는데, 선언을 호이스팅해도 초기화 전까지 메모리에 공간이 없다.<br>그래서 변수를 참조할 수 없고 이것을 TDZ라고 부른다.)</p>
<ul>
  <li>let, const 선언도 호이스팅이 된다.</li>
  <li>스코프에 진입할 때 변수를 만들고, TDZ가 생성이 되지만 코드 실행이 (=실행 컨텍스트가) 변수가 있는 실제 위치에 도달할 때까지 접근을 못했을 뿐이다.</li>
  * 면접에 자주 나오므로 알아두자!
</ul>
<hr/>

<h2>자료형</h2>
<ul>
  <li>자바스크립트는 8가지 기본 자료형이 있다.</li>
  <li>객체를 제외한 7가지를 원시형(primitive type)라고 부른다.</li>
</ul>
<ol>
  <li>숫자형: 정수, 부동 소수점을 저장</li>
  <li>BigInt형: 아주 큰 숫자를 저장</li>
  <li>문자형: 문자열을 저장</li>
  <li>boolean형: 논리 값(true/false)</li>
  <li>undefiend: 값이 할당되지 않음을 나타내는 독립 자료형</li>
  <li>null: 값이 존재하지 않음을 나타내는 독립 자료형</li>
  <li>객체형: 복잡한 자료구조를 저장하는 데 쓰임</li>
  <li>심볼형: 고유 식별자를 만들 때 쓰임</li>
</ol>