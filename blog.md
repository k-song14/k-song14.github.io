---
layout: page
title: Blog Posts
permalink: /blog/
---

<!DOCTYPE html>
<html>
<head>
<style>
article    {color: lavender;}
</style>
</head>
<body>
  
<article>
  
  <b>Hello!</b>

  <p>On this page you can see some projects I've worked on in the past. Most of my projects are created using Python, but I'll be adding some projects in R and HTML/CSS soon as well!</p>

  <p>I hope you find these posts helpful and enjoy reading them!</p>

</article>

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <h1><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>

      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}
</div>
