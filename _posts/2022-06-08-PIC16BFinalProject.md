---
layout: post
title: "Generating Lyrics with Genius Lyrics"
authors: Kelly Song, Abhi Vemulapti and Chloe Florit
---

**Hello everyone!**

Today's post will be about demonstrating how to use our Lyric Generator, as well as our building process!

If you'd like to follow along, our code is available in our <a href=https://github.com/k-song14/lyric_generator><u>Github repository</u>!</a>

We will go through each major portion: web scraping, implementing our Markov model, and integrating web scraping and our model with Flask.

## Web Scraping

In order to be able to generate random lyrics, we need to start with a dataset of lyrics. We decided to scrape lyrics off of the Genius website 

