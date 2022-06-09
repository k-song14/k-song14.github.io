<!DOCTYPE html>
<html>
<head>
	<title>temp_app</title>
</head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
<link rel="stylesheet" href="{{ url_for('static', filename='mystyle.css') }}">

<!-- Navbar -->
<div>
	<h3>Lyric Generator</h3>
	<div class="w3-bar w3-black w3-card w3-large w3-center">
	  <a class="w3-bar-item w3-button w3-border-right w3-hide-small w3-padding-large w3-hover-white" style="width:25%">Home</a>
	  <a class="w3-bar-item w3-button w3-border-right w3-hide-small w3-padding-large w3-hover-white" style="width:25%" onClick="document.getElementById('about').scrollIntoView();">About</a>
	  <a class="w3-bar-item w3-button w3-border-right w3-hide-small w3-padding-large w3-hover-white" style="width:25%" onClick="document.getElementById('submit').scrollIntoView();">Submission form</a>
	  <a class="w3-bar-item w3-button w3-border-right w3-hide-small w3-padding-large w3-hover-white" style="width:25%" onClick="document.getElementById('creator_section').scrollIntoView();">Creators</a>
	</div>
</div>

<!-- Header -->
<header class="w3-container w3-white w3-center" style="padding:128px 100px">
	<h1 class="w3-margin w3-jumbo">Lyric Generator</h1>
	<h2>Create songs inspired by your favorite artists!</h2>
	<button class="w3-button w3-black w3-padding-large w3-large w3-margin-top" onClick="document.getElementById('submit').scrollIntoView();">Get Started</button>
  </header>
  
<!-- About section -->
<div class="w3-row-padding w3-padding-64 w3-container w3-black">
	<div class="w3-content" id="about">
		<h1 id="about_title">ABOUT</h1>
		<p class="yellow">LEARN MORE ABOUT THE LYRIC GENERATOR</p>
		<br>
		<h2><i>How does it work?</i></h2>
		<p>The user is prompted to input as many artist names as they would like into the <a onClick="document.getElementById('submit').scrollIntoView();"><u>lyric generator submission form</u></a>. 
		From there, our web app takes the artists names, converts them into urls for each artists' genius page and uses a Scrapy spider to scrape each artists'
		entire discography. This data is then reformatted and read into our Markov model to generate song lyrics in the style of the artists chosen! </p>
		<br>
		<h2><i>Where do you get your data from?</i></h2>
		<p>All of our data is lyrics scraped from the <a href="https://genius.com/"><u>Genius website</u>!</a></p>
		<br>
		<h2><i>How many artists can I choose?</i></h2>
		<p>You can input as many artists as you want! However, keep in mind that the more artists you choose, the longer it will take for the model to run. We reccomend selecting 1-3 artists at a time!</p>
		<br>
		<h2><i>Where can I access your code?</i></h2>
		<p>You can find all of our code in our <a href= https://github.com/k-song14/lyric_generator><u>Github respository</u></a>!</p>
	</div>
</div>

<!-- Submission Section -->
<div class="w3-row-padding w3-padding-64 w3-container" id="center_form">
	<div class="w3-content">
	  <div class="w3-twothird" id="submit">
		<br>
		<br>
		<br>
		<br>
		<br>
		<br>
		<br>
		<h1 id="lg_title">LYRIC GENERATOR</h1>
		<p id="pink">Use this submission form to input your artists!</p>

		<body>
			<div class="w3-container w3-center">
				<div class="w3-panel w3-card" style="height: 250px;">
					<h2>Enter artist names:</h2><br><br>
					<form id="input-form" method="post" action="/">
						<input type="text" name="inp" style="width: 300px;" placeholder="Artist Name"/><br><br>
						<input class="w3-hover-yellow" id="submitbutton" type="submit" name="submitbutton" value="Create Song"/>
					</form>
				</div>
			</div>
		</body>
  
	  </div>
	</div>
  </div>

<br>
<br>
<br>
<br>
<br>
<br>
<br>

<!-- Creator section -->
<div class="w3-row-padding w3-padding-64 w3-container w3-black" id="creator_section">
	<div class="w3-content">
	  <div class="w3-display-container">
		<h1 class="creator">CREATORS</h1>
		<p class="yellow">MEET THE CREATORS OF THE LYRIC GENERATOR</p>
		<br>

		<div class="w3-row-padding">
			<div class="w3-container w3-third w3-hover-opacity w3-hover-yellow">
				<a href="https://github.com/k-song14">
					
					<img style="width:100%;" src={{ url_for('static', filename='kelly_img.jpeg') }}>
				</a>
				<h4>Kelly Song</h4>
				<p class="center">  
					Class of 2023
					<br>
					Statistics Major
				</p>
			</div>
			<div class="w3-container w3-third w3-hover-opacity w3-hover-yellow">
				<a href="https://github.com/avvemul">
					<img style="width:100%;" src={{ url_for('static', filename='abhi_img.jpeg') }}>

				</a>
				<h4>Abhi Vemulapati</h4>	
				<p class="center">
		
					Class of 2022
					<br>
					Mathematics/Economics Major
		
				</p>
			</div>
			<div class="w3-container w3-third w3-hover-opacity w3-hover-yellow">
				<a href="https://github.com/chloeflorit">
					<img style="width:100%;" src={{ url_for('static', filename='chloe_img.jpeg') }}>

				</a>		
				<h4>Chloe Florit</h4>	
				<p class="center">
					Class of 2023
					<br>
					Mathematics/Economics Major
				</p>
			</div>
		</div>
	
	  </div>
	  
	</div>
  </div>

<!-- Footer -->
<footer class="w3-center w3-padding-48">
	<small>This page is &copy; 2022 Abhi Vemulapati, Chloe Florit, and Kelly Song</small> 
  </footer>

</html>