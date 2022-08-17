# UCGHI Summary Report 2019-2022 

## Import Dataframes 

We begin by importing 


```python
# import pandas for dataframe
import pandas as pd
```

Get dataframe with Ambassador Demographics


```python
# read in and check ambassador demographics csv
df = pd.read_csv("2019-2022 Ambassador Demographics - Sheet1.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Major(s) and/or Minor(s)</th>
      <th>Campus</th>
      <th>Degree</th>
      <th>COE</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-2022</td>
      <td>Sociology</td>
      <td>UCSB</td>
      <td>Graduate</td>
      <td>PH</td>
      <td>Alex Maldonado</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-2022</td>
      <td>Human Biology and Society / Global Health</td>
      <td>UCLA</td>
      <td>Undergraduate</td>
      <td>PH</td>
      <td>Alma Rincongallardo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-2022</td>
      <td>Global Studies</td>
      <td>UCSB</td>
      <td>Undergraduate</td>
      <td>PH</td>
      <td>Alyssa Mandujano</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-2022</td>
      <td>Urban and Regional Planning</td>
      <td>UCLA</td>
      <td>Graduate</td>
      <td>PH</td>
      <td>Amanda Caswell</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-2022</td>
      <td>Biology / Environmental Science</td>
      <td>UCR</td>
      <td>Undergraduate</td>
      <td>PH</td>
      <td>Andrew Tseng</td>
    </tr>
  </tbody>
</table>
</div>



Get dataframe with campus coordinates


```python
# import campus coordinates and check
df2 = pd.read_csv("Campus coordinates - Sheet1.csv")
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Campus</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>COUNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UCR</td>
      <td>33.9737</td>
      <td>117.3281</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>UCSD</td>
      <td>32.8801</td>
      <td>117.2340</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UCSB</td>
      <td>34.4140</td>
      <td>119.8489</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UCB</td>
      <td>37.8719</td>
      <td>122.2585</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UCLA</td>
      <td>34.0689</td>
      <td>118.4452</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>



We then merge the two dataframes into one in order to obtain the corresponding coordinates for each campus for each student.

This will make it simpler for us later on. We'll be using this merged dataframe for most of the code.


```python
# merge dataframes to obtain coordinates
df3 = df.merge(df2, on="Campus")
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Major(s) and/or Minor(s)</th>
      <th>Campus</th>
      <th>Degree</th>
      <th>COE</th>
      <th>Name</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>COUNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-2022</td>
      <td>Sociology</td>
      <td>UCSB</td>
      <td>Graduate</td>
      <td>PH</td>
      <td>Alex Maldonado</td>
      <td>34.414</td>
      <td>119.8489</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-2022</td>
      <td>Global Studies</td>
      <td>UCSB</td>
      <td>Undergraduate</td>
      <td>PH</td>
      <td>Alyssa Mandujano</td>
      <td>34.414</td>
      <td>119.8489</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-2022</td>
      <td>Biological Anthropology / Sociology</td>
      <td>UCSB</td>
      <td>Undergraduate</td>
      <td>PH</td>
      <td>Ashley Willis</td>
      <td>34.414</td>
      <td>119.8489</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-2022</td>
      <td>Chemistry</td>
      <td>UCSB</td>
      <td>Undergraduate</td>
      <td>PH</td>
      <td>Isabella Perez</td>
      <td>34.414</td>
      <td>119.8489</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-2022</td>
      <td>Psychology</td>
      <td>UCSB</td>
      <td>Undergraduate</td>
      <td>CGHJ</td>
      <td>Arianna Macias</td>
      <td>34.414</td>
      <td>119.8489</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
# import enrollment info for each campus

enroll = pd.read_csv('UC Enrollment.csv')
enroll.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Campus</th>
      <th>Enrollment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>UCB</td>
      <td>43185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019</td>
      <td>UCLA</td>
      <td>44371</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>UCM</td>
      <td>8847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019</td>
      <td>UCD</td>
      <td>38364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019</td>
      <td>UCSD</td>
      <td>38736</td>
    </tr>
  </tbody>
</table>
</div>



## Check Demographics Info

Next, we'll be looking through the demographics information to better understand and study each cohort / the ambassadors as a whole.

The questions we are currently interested in include:

**Areas of Study** - how many unique majors are there, and how many students fall into each category?

**Campus** - which campuses have the most ambassadors per year? which have the least?

**Returner status** - how many ambassadors return each year?

**Degree type** - how many students are undergraduate students vs. graduate students vs. doctoral students vs. other?

**COE** - how many students are in each COE each year?

### Areas of Study

For areas of study, we will consider both majors and minors / specializations of the student ambassadors. 

We begin by creating a list of all the majors and minors from the demographics dataframe.


```python
studies = list(df3["Major(s) and/or Minor(s)"])
len(studies)
```




    157



In order to create a data visualization that is readable, we'll group the subjects into 4 categories. The following are the categories, as well as some examples of majors that fall into them:

**Public Health / Global Health**: Community Health Sciences, Epidemiology, Global Health, Global Studies, etc.

**Computing / Mathematics / Engineering**: Engineering, Statistics, Bioinformatics, Computer Science

**Life / Physical Sciences**: Biology, Psychology, Neurobiology, Brain Sciences, Medicine, Biomedical Sciences, Nursing, Pharmacy, Geography, Chemistry, Urban and Regional Planning

**Social Sciences**: Anthropology, Sociology, Gender Studies, Language, Political Science, Policy, Law, International Development / Relations, Labor Studies, Social Welfare, Legal Studies, Economics

In order to create our data visualization / see how many ambassadors fall into each category, we'll create a dictionary with the categories as the keys. 

For each category, if certain key words exist in ane element in list of majors and minors, we'll add 1 to that category. For example, if the words "Public Health" or "Global" is in the element, we'll add 1 to "Public Health / Global Health."

***Note***: *if an ambassador has multiple majors and/or minors, they will be counted more than once. For example, if am ambassador is majoring in public health and minoring in bioinformatics, 1 will be added to both the Public Health/Global Health category and the Computing/Mathematics/Engineering category.*


```python
studies_dict = {"Public Health / Global Health": 0,
               "Computing / Mathematics / Engineering" : 0,
               "Life / Physical Sciences" : 0,
               "Social Sciences": 0}

for i in studies:
    if any(word in i for word in ["Public Health", "Global"]):
        studies_dict["Public Health / Global Health"] += 1
    if any(word in i for word in ["Engineering", "Computer", "Statistics", "Bioinformatics"]):
        studies_dict["Computing / Mathematics / Engineering"] += 1
    if any(word in i for word in ["Bio", "Psychology", "Brain Sciences", "Medicine", "Nursing", "Pharmacy", "Geo", "Chemistry", "Urban and Regional Planning", "Environment"]):
        studies_dict["Life / Physical Sciences"] += 1
    if any(word in i for word in ["Poli", "Law", "International", "Labor", "Social Welfare", "Legal", "Economics", "Anthropology", "Gender", "Sociology", "Language"]):
        studies_dict["Social Sciences"]  += 1
```

Now that we have our categories with the corresponding values/count, let's create a bar chart. 


```python
import plotly.express as px

fig = px.histogram(x=studies_dict.keys(), 
                   y=studies_dict.values(), 
                   title="Bar Chart of Main Areas of Study", 
                   color_discrete_sequence=['navy'])

fig.update_layout(xaxis_title="Area of Study")
fig.show()
```


<div>                            <div id="3a9f3229-4d64-498a-99d7-31d896509148" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("3a9f3229-4d64-498a-99d7-31d896509148")) {                    Plotly.newPlot(                        "3a9f3229-4d64-498a-99d7-31d896509148",                        [{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"x=%{x}<br>sum of y=%{y}<extra></extra>","legendgroup":"","marker":{"color":"navy","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"x":["Public Health / Global Health","Computing / Mathematics / Engineering","Life / Physical Sciences","Social Sciences"],"xaxis":"x","y":[76,4,75,29],"yaxis":"y","type":"histogram"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Area of Study"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"sum of y"}},"legend":{"tracegroupgap":0},"title":{"text":"Bar Chart of Main Areas of Study"},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('3a9f3229-4d64-498a-99d7-31d896509148');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


As we can see from the chart, our largest categories are Public/Global Health and Life/Physical Sciences. The smallest category is Computing/Mathematics/Engineering.

This seems logical, as the UCGHI Student Ambassador program focuses on Global Health issues, which tends to attract those interested in public/global health and the life sciences. 

We should, however, keep in mind that the different campuses have different majors; some campuses may have more students in life sciences because there are more options or they're more accessible.

Based on this chart, we can see that it may be beneficial to reach out to more departments in computing/math/engineering if we want a more interdisciplinary cohort of students.

### Campus / Center of Expertise

Next, we will be looking at the different campuses our student ambassadors come from, as well as the centers of expertise these students belong to. 

We are interested in the amount of students that come from each campus. We'll be looking at which campuses have produced the most ambassadors and which have produced the least, as well as how many ambassadors are planetary health track vs. the center of gender health and justice track.

Let's begin by visualizing this geographically. The size of each dot corresponds the amount of ambassadors.
Feel free to zoom in to look more closely at the map.


```python
import plotly.express as px
fig = px.scatter_geo(df3, lat='LATITUDE', 
                        lon=df3['LONGITUDE']*-1, 
                        size="COUNT",
                        hover_name="Campus",
                        color="Campus",
                        scope="usa",
                        center=dict(lat=35.3733, lon=-119.0187))

fig.update_layout(
        title_text = '2019-2022 Student Ambassadors per Campus',
    )

fig.show()
```


<div>                            <div id="bebd7562-ed42-4e5d-917b-267314e8a1b2" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("bebd7562-ed42-4e5d-917b-267314e8a1b2")) {                    Plotly.newPlot(                        "bebd7562-ed42-4e5d-917b-267314e8a1b2",                        [{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCSB<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB"],"lat":[34.414,34.414,34.414,34.414,34.414,34.414,34.414,34.414,34.414,34.414,34.414,34.414,34.414],"legendgroup":"UCSB","lon":[-119.8489,-119.8489,-119.8489,-119.8489,-119.8489,-119.8489,-119.8489,-119.8489,-119.8489,-119.8489,-119.8489,-119.8489,-119.8489],"marker":{"color":"#636efa","size":[13,13,13,13,13,13,13,13,13,13,13,13,13],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCSB","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCLA<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA"],"lat":[34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689,34.0689],"legendgroup":"UCLA","lon":[-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452,-118.4452],"marker":{"color":"#EF553B","size":[26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCLA","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCR<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCR","UCR","UCR","UCR","UCR","UCR"],"lat":[33.9737,33.9737,33.9737,33.9737,33.9737,33.9737],"legendgroup":"UCR","lon":[-117.3281,-117.3281,-117.3281,-117.3281,-117.3281,-117.3281],"marker":{"color":"#00cc96","size":[6,6,6,6,6,6],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCR","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCSC<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC"],"lat":[36.9821,36.9821,36.9821,36.9821,36.9821,36.9821,36.9821,36.9821,36.9821,36.9821,36.9821,36.9821,36.9821],"legendgroup":"UCSC","lon":[-122.0593,-122.0593,-122.0593,-122.0593,-122.0593,-122.0593,-122.0593,-122.0593,-122.0593,-122.0593,-122.0593,-122.0593,-122.0593],"marker":{"color":"#ab63fa","size":[13,13,13,13,13,13,13,13,13,13,13,13,13],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCSC","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCSD<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD"],"lat":[32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801,32.8801],"legendgroup":"UCSD","lon":[-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234,-117.234],"marker":{"color":"#FFA15A","size":[20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCSD","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCB<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB"],"lat":[37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719,37.8719],"legendgroup":"UCB","lon":[-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585,-122.2585],"marker":{"color":"#19d3f3","size":[20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCB","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCI<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI"],"lat":[33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405,33.6405],"legendgroup":"UCI","lon":[-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443,-117.8443],"marker":{"color":"#FF6692","size":[17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCI","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=Charles Drew<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["Charles Drew","Charles Drew","Charles Drew","Charles Drew","Charles Drew"],"lat":[33.9256,33.9256,33.9256,33.9256,33.9256],"legendgroup":"Charles Drew","lon":[-118.2425,-118.2425,-118.2425,-118.2425,-118.2425],"marker":{"color":"#B6E880","size":[5,5,5,5,5],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"Charles Drew","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCD<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD"],"lat":[38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382,38.5382],"legendgroup":"UCD","lon":[-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617,-121.7617],"marker":{"color":"#FF97FF","size":[18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCD","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCSF<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCSF","UCSF","UCSF","UCSF","UCSF","UCSF","UCSF"],"lat":[37.7632,37.7632,37.7632,37.7632,37.7632,37.7632,37.7632],"legendgroup":"UCSF","lon":[-122.4582,-122.4582,-122.4582,-122.4582,-122.4582,-122.4582,-122.4582],"marker":{"color":"#FECB52","size":[7,7,7,7,7,7,7],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCSF","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UCM<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UCM","UCM","UCM","UCM","UCM","UCM","UCM","UCM","UCM","UCM","UCM"],"lat":[37.3647,37.3647,37.3647,37.3647,37.3647,37.3647,37.3647,37.3647,37.3647,37.3647,37.3647],"legendgroup":"UCM","lon":[-120.4241,-120.4241,-120.4241,-120.4241,-120.4241,-120.4241,-120.4241,-120.4241,-120.4241,-120.4241,-120.4241],"marker":{"color":"#636efa","size":[11,11,11,11,11,11,11,11,11,11,11],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UCM","showlegend":true,"type":"scattergeo"},{"geo":"geo","hovertemplate":"<b>%{hovertext}</b><br><br>Campus=UC Hastings<br>COUNT=%{marker.size}<br>LATITUDE=%{lat}<br>lon=%{lon}<extra></extra>","hovertext":["UC Hastings"],"lat":[37.7812],"legendgroup":"UC Hastings","lon":[-122.4158],"marker":{"color":"#EF553B","size":[1],"sizemode":"area","sizeref":0.065,"symbol":"circle"},"mode":"markers","name":"UC Hastings","showlegend":true,"type":"scattergeo"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{"lat":35.3733,"lon":-119.0187},"scope":"usa"},"legend":{"title":{"text":"Campus"},"tracegroupgap":0,"itemsizing":"constant"},"margin":{"t":60},"title":{"text":"2019-2022 Student Ambassadors per Campus"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('bebd7562-ed42-4e5d-917b-267314e8a1b2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


We'll now be looking at the raw data / count of ambassadors, then we'll account for the student population on each campus.

***Note***: *we should keep in mind that these are students that got accepted into the ambassador program; there may have been more applicants / interested students from campuses that had less students accepted.*


```python
# Import necessary packages

import plotly.graph_objects as go
import numpy as np

# Initialize figure

fig = go.Figure()

# Add Traces

        
fig.add_trace(
    go.Histogram(x=np.array(df3['Campus'][df3["COE"] == "PH"]), name="PH", marker_color = 'skyblue'))
fig.add_trace(
    go.Histogram(x=np.array(df3['Campus'][df3["COE"] == "CGHJ"]), name="CGHJ", marker_color = 'navy'))

for i in df3['Year'].unique():
    for j in df3['COE'].unique():
        if j == "PH":
            fig.add_trace(go.Histogram(x=np.array(df3['Campus'][(df3["Year"] == i) & (df3["COE"] == j)]), name=j, marker_color = 'skyblue'))
        if j == "CGHJ":
            fig.add_trace(go.Histogram(x=np.array(df3['Campus'][(df3["Year"] == i) & (df3["COE"] == j)]), name=j, marker_color = 'navy'))



# Add dropdown
fig.update_layout(
    updatemenus=[
        dict(active=0,
            buttons=list([
                dict(
                    label="All",
                    method="update",
                    args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "All Student Ambassadors"}]),
                dict(
                    label="2021-2022",
                    method="update",
                    args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "2021-2022 Student Ambassadors"}]),
                dict(
                    label="2020-2021",
                    method="update",
                    args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "2020-2021 Student Ambassadors"}]
                ),
                dict(
                    label="2019-2020",
                    method="update",
                    args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "2019-2020 Student Ambassadors"}]
                )
             ])
        )
    ])

# Set title and barmode
fig.update_layout(title_text="Student Ambassadors per Campus per Year", barmode='stack')

fig.show()
```


<div>                            <div id="39102c3d-8030-409b-a986-723b316557e7" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("39102c3d-8030-409b-a986-723b316557e7")) {                    Plotly.newPlot(                        "39102c3d-8030-409b-a986-723b316557e7",                        [{"marker":{"color":"skyblue"},"name":"PH","x":["UCSB","UCSB","UCSB","UCSB","UCSB","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCR","UCR","UCR","UCR","UCR","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSC","UCSD","UCSD","UCSD","UCSD","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","Charles Drew","Charles Drew","Charles Drew","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCD","UCSF","UCM","UCM"],"type":"histogram"},{"marker":{"color":"navy"},"name":"CGHJ","x":["UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCSB","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCR","UCSC","UCSC","UCSC","UCSC","UCSC","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCB","UCI","UCI","UCI","UCI","UCI","UCI","UCI","UCI","Charles Drew","Charles Drew","UCD","UCD","UCD","UCD","UCD","UCD","UCSF","UCSF","UCSF","UCSF","UCSF","UCSF","UCM","UCM","UCM","UCM","UCM","UCM","UCM","UCM","UCM","UC Hastings"],"type":"histogram"},{"marker":{"color":"skyblue"},"name":"PH","x":["UCSB","UCSB","UCSB","UCSB","UCLA","UCLA","UCLA","UCLA","UCLA","UCR","UCR","UCSC","UCSC","UCSC","UCSD","UCSD","UCB","UCB","UCI","UCI","UCI","Charles Drew","UCD","UCD"],"type":"histogram"},{"marker":{"color":"navy"},"name":"CGHJ","x":["UCSB","UCSB","UCSB","UCSB","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCSC","UCSC","UCSD","UCSD","UCSD","UCSD","UCB","UCB","UCI","UCI","UCI","UCD","UCSF","UCSF","UCM","UCM"],"type":"histogram"},{"marker":{"color":"skyblue"},"name":"PH","x":["UCSB","UCLA","UCLA","UCLA","UCLA","UCR","UCR","UCSC","UCSC","UCSC","UCSC","UCB","UCB","UCB","UCI","UCI","UCI","UCI","UCI","Charles Drew","Charles Drew","UCD","UCD","UCD","UCD","UCD","UCSF","UCM","UCM"],"type":"histogram"},{"marker":{"color":"navy"},"name":"CGHJ","x":["UCSB","UCSB","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCLA","UCR","UCSC","UCSC","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCSD","UCB","UCB","UCB","UCB","UCB","UCB","UCI","UCI","UCI","UCI","Charles Drew","Charles Drew","UCD","UCD","UCD","UCD","UCSF","UCSF","UCM","UCM","UC Hastings"],"type":"histogram"},{"marker":{"color":"skyblue"},"name":"PH","x":["UCLA","UCLA","UCR","UCSC","UCSD","UCSD","UCB","UCB","UCB","UCI","UCD","UCD","UCD","UCD","UCD"],"type":"histogram"},{"marker":{"color":"navy"},"name":"CGHJ","x":["UCSB","UCSB","UCLA","UCSC","UCSD","UCSD","UCSD","UCSD","UCB","UCB","UCB","UCB","UCI","UCD","UCSF","UCSF","UCM","UCM","UCM","UCM","UCM"],"type":"histogram"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"updatemenus":[{"active":0,"buttons":[{"args":[{"visible":[true,true,false,false,false,false,false,false]},{"title":"All Student Ambassadors"}],"label":"All","method":"update"},{"args":[{"visible":[false,false,true,true,false,false,false,false]},{"title":"2021-2022 Student Ambassadors"}],"label":"2021-2022","method":"update"},{"args":[{"visible":[false,false,false,false,true,true,false,false]},{"title":"2020-2021 Student Ambassadors"}],"label":"2020-2021","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,true,true]},{"title":"2019-2020 Student Ambassadors"}],"label":"2019-2020","method":"update"}]}],"title":{"text":"Student Ambassadors per Campus per Year"},"barmode":"stack"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('39102c3d-8030-409b-a986-723b316557e7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


Click on the different options on the dropdown menu to see how many ambassadors come from each campus, as well as how many are planetary health vs. center for gender health and justice. To see the count, hover over each bar / bar stack in the figure.

The figure includes data from each of the 3 cohorts, as well as a combination of all of them.

Keep in mind that this figure **does not** account for the ratio of ambassadors to student population on each campus (we'll be looking at that soon). 

Before we move onto the ratio of ambassadors to student population on each campus, let's first take a look at the overall percentage of ambassadors for each center of expertise.


```python
fig = px.pie(df3, 
             names='COE', 
             title='Student Ambassador COE 2019-2022',
             color="COE",
             color_discrete_map = {"PH":'skyblue', "CGHJ": "navy"})
fig.show()
```


<div>                            <div id="0248ce0e-9982-4313-84a0-1bbac7b6efed" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("0248ce0e-9982-4313-84a0-1bbac7b6efed")) {                    Plotly.newPlot(                        "0248ce0e-9982-4313-84a0-1bbac7b6efed",                        [{"customdata":[["PH"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["CGHJ"],["PH"],["PH"],["PH"],["PH"],["CGHJ"],["PH"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["PH"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["PH"],["CGHJ"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["CGHJ"],["CGHJ"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["CGHJ"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["PH"],["PH"],["CGHJ"],["PH"],["PH"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["PH"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["PH"],["PH"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"],["CGHJ"]],"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"COE=%{customdata[0]}<extra></extra>","labels":["PH","PH","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","PH","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","PH","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","CGHJ","PH","PH","PH","PH","CGHJ","PH","PH","PH","PH","CGHJ","CGHJ","PH","PH","PH","PH","CGHJ","CGHJ","PH","CGHJ","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","CGHJ","CGHJ","PH","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","PH","CGHJ","CGHJ","CGHJ","PH","PH","PH","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","PH","CGHJ","PH","PH","PH","CGHJ","CGHJ","PH","PH","CGHJ","PH","PH","PH","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","PH","PH","PH","CGHJ","CGHJ","CGHJ","PH","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","PH","PH","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ","CGHJ"],"legendgroup":"","marker":{"colors":["skyblue","skyblue","skyblue","skyblue","navy","navy","navy","navy","skyblue","navy","navy","navy","navy","skyblue","skyblue","skyblue","skyblue","skyblue","navy","navy","navy","navy","navy","navy","navy","skyblue","skyblue","skyblue","skyblue","navy","navy","navy","navy","navy","navy","navy","skyblue","skyblue","navy","skyblue","skyblue","skyblue","skyblue","navy","skyblue","skyblue","skyblue","skyblue","navy","navy","skyblue","skyblue","skyblue","skyblue","navy","navy","skyblue","navy","skyblue","skyblue","navy","navy","navy","navy","navy","navy","navy","navy","navy","navy","navy","navy","skyblue","skyblue","navy","navy","navy","navy","skyblue","skyblue","navy","navy","skyblue","skyblue","skyblue","navy","navy","navy","navy","navy","navy","skyblue","skyblue","skyblue","navy","navy","navy","navy","skyblue","skyblue","skyblue","navy","navy","navy","skyblue","skyblue","skyblue","skyblue","skyblue","navy","navy","navy","navy","skyblue","navy","skyblue","skyblue","skyblue","navy","navy","skyblue","skyblue","navy","skyblue","skyblue","skyblue","skyblue","skyblue","navy","navy","navy","navy","skyblue","skyblue","skyblue","skyblue","skyblue","navy","navy","navy","skyblue","navy","navy","navy","navy","navy","navy","skyblue","skyblue","navy","navy","navy","navy","navy","navy","navy","navy"]},"name":"","showlegend":true,"type":"pie"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"legend":{"tracegroupgap":0},"title":{"text":"Student Ambassador COE 2019-2022"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('0248ce0e-9982-4313-84a0-1bbac7b6efed');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


More than half of the overall population of ambassadors are in the center for gender health and justice.

There could be multiple possible factors contributing this, such as the CGHJ being a more active center, more interest/applicants for this center and/or more students being accepted into this center, etc.







Moving on, for the next figure, we'll be accounting for the student population on each campus for each school year. We are just interested in the proportion of ambassadors in relation the how many students are on campus, we we will not be including the number of planetary health vs. center for gender health and justice ambassadors.

We will be using the enroll dataframe as well as the demographics dataframe to calculate the proportions.

***Note***: *the enroll dataframe has enrollment data from all the UCs except UC Hastings, and it does not have the enrollment data from Charles Drew. Therefore, we'll just be looking at these UCs.*


```python
enroll.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Campus</th>
      <th>Enrollment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-2020</td>
      <td>UCB</td>
      <td>43185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-2020</td>
      <td>UCLA</td>
      <td>44371</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-2020</td>
      <td>UCM</td>
      <td>8847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-2020</td>
      <td>UCD</td>
      <td>38364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-2020</td>
      <td>UCSD</td>
      <td>38736</td>
    </tr>
  </tbody>
</table>
</div>



Let's see which campuses have the largest/smallest overall student population.


```python
enroll.groupby(['Campus'])['Enrollment'].sum().sort_values(ascending=False)
```




    Campus
    UCLA    135076
    UCB     130548
    UCSD    120197
    UCD     117488
    UCI     109716
    UCR      78828
    UCSB     78617
    UCSC     58496
    UCM      26958
    UCSF      9546
    Name: Enrollment, dtype: int64



Moving on, let's create an empty dataframe where we can store the proportions. This will be used for our visualizations.


```python
prop = pd.DataFrame(columns = ['Year', 'Campus', 'Prop'], index = range(30))
prop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Campus</th>
      <th>Prop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We'll change the years in enroll to match those in df3.


```python
for k in range(len(enroll['Year'])):
    if enroll['Year'][k] == 2019:
        enroll['Year'][k] = '2019-2020'
    elif enroll['Year'][k] == 2020:
        enroll['Year'][k] = '2020-2021'
    elif enroll['Year'][k] == 2021:
        enroll['Year'][k] = '2021-2022'
        
enroll['Year'].unique()
```




    array(['2019-2020', '2020-2021', '2021-2022'], dtype=object)




```python
for i in df3['Year'].unique():
    for j in df3['Campus'].unique():
        for k in range(len(enroll)):
            if (enroll['Year'][k] == i) & (enroll['Campus'][k] == j):
                prop['Year'][k] = i
                prop['Campus'][k] = j
                prop['Prop'][k] = len(df3[(df3['Year'] == i) & (df3['Campus'] == j)]) / enroll['Enrollment'][k]
```


```python
prop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Campus</th>
      <th>Prop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-2020</td>
      <td>UCB</td>
      <td>0.000162</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-2020</td>
      <td>UCLA</td>
      <td>0.000068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-2020</td>
      <td>UCM</td>
      <td>0.000565</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-2020</td>
      <td>UCD</td>
      <td>0.000156</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-2020</td>
      <td>UCSD</td>
      <td>0.000155</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.histogram(prop, 
                   x="Campus", 
                   y='Prop',
                   animation_frame="Year",
                   title="Proportion of Student Ambassadors per Campus",
                   color_discrete_sequence=['skyblue'])
  
fig["layout"].pop("updatemenus")
fig.show()
```


<div>                            <div id="e466cf91-e23e-47f7-b8a9-9bcddf3c4daa" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e466cf91-e23e-47f7-b8a9-9bcddf3c4daa")) {                    Plotly.newPlot(                        "e466cf91-e23e-47f7-b8a9-9bcddf3c4daa",                        [{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Year=2019-2020<br>Campus=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"","marker":{"color":"skyblue","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"x":["UCB","UCLA","UCM","UCD","UCSD","UCSB","UCSC","UCI","UCSF","UCR"],"xaxis":"x","y":[0.00016209331943962024,6.761172838114986e-05,0.0005651633322030067,0.00015639662183296842,0.00015489467162329616,7.60051683514479e-05,0.00010259567046270647,5.418879375745096e-05,0.0006289308176100629,3.914353935882883e-05],"yaxis":"y","type":"histogram"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Campus"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"sum of Prop"}},"legend":{"tracegroupgap":0},"title":{"text":"Proportion of Student Ambassadors per Campus"},"barmode":"relative","sliders":[{"active":0,"currentvalue":{"prefix":"Year="},"len":0.9,"pad":{"b":10,"t":60},"steps":[{"args":[["2019-2020"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2019-2020","method":"animate"},{"args":[["2020-2021"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2020-2021","method":"animate"},{"args":[["2021-2022"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2021-2022","method":"animate"}],"x":0.1,"xanchor":"left","y":0,"yanchor":"top"}]},                        {"responsive": true}                    ).then(function(){
                            Plotly.addFrames('e466cf91-e23e-47f7-b8a9-9bcddf3c4daa', [{"data":[{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Year=2019-2020<br>Campus=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"","marker":{"color":"skyblue","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"x":["UCB","UCLA","UCM","UCD","UCSD","UCSB","UCSC","UCI","UCSF","UCR"],"xaxis":"x","y":[0.00016209331943962024,6.761172838114986e-05,0.0005651633322030067,0.00015639662183296842,0.00015489467162329616,7.60051683514479e-05,0.00010259567046270647,5.418879375745096e-05,0.0006289308176100629,3.914353935882883e-05],"yaxis":"y","type":"histogram"}],"name":"2019-2020"},{"data":[{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Year=2020-2021<br>Campus=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"","marker":{"color":"skyblue","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"x":["UCB","UCLA","UCM","UCD","UCSD","UCSB","UCSC","UCI","UCSF","UCR"],"xaxis":"x","y":[0.00021263023601956197,0.00024669761600394716,0.00044355732978487467,0.00023033219020320418,0.00020214271275520516,0.00011459566828373888,0.0003131360576170346,0.00024791339558714154,0.0009372071227741331,0.00011349020201255958],"yaxis":"y","type":"histogram"}],"name":"2020-2021"},{"data":[{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Year=2021-2022<br>Campus=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"","marker":{"color":"skyblue","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"x":["UCB","UCLA","UCM","UCD","UCSD","UCSB","UCSC","UCI","UCSF","UCR"],"xaxis":"x","y":[8.881783462119193e-05,0.00026021337496747333,0.00021994941163532388,7.49063670411985e-05,0.00014324937328399188,0.0003062318174858368,0.00025200342724661053,0.0001643610464319956,0.000631911532385466,7.449621931686966e-05],"yaxis":"y","type":"histogram"}],"name":"2021-2022"}]);
                        }).then(function(){

var gd = document.getElementById('e466cf91-e23e-47f7-b8a9-9bcddf3c4daa');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


Click through the slider to see the different proportions throughout the years.

We'll now be creating two visualizations that will allow us to compare the original data vs. the data that takes into account the student population on each campus.


For the original data, we'll create a dataframe with count of each campus per year.


```python
count = pd.DataFrame(columns = ['Year', 'Campus', 'Count'], index = range(30))

for i in df3['Year'].unique():
    for j in df3['Campus'].unique():
        for k in range(len(count)):
            if (enroll['Year'][k] == i) & (enroll['Campus'][k] == j):
                count['Year'][k] = i
                count['Campus'][k] = j
                count['Count'][k] = len(df3[(df3['Year'] == i) & (df3['Campus'] == j)]) 
                
count.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Campus</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-2020</td>
      <td>UCB</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-2020</td>
      <td>UCLA</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-2020</td>
      <td>UCM</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-2020</td>
      <td>UCD</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-2020</td>
      <td>UCSD</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.histogram(count,
             x='Year',
             y='Count',
             color='Campus',
             title='Student Ambassadors per Campus from 2019-2022',
             color_discrete_sequence=['navy', 'skyblue', 'blue', 'royalblue', 'deepskyblue', 'turquoise', 'cyan', 'darkturquoise', 'lightgreen', 'teal'])

fig.show()
```


<div>                            <div id="85c998aa-5996-470a-bd05-cba63834c7cf" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("85c998aa-5996-470a-bd05-cba63834c7cf")) {                    Plotly.newPlot(                        "85c998aa-5996-470a-bd05-cba63834c7cf",                        [{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCB<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCB","marker":{"color":"navy","pattern":{"shape":""}},"name":"UCB","offsetgroup":"UCB","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[5,9,4],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCLA<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCLA","marker":{"color":"skyblue","pattern":{"shape":""}},"name":"UCLA","offsetgroup":"UCLA","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[3,11,12],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCM<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCM","marker":{"color":"blue","pattern":{"shape":""}},"name":"UCM","offsetgroup":"UCM","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[3,4,2],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCD<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCD","marker":{"color":"royalblue","pattern":{"shape":""}},"name":"UCD","offsetgroup":"UCD","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[6,8,3],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCSD<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCSD","marker":{"color":"deepskyblue","pattern":{"shape":""}},"name":"UCSD","offsetgroup":"UCSD","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[4,8,6],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCSB<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCSB","marker":{"color":"turquoise","pattern":{"shape":""}},"name":"UCSB","offsetgroup":"UCSB","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[1,3,8],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCSC<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCSC","marker":{"color":"cyan","pattern":{"shape":""}},"name":"UCSC","offsetgroup":"UCSC","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[2,6,5],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCI<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCI","marker":{"color":"darkturquoise","pattern":{"shape":""}},"name":"UCI","offsetgroup":"UCI","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[1,9,6],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCSF<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCSF","marker":{"color":"lightgreen","pattern":{"shape":""}},"name":"UCSF","offsetgroup":"UCSF","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[1,3,2],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCR<br>Year=%{x}<br>sum of Count=%{y}<extra></extra>","legendgroup":"UCR","marker":{"color":"teal","pattern":{"shape":""}},"name":"UCR","offsetgroup":"UCR","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[1,3,2],"yaxis":"y","type":"histogram"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"sum of Count"}},"legend":{"title":{"text":"Campus"},"tracegroupgap":0},"title":{"text":"Student Ambassadors per Campus from 2019-2022"},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('85c998aa-5996-470a-bd05-cba63834c7cf');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


**Student ambassador demographics raw data:**

*2019-2020 cohort*: 
1. UC Davis
2. UC Berkeley 
3. UCSD

*2020-2021 cohort*: 
1. UCLA
2. UCI / UC Berkeley
3. UC Davis / UCSD

*2021-2022 cohort*: 
1. UCLA
2. UCSB
3. UCI / UCSD

*Overall*: UCLA


```python
fig = px.histogram(prop,
             x='Year',
             y='Prop',
             color='Campus',
             title='Proportion of Student Ambassadors per Campus from 2019-2022',
             color_discrete_sequence=['navy', 'skyblue', 'blue', 'royalblue', 'deepskyblue', 'turquoise', 'cyan', 'darkturquoise', 'lightgreen', 'teal'])

fig.show()
```


<div>                            <div id="4792e3f7-e86a-46da-a996-59942c4a1b70" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("4792e3f7-e86a-46da-a996-59942c4a1b70")) {                    Plotly.newPlot(                        "4792e3f7-e86a-46da-a996-59942c4a1b70",                        [{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCB<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCB","marker":{"color":"navy","pattern":{"shape":""}},"name":"UCB","offsetgroup":"UCB","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[0.00016209331943962024,0.00021263023601956197,8.881783462119193e-05],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCLA<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCLA","marker":{"color":"skyblue","pattern":{"shape":""}},"name":"UCLA","offsetgroup":"UCLA","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[6.761172838114986e-05,0.00024669761600394716,0.00026021337496747333],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCM<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCM","marker":{"color":"blue","pattern":{"shape":""}},"name":"UCM","offsetgroup":"UCM","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[0.0005651633322030067,0.00044355732978487467,0.00021994941163532388],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCD<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCD","marker":{"color":"royalblue","pattern":{"shape":""}},"name":"UCD","offsetgroup":"UCD","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[0.00015639662183296842,0.00023033219020320418,7.49063670411985e-05],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCSD<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCSD","marker":{"color":"deepskyblue","pattern":{"shape":""}},"name":"UCSD","offsetgroup":"UCSD","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[0.00015489467162329616,0.00020214271275520516,0.00014324937328399188],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCSB<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCSB","marker":{"color":"turquoise","pattern":{"shape":""}},"name":"UCSB","offsetgroup":"UCSB","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[7.60051683514479e-05,0.00011459566828373888,0.0003062318174858368],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCSC<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCSC","marker":{"color":"cyan","pattern":{"shape":""}},"name":"UCSC","offsetgroup":"UCSC","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[0.00010259567046270647,0.0003131360576170346,0.00025200342724661053],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCI<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCI","marker":{"color":"darkturquoise","pattern":{"shape":""}},"name":"UCI","offsetgroup":"UCI","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[5.418879375745096e-05,0.00024791339558714154,0.0001643610464319956],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCSF<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCSF","marker":{"color":"lightgreen","pattern":{"shape":""}},"name":"UCSF","offsetgroup":"UCSF","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[0.0006289308176100629,0.0009372071227741331,0.000631911532385466],"yaxis":"y","type":"histogram"},{"alignmentgroup":"True","bingroup":"x","histfunc":"sum","hovertemplate":"Campus=UCR<br>Year=%{x}<br>sum of Prop=%{y}<extra></extra>","legendgroup":"UCR","marker":{"color":"teal","pattern":{"shape":""}},"name":"UCR","offsetgroup":"UCR","orientation":"v","showlegend":true,"x":["2019-2020","2020-2021","2021-2022"],"xaxis":"x","y":[3.914353935882883e-05,0.00011349020201255958,7.449621931686966e-05],"yaxis":"y","type":"histogram"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"sum of Prop"}},"legend":{"title":{"text":"Campus"},"tracegroupgap":0},"title":{"text":"Proportion of Student Ambassadors per Campus from 2019-2022"},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('4792e3f7-e86a-46da-a996-59942c4a1b70');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


**Student ambassador demographics data when accounting for student population:**

*2019-2020 cohort*: 
1. UCSF
2. UC Merced
3. UC Berkeley 

*2020-2021 cohort*: 
1. UCSF
2. UC Merced
3. UCSC

*2021-2022 cohort*: 
1. UCSF
2. UCSB
3. UCLA

*Overall*: UCSF

From the figures above, we can see that UCLA has the most student ambassadors when we don't account for the population. There could be multiple reasons for this, such as the prescence of the Center for Gender Health and Justice, the fact that UCLA has the largest student population, etc.

When we do account for the student population, UCSF has the highest proportion of student ambassadors. However, we should note that UCSF is a graduate school and only has about 3,000-4,000 students enrolled per year. 

***Note:*** UCSF and UC Merced have the smallest overall student populations and UCLA and UC Berkeley have the largest.

### Degree Type

Now we'll be focusing on the different degrees our ambassadors are studying towards. 

First, we need to correct some mispellings in the dataframe. We'll do this then use the unique() function to make sure it worked.


```python
df3['Degree'][df3['Degree'] == "Undrgraduate"] = "Undergraduate"
df3['Degree'][df3['Degree'] == "Undergraduate "] = "Undergraduate"
df3['Degree'][df3['Degree'] == "Graduate "] = "Graduate"
df3['Degree'].unique()
```

    /Users/kellysong/opt/anaconda3/envs/PIC16B/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    /Users/kellysong/opt/anaconda3/envs/PIC16B/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    /Users/kellysong/opt/anaconda3/envs/PIC16B/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    





    array(['Graduate', 'Undergraduate', nan, 'MD', 'PhD', 'JD'], dtype=object)



Using the data, we'll create a pie chart to see what degrees most ambassadors are studying for.


```python
fig = px.pie(df3, names='Degree', title='Student Ambassador Degrees 2019-2022', color_discrete_sequence=["navy", "skyblue", "darkturquoise", "teal", "royalblue", "deepskyblue"])
fig.show()
```


<div>                            <div id="0534857a-e141-4949-8863-6a4d76aa2299" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("0534857a-e141-4949-8863-6a4d76aa2299")) {                    Plotly.newPlot(                        "0534857a-e141-4949-8863-6a4d76aa2299",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"Degree=%{label}<extra></extra>","labels":["Graduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate",null,"Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Graduate","Graduate","Undergraduate","Undergraduate","Undergraduate","Graduate","Graduate","Undergraduate","Undergraduate","Graduate","MD","Graduate","Undergraduate","Undergraduate","Graduate","Undergraduate","Graduate","Undergraduate","Graduate","Undergraduate","Undergraduate","Graduate","Graduate","Undergraduate","Graduate","Undergraduate","Undergraduate","PhD","PhD","Graduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Graduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","PhD","Graduate","Undergraduate","Undergraduate","PhD","Graduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate",null,"Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","PhD","Undergraduate",null,"Undergraduate",null,"Undergraduate","Undergraduate","Undergraduate","Graduate","Graduate","Undergraduate","PhD","PhD","Graduate","Graduate","Graduate","Undergraduate","PhD","PhD","Undergraduate","PhD","Undergraduate","Undergraduate","Graduate",null,"Graduate","Undergraduate","Graduate","Graduate","Graduate","Graduate","Graduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Graduate","Undergraduate","Undergraduate","Graduate","Undergraduate","Undergraduate","PhD","PhD","PhD","Undergraduate","Undergraduate","PhD","Graduate","PhD","Graduate","MD","Graduate","Graduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","Undergraduate","JD"],"legendgroup":"","name":"","showlegend":true,"type":"pie"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"legend":{"tracegroupgap":0},"title":{"text":"Student Ambassador Degrees 2019-2022"},"piecolorway":["navy","skyblue","darkturquoise","teal","royalblue","deepskyblue"]},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('0534857a-e141-4949-8863-6a4d76aa2299');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


As we can see from the chart, student ambassadors the past three years have been overwhelmingly undergraduate students. Not even all the other degrees combined can surpass, or even match, the amount of undergraduates. The next largest degree is graduate, followed by PhD, MD and then JD.

***Note***: the 3.18% null are the students who did not have their degree type filled out in the dataframe. Since this data was manually filled out from the UCGHI website, not all the information was available.*

The chart suggests that the program either appeals more or is advertised more to undergraduate students. In the future, it may be beneficial to target more students working towards different degrees (especially M.D. and J.D.) for a more diverse cohort.

### Returners

Now, let's take a look at how many students returned to the program throughout the years.

***Note***: *this only accounts for those who were in the 2019-2022 programs and returned the following year(s). Since 2019-2020 was the first cohort, there were no returners that year.*


```python
returners = {}

for i in range(len(df3["Name "])):
    for j in df3["Name "]: 
        returners[j] = 0
    #once we have the character in dictionary, add up occurences
    for j in df3["Name "]:
        returners[j] += 1
```


```python
for key, val in returners.items():
    if val > 1:
        print(key, val)
```

    Kelly Song 2
    Claire Amabile 2
    Sean Sugai 2
    Shirelle Mizrahi 2
    Vandana Teki 3
    Donna Pham 2
    Colette Kirkpatrick 2
    Eniola Owoyele 2
    Geremy Lowe 2
    Natasha Glendening 2
    Kalani Phillips 3
    Catthi Ly 2
    Sydney Adams 2



```python
returner_list = []

for key, val in returners.items():
    if val > 1:
        returner_list.append(key)
        
        
len(returner_list)
```




    13



There are about 13 returning members from the previous 3 years, with 3 ambassadors being present for all 3 years.

### Survey Results

For our last section, we will just be looking at the amount of ambassadors per cohort, as well as the proportion of ambassadors that responded to the post program survey.


***Note***: *Only survey data from the 2020-2021 and 2021-2022 cohorts were available.*


```python
len(df3[df3["Year"] == "2019-2020"]), len(df3[df3["Year"] == "2020-2021"]), len(df3[df3["Year"] == "2021-2022"])
```




    (36, 70, 51)



2019-2020: 36 Ambassadors 

2020-2021: 70 Ambassadors

2021-2022: 51 Ambassadors


```python
19/70, 16/51
```




    (0.2714285714285714, 0.3137254901960784)



2020-2021: 27% of ambassadors participated in post-program survey

2021-2022: 31% of ambassadors participated in post-program survey

## References

***Student demographic data:***
https://ucghi.universityofcalifornia.edu/get-involved/ucghi-student-ambassador-program

***Student enrollment data:***
https://www.universityofcalifornia.edu/about-us/information-center/fall-enrollment-glance 

***Campus Coordinates:***
https://www.google.com/


```python

```
