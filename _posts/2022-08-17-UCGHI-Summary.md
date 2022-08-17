---
layout: post
title:  "UCGHI Summary Report 2019-2022"
author: Kelly Song
---

## Introduction

**Hello Everyone!**

Today we'll be going over the UCGHI Student Ambassador Summary Report for the 2019-2022 cohorts. 

## Import Dataframes 

We begin by importing the necessary csv files.

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


Get dataframe with campus coordinates


```python
# import campus coordinates and check
df2 = pd.read_csv("Campus coordinates - Sheet1.csv")
df2.head()
```

<div>
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

{% include study_chart.html %}

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

{% include campus_map.html %}

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

{% include dropdown.html %}

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

{% include COE.html %}

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

{% include slider_prop.html %}

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

{% include bar1_percamp.html %}

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

{% include bar2_percamp.html %}

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

{% include degrees.html %}

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
https://www.google.com


**Link to slides:**
https://docs.google.com/presentation/d/1nX_3GqWHz-3xfUui9WDKPX_qctJ5IM3yGVXae5Eg_a0/edit?usp=sharing
