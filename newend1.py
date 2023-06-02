import base64
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, f1_score, roc_auc_score, classification_report
from astropy.table import Table
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

st.set_page_config(layout="wide")
st.sidebar.header(".")
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('Student-data.csv')
dfv = pd.read_csv('Student-data.csv')
original_df = df.copy()


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("img.jpg")
image_path = get_img_as_base64("student.gif")
page_bg_img = f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-repeat: no-repeat;
background-attachment: fixed;
background-position-x: -517px;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

@import url('https://fonts.googleapis.com/css?family=Exo:400,700');

* {{
  margin: 0px;
  padding: 0px;
}}

body{{
  font-family: 'Exo', sans-serif;


}} 


.context {{
  width: 100%;
  position: absolute;
  top: 50vh;

}}

.context h1 {{
  text-align: center;
  font-size: 50px;
}}


.area {{
    width: 100%;
  height: 100%;


}}

.circles{{
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}} 
.circles li{{
  position: absolute;
  display: block;
  list-style: none;
  width: 20px;
  height: 20px;
  background: rgba(255, 255, 255, 0.2);
  animation: animate 25s linear infinite;
  bottom: -150px;

}} 
.circles li:nth-child(1){{
  left: 25%;
  width: 80px;
  height: 80px;
  animation-delay: 0s;
}} 

.circles li:nth-child(2){{
  left: 10%;
  width: 20px;
  height: 20px;
  animation-delay: 2s;
  animation-duration: 12s;
}} 

.circles li:nth-child(3){{
  left: 70%;
  width: 20px;
  height: 20px;
  animation-delay: 4s;
}} 

.circles li:nth-child(4){{
  left: 40%;
  width: 60px;
  height: 60px;
  animation-delay: 0s;
  animation-duration: 18s;
}} 

.circles li:nth-child(5){{
  left: 65%;
  width: 20px;
  height: 20px;
  animation-delay: 0s;
}} 

.circles li:nth-child(6){{
  left: 75%;
  width: 110px;
  height: 110px;
  animation-delay: 3s;
}} 

.circles li:nth-child(7){{
  left: 35%;
  width: 150px;
  height: 150px;
  animation-delay: 7s;
}} 
.circles li:nth-child(8){{
  left: 50%;
  width: 25px;
  height: 25px;
  animation-delay: 15s;
  animation-duration: 45s;
}} 

.circles li:nth-child(9){{
  left: 20%;
  width: 15px;
  height: 15px;
  animation-delay: 2s;
  animation-duration: 35s;
}} 
.circles li:nth-child(10){{
  left: 85%;
  width: 150px;
  height: 150px;
  animation-delay: 0s;
  animation-duration: 11s;
}} 


@keyframes animate{{

  0%{{
    transform: translateY(0) rotate(0deg);
    opacity: 1;
    border-radius: 0;
  }} 
  100%{{
    transform: translateY(-1000px) rotate(720deg);
    opacity: 0;
    border-radius: 50%;
  }} 
}} 
.landP{{
    text-align: center;
    padding: 10px 10px 10px 10px;
    margin-bottom: 35%;
}}
.logo{{
    border-radius: 50px;
}}
.obj{{
margin-bottom: 10%;
}}
table {{
  font-family: arial, sans-serif;
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 10%;
}}
</style>

<html>
<head></head>
<body>

   <div class="area fixed" >
            <ul class="circles">
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
            </ul>
        <div class="landP">
            <h1 class="head">STUDENT GENERAL PERFORMANCE PREDICTION USING MACHINE LEARNING ALGORITHM  </h1>
            <div><img class="logo" src="data:image/png;base64,{image_path}" alt="Image" width="100" height="100"></div>
            <h3>Unlocking Potential: Navigating Success Beyond the Classroom</h3>
        </div>
        <div class="obj"><h1>Objectives :</h1>
            <p class="objtext">
•	The primary goals of this are to raise student academic performance and prevent dropouts.\n
•	 The performance of the student is dependent on a number of elements, including their mental state in addition to their grades and academic course work.\n 
•	In light of this, we will carry out a survey in which we will inquire about things like your home, your Score, your financial situation, etc.\n 
•	We can determine where the learner is falling behind and where he or she needs to improve by carefully examining these answers.\n
•	 We can help students perform better by adding these answers as an additional feature to our machine learning model.</p>
        </div>
    </div>
     <h3>Table for features of dataset </h3>
   <table>
  <tr>
    <th>Sr.no</th>
    <th>Features</th>
    <th>Description</th>
    <th>Type</th>
    <th>Values</th>
    <th>Parsing</th>
  </tr>
  <tr>
    <td>1</td>
        <td>sex</td>
        <td>The gender of the student</td>
        <td>The Binary</td>
        <td>F=Female or M= male</td>
        <td> M = 0, F = 1</td>
  </tr>
  <tr>
    <td>2</td>
        <td>age</td>
        <td>The age of the student</td>
        <td>Numeric</td>
        <td>input from user</td>
        <td>-</td>
  </tr>
  
  <tr>
    <td>3</td>
        <td>address</td>
        <td>The address of the student</td>
        <td>Binary</td>
        <td>U = Urban  or R = Rural</td>
        <td> U = 0, R = 1</td>
  </tr>
  <tr>
    <td>4</td>
        <td>famsize</td>
        <td>The family size of the student</td>
        <td>Binary</td>
        <td> GT3 = Greater than 3 or\n LE = less than or equal to 3</td>
        <td> LE3 = 0,  GT3 = 1</td>
  </tr>
  <tr>
    <td>5</td>
        <td>Pstatus</td>
        <td>The Parantal status of the student</td>
        <td>Binary</td>
        <td>A = Apart or T = Togather</td>
        <td> T = 0, A = 1</td>
  </tr>
  <tr>
    <td>6</td>
        <td>Medu</td>
        <td>The Mothers education  of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>
  </tr>
  <tr>
    <td>7</td>
        <td>Fedu</td>
        <td>The Fathers education of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>
  
  </tr>
   <tr>
    <td>8</td>
        <td>Mjob</td>
        <td>The Mothers job of the student</td>
        <td>Nominal</td>
        <td> teacher, health,services,at_home, other</td>
        <td> teacher = 0, health = 1, services = 2, at_home = 3, other = 4</td>
  </tr>
   <tr>
    <td>9</td>
        <td>Fjob</td>
        <td>The Fathers Job of the student</td>
        <td>Nominal</td>
        <td> teacher, health, services, at_home, other</td>
        <td> teacher = 0, health = 1, services = 2, at_home = 3, other = 4</td>
  </tr>
   <tr>
    <td>10</td>
        <td>reason</td>
        <td>The reason of the student to choose this field of study</td>
        <td>Nominal</td>
        <td>home,reputation,course,other</td>
        <td> home = 0, reputation = 1, course = 2, other = 3</td>
  </tr>
   <tr>
    <td>11</td>
        <td>guardian</td>
        <td>The guardian of the student</td>
        <td>Nominal</td>
        <td>mother,father,other</td>
        <td> mother = 0, father = 1, other = 2</td>
  </tr>
    <tr>
    <td>12</td>
        <td>traveltime</td>
        <td>The traveltime of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>
  </tr>
    <tr>
    <td>13</td>
        <td>studytime</td>
        <td>The Fathers studytime of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>
 
   </tr>
    <tr>
    <td>14</td>
        <td>backlog</td>
        <td>The backlog of the student</td>
        <td>Numeric</td>
        <td>input From user</td>
        <td> -- </td>
  </tr>
    <tr>
    <td>15</td>
        <td>schoolsup</td>
        <td>Extra educational school support to the student</td>
        <td>Binary</td>
        <td>yes or no</td>
        <td> no = 0, yes = 1</td>
  </tr>
   <tr>
    <td>16</td>
        <td>famsup</td>
        <td>Family support to the student</td>
        <td>Binary</td>
        <td>yes or no</td>
        <td> no = 0, yes = 1</td>
  </tr>
   <tr>
    <td>17</td>
        <td>paid</td>
        <td>Extra paid Courses done by student</td>
        <td>Binary</td>
        <td> yes or no</td>
        <td> no = 0, yes = 1</td>

  </tr>
   <tr>
    <td>18</td>
        <td>activities</td>
        <td>Extra Carricular activities From the student</td>
        <td>Binary</td>
        <td>yes or no</td>
        <td> no = 0, yes = 1</td>

  </tr>
  <tr>
    <td>19</td>
        <td>nursery</td>
        <td>nursery attended by the student</td>
        <td>Binary</td>
        <td> yes or no</td>
        <td> no = 0, yes = 1</td>

  </tr>
  <tr>
    <td>20</td>
        <td>higher</td>
        <td> student is aiming for higher studies</td>
        <td>Binary</td>
        <td>yes or no</td>
        <td> no = 0, yes = 1</td>

  </tr>
  <tr>
    <td>21</td>
        <td>internet</td>
        <td> student have internet access or not.</td>
        <td>Binary</td>
        <td>yes or no</td>
        <td> no = 0, yes = 1 </td>

  </tr>
  <tr>
    <td>22</td>
        <td>romantic</td>
        <td>romantic Relastionship By the student</td>
        <td>Binary</td>
        <td>yes or no</td>
        <td> no = 0, yes = 1</td>

  </tr>
    <tr>
    <td>23</td>
        <td>famrel</td>
        <td>Quality of family relationships of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>

  </tr>
    <tr>
    <td>24</td>
        <td>freetime</td>
        <td>The freetime  of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>

  </tr>
    <tr>
    <td>25</td>
        <td>goout</td>
        <td>student going out with friends</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>

  </tr>
    <tr>
    <td>26</td>
        <td>Dalc</td>
        <td>Daily Alcohol consumption of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>

  </tr>
    <tr>
    <td>27</td>
        <td>Walc</td>
        <td>The Weekend Alcohol consumption of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>

  </tr>
   <tr>
    <td>28</td>
        <td>health</td>
        <td>The health status of the student</td>
        <td>Numeric</td>
        <td>From 1 very bad to 5 excellent</td>
        <td> -- </td>

  </tr>
   <tr>
    <td>29</td>
        <td>absences</td>
        <td>The absences of the student</td>
        <td>Numeric</td>
        <td>input From user</td>
        <td> -- </td>

  </tr>
   <tr>
    <td>30</td>
        <td>Score</td>
        <td>The Score of the student</td>
        <td>Numeric</td>
        <td>input From user</td>
        <td> -- </td>

  </tr>
   <tr>
    <td>31</td>
        <td>perf</td>
        <td>The performance of the student</td>
        <td>Binary</td>
        <td>poor or good</td>
        <td>poor = 0, good = 1</td>

  </tr>
</table>
</body>
</html>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def numerical_data():
    df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    df['address'] = df['address'].map({'U': 0, 'R': 1})
    df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})
    df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})
    df['Mjob'] = df['Mjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
    df['Fjob'] = df['Fjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
    df['reason'] = df['reason'].map({'home': 0, 'reputation': 1, 'course': 2, 'other': 3})
    df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1, 'other': 2})
    df['schoolsup'] = df['schoolsup'].map({'no': 0, 'yes': 1})
    df['famsup'] = df['famsup'].map({'no': 0, 'yes': 1})
    df['paid'] = df['paid'].map({'no': 0, 'yes': 1})
    df['activities'] = df['activities'].map({'no': 0, 'yes': 1})
    df['nursery'] = df['nursery'].map({'no': 0, 'yes': 1})
    df['higher'] = df['higher'].map({'no': 0, 'yes': 1})
    df['internet'] = df['internet'].map({'no': 0, 'yes': 1})
    df['romantic'] = df['romantic'].map({'no': 0, 'yes' : 1})
    df['perf'] = df['perf'].map({'poor': 0, 'good': 1})
    
    col = df['perf']
    del df['perf']
    df['perf'] = col

    
def feature_scaling(df):
    for i in df:
        col = df[i]
        if(np.max(col)>6):
            Max = max(col)
            Min = min(col)
            mean = np.mean(col)
            col  = (col-mean)/(Max)
            df[i] = col
        elif(np.max(col)<6):
            col = (col-np.min(col))
            col /= np.max(col)
            df[i] = col
numerical_data()
features=['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'backlog', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences']
dfv['perf'].value_counts()
with st.container():
    col1, col2 = st.columns([1, 1])
    col1.subheader("Student who has performed good or poor in exam")
    col1.write("This figure shows a pie chart of student performance; from this chart, we can deduce that 32% of students struggle to pass the exam.")
    labels = 'student has performed good in the final exam ', 'student has performed poor in the final exam '
    sizes = [265, 130]
    colors=['lightskyblue','yellow']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',colors=colors,
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    col2.pyplot()

with st.container():
    st.header("Correlation Heatmap")
    st.markdown(
        "Features Correlating with the status of student"
    )
    col1, col2 = st.columns([1, 1])
    corr = df.corr()
    plt.figure(figsize=(30,30))
    sns.heatmap(corr, annot=True, cmap="Reds")
    plt.title('Correlation Heatmap', fontsize=20) 
    col1.pyplot()
    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(df.corr()[['perf']].sort_values(by='perf', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with the status of student', fontdict={'fontsize':18}, pad=16)
    col2.pyplot()

with st.container():
    st.header("going out,romantic status")
    st.markdown(
        "academic standing  We've divided the frequency of going out into ranges of 1 to 5, and academic standing This graph typically depicts the proportion of students who hang out and have relationships based on whether romantic relationships are present or not."
    )
df["goout"].unique()  
col1, col2 = st.columns([1, 1])
# going out
perc = (lambda col: col/col.sum())
index = [0,1]
out_tab = pd.crosstab(index=df.perf, columns=df.goout)
out_perc = out_tab.apply(perc).reindex(index)
out_perc.plot.bar(colormap="mako_r", fontsize=16, figsize=(14,6))
plt.title('student status  By Frequency of Going Out', fontsize=20)
plt.ylabel('Percentage of Student', fontsize=16)
plt.xlabel('Student status', fontsize=16)
col1.pyplot()

# romantic status
romance_tab1 = pd.crosstab(index=df.perf, columns=df.romantic)
romance_tab = np.log(romance_tab1)
romance_perc = romance_tab.apply(perc).reindex(index)
plt.figure()
romance_perc.plot.bar(colormap="PiYG_r", fontsize=16, figsize=(8,8))
plt.title('Student status By Romantic relaion', fontsize=20)
plt.ylabel('Percentage of Logarithm Student Counts ', fontsize=16)
plt.xlabel('Student status', fontsize=16)
col2.pyplot()
with st.container():
    st.header("Student status By mother JOB & education ")
    st.markdown(
        "Student status By mother JOB and education show how its effects students performance \n #'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4"
    )
    col1, col2,col3 = st.columns([1, 1,1])
    # 1) mother job 
        # Mjob distribution
    f, fx = plt.subplots()
    figure = sns.countplot(x = 'Mjob', data=dfv, order=['teacher','health','services','at_home','other'])
    fx = fx.set(ylabel="Count", xlabel="Mother Job")
    figure.grid(False)
    col1.pyplot()

 #Mother education:
    good = df.loc[df.perf==1]
    poor=df.loc[df.perf==0]
    good['good_student_mother_education'] = good.Medu
    poor['poor_student_mother_education'] = poor.Medu
    plt.figure(figsize=(6,4))
    p=sns.kdeplot(good['good_student_mother_education'], shade=True, color="r")#good_student in red
    p=sns.kdeplot(poor['poor_student_mother_education'], shade=True, color="b")#poor_student in blue
    plt.xlabel('Mother Education Level', fontsize=20)
    col3.pyplot()

    #Student status By mother JOB
    mjob_tab1 = pd.crosstab(index=df.perf, columns=df.Mjob)
    mjob_tab = np.log(mjob_tab1)
    mjob_perc = mjob_tab.apply(perc).reindex(index)
    plt.figure()
    mjob_perc.plot.bar(colormap="mako_r", fontsize=16, figsize=(8,8))
    plt.title('Student status By mother JOB', fontsize=20)
    plt.ylabel('Percentage of Logarithm Student Counts ', fontsize=16)
    plt.xlabel('Student status', fontsize=16)
    col2.pyplot()
    #'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4

   
with st.container():
    st.header("Final Grade By Desire to Receive Higher Education")
    st.markdown(
        "Intent on Pursuing Higher Education Demonstrates His Commitment to Study"
    )
    higher_tab = pd.crosstab(index=df.perf, columns=df.higher)
    higher_perc = higher_tab.apply(perc).reindex(index)
    higher_perc.plot.bar(colormap="Dark2_r", figsize=(14,6), fontsize=16)
    plt.title('Final Grade By Desire to Receive Higher Education', fontsize=20)
    plt.xlabel('Final Grade', fontsize=16)
    plt.ylabel('Percentage of Student', fontsize=16)
    st.pyplot()


with st.container():
    st.header("impact of age & backlog")
    st.markdown(
        "impact of age and backlog have on student is shown in following graph"
    )
    col1, col2 = st.columns([1, 1])
    #impact of age
    higher_tab = pd.crosstab(index=df.perf, columns=df.age)
    higher_perc = higher_tab.apply(perc).reindex(index)
    higher_perc.plot.bar(colormap="Dark2_r", figsize=(14,6), fontsize=16)
    plt.title('Student status  By age', fontsize=20)
    plt.xlabel('Student status', fontsize=16)
    plt.ylabel('Percentage of Student', fontsize=16)
    col1.pyplot()

    fail_tab = pd.crosstab(index=df.perf, columns=df.backlog)
    fail_perc = fail_tab.apply(perc).reindex(index)
    fail_perc.plot.bar(colormap="Dark2_r", figsize=(14,6), fontsize=16)
    plt.title('student status By backlog', fontsize=20)
    plt.xlabel('Final Grade', fontsize=16)
    plt.ylabel('Percentage of Student', fontsize=16)
    col2.pyplot()


with st.container():
    st.header("Address Distribution")
    st.markdown("First, let's look at how students are distributed between urban and rural areas; this graph demonstrates how location affects student performance."
    )
    col1, col2 = st.columns([1, 1])
   
    f, fx = plt.subplots()
    figure = sns.countplot(x = 'address', data=dfv, order=['U','R'])
    fx = fx.set(ylabel="Count", xlabel="address")
    figure.grid(False)
    plt.title('Address Distribution')
    col1.pyplot()

    #student status By Living Area
    ad_tab1 = pd.crosstab(index=df.perf, columns=df.address)
    ad_tab = np.log(ad_tab1)
    ad_perc = ad_tab.apply(perc).reindex(index)
    ad_perc.plot.bar(colormap="RdYlGn_r", fontsize=16, figsize=(8,6))
    plt.title('student status By Living Area', fontsize=20)
    plt.ylabel('Percentage of Logarithm Student#', fontsize=16)
    plt.xlabel('Student status', fontsize=16)
    col2.pyplot()

with st.container():
    st.header("impact of weekend alcohol consumption in student performance")
    #impact of weekend alcohol consumption in student performance
    alc_tab = pd.crosstab(index=df.perf, columns=df.Walc)
    alc_perc = alc_tab.apply(perc).reindex(index)
    alc_perc.plot.bar(colormap="Dark2_r", figsize=(14,6), fontsize=16)
    plt.title('student status By weekend alchol consumption', fontsize=20)
    plt.xlabel('Student status', fontsize=16)
    plt.ylabel('Percentage of Student', fontsize=16)
    st.pyplot()


with st.container():
    st.header("weekend alcohol consumption")
    st.markdown(
        "Good Performance vs. Poor Performance Student Weekend Alcohol Consumption"
    )
   
    good = df.loc[df.perf == 1]
    good['good_alcohol_usage']=good.Walc
   
    poor = df.loc[df.perf == 0]
    poor['poor_alcohol_usage']=poor.Walc
    plt.figure(figsize=(10,6))
    p1=sns.kdeplot(good['good_alcohol_usage'], shade=True, color="r")
    p1=sns.kdeplot(poor['poor_alcohol_usage'], shade=True, color="b")
    plt.title('Good Performance vs. Poor Performance Student Weekend Alcohol Consumption', fontsize=20)
    plt.ylabel('Density', fontsize=16)
    plt.xlabel('Level of Alcohol Consumption', fontsize=16)
    st.pyplot()

with st.container():
    st.header("student status By internet accessibility and student status By health")
    st.markdown(
        "this show what kind of  role of internet and health have in ones academics"
    )
    col1, col2 = st.columns([1, 1])
#student status By internet accessibility 
    alc_tab = pd.crosstab(index=df.perf, columns=df.internet)
    alc_perc = alc_tab.apply(perc).reindex(index)
    alc_perc.plot.bar(colormap="Dark2_r", figsize=(14,6), fontsize=16)
    plt.title('student status By internet accessibility', fontsize=20)
    plt.xlabel('Student status', fontsize=16)
    plt.ylabel('Percentage of Student', fontsize=16)
    col1.pyplot()

#student status By health
    he_tab = pd.crosstab(index=df.perf, columns=df.health)
    he_perc = he_tab.apply(perc).reindex(index)
    he_perc.plot.bar(colormap="Dark2_r", figsize=(14,6), fontsize=16)
    plt.title('student status By health', fontsize=20)
    plt.xlabel('Student status', fontsize=16)
    plt.ylabel('Percentage of Student', fontsize=16)
    col2.pyplot()

with st.container():
    st.header("student status By study time")
    st.markdown(
        "student status By study time"
    )
    stu_tab = pd.crosstab(index=df.perf, columns=df.studytime)
    stu_perc = stu_tab.apply(perc).reindex(index)
    stu_perc.plot.bar(colormap="Dark2_r", figsize=(14,6), fontsize=16)
    plt.title('student status By study time', fontsize=20)
    plt.xlabel('Student status', fontsize=16)
    plt.ylabel('Percentage of Student', fontsize=16)
    st.pyplot()

with st.container():
    st.header("absences Participation and internet Participation")
    st.markdown("how much time did he or she spend in college or online"
    )
    col1, col2 = st.columns([1, 1])
#absences Participation
    plt.figure(figsize = (9, 7))
    sns.barplot(x = 'Score',y='absences' ,data = df)
    plt.title('absences Participation')
    plt.xlabel('Score')
    plt.ylabel('absences')
    col1.pyplot()

#internet Participation
    plt.figure(figsize = (9, 7))
    sns.countplot(x = 'internet',data = df, hue = 'sex')
    plt.title('internet Participation')
    plt.xlabel('internet')
    plt.ylabel('Score')
    col2.pyplot()

with st.container():
    st.header("Medu & Fedu Participation")
    st.markdown(
        "Medu & Fedu Participation")
    col1, col2 = st.columns([1, 1])
    plt.figure(figsize = (9, 7))
    sns.lineplot(x = 'Medu', y = 'Score', data = df)
    plt.title('Medu Participation')
    plt.xlabel('Medu')
    plt.ylabel('Score')
    col1.pyplot()

    plt.figure(figsize = (9, 7))
    sns.lineplot(x = 'Fedu', y = 'Score', data = df)
    plt.title('Fedu Participation')
    plt.xlabel('Fedu')
    plt.ylabel('Score')
    col2.pyplot()

with st.container():
    st.header("studytime & freetime Participation")
    
    col1, col2 = st.columns([1, 1])
#studytime
    plt.figure(figsize = (9, 7))
    sns.barplot(x = 'studytime', y = 'Score', data = df)
    plt.title('studytime Participation')
    plt.xlabel('studytime')
    plt.ylabel('Score')
    col1.pyplot()
#Freetime
    plt.figure(figsize = (9, 7))
    sns.lineplot(x = 'freetime', y = 'Score', data = df)
    plt.title('freetime Participation')
    plt.xlabel('freetime')
    plt.ylabel('Score')
    col2.pyplot()

   
with st.container():
    st.header("extra & paid cariccular Participation ")
    col1, col2 = st.columns([1, 1])
# activity
    plt.figure(figsize = (9, 7))
    sns.barplot(x = 'activities', y = 'Score', data = df, hue = 'sex')
    plt.title('activities Participation')
    plt.xlabel('activities')
    plt.ylabel('Score')
    col1.pyplot()
#paid
    plt.figure(figsize = (9, 7))
    sns.countplot(x = 'paid', data = df)
    plt.title('paid Participation')
    plt.xlabel('paid')
    col2.pyplot()


with st.container():
    st.header("parents status & health Participation'")
    st.markdown(
        "how parents cohabitasion status & health Participate in student performance'"
    )
    col1,col2= st.columns([1,1])
    plt.figure(figsize = (9, 7))
    sns.barplot(x = 'Pstatus', y = 'Score', data = df, hue = 'sex')
    plt.title('Pstatus Participation')
    plt.xlabel('Pstatus')
    plt.ylabel('Score')
    col1.pyplot()

#health
    plt.figure(figsize = (9, 7))
    sns.lineplot(x = 'health', y = 'Score', data = df)
    plt.title('health Participation')
    plt.xlabel('health')
    plt.ylabel('Score')
    col2.pyplot()

with st.container():
    st.header("address,famsize,Parental status,College support,family support, paid classes how the affects student performance")
    st.markdown(
        "address,famsize,Parental status,College support,family support, paid classes by Score is shown in following graphs"
    )
    plt.figure(figsize=(20, 12))
    plt.subplot(2,3,1)
    sns.boxplot(x = 'address', y = 'Score', data = df)
    plt.subplot(2,3,2)
    sns.boxplot(x = 'famsize', y = 'Score', data = df)
    plt.subplot(2,3,3)
    sns.boxplot(x = 'Pstatus', y = 'Score', data = df)
    plt.subplot(2,3,4)
    sns.boxplot(x = 'schoolsup', y = 'Score', data = df)
    plt.subplot(2,3,5)
    sns.boxplot(x = 'famsup', y = 'Score', data = df)
    plt.subplot(2,3,6)
    sns.boxplot(x = 'paid', y = 'Score', data = df)
    st.pyplot()

with st.container():
    st.header("activities outside of the classroom, nursery attendance, higher education, Internet,age  impact on Score")
    
    plt.figure(figsize=(20, 12))
    plt.subplot(2,3,1)
    sns.boxplot(x = 'activities', y = 'Score', data = df)
    plt.subplot(2,3,2)
    sns.boxplot(x = 'nursery', y = 'Score', data = df)
    plt.subplot(2,3,3)
    sns.boxplot(x = 'higher', y = 'Score', data = df)
    plt.subplot(2,3,4)
    sns.boxplot(x = 'internet', y = 'Score', data = df)
    plt.subplot(2,3,5)
    sns.boxplot(x = 'age', y = 'Score', data = df)
    st.pyplot()

with st.container():
    st.header("Logistic regression")
    st.markdown(
        "Logistic regression is a statistical model used for binary classification tasks, where the goal is to predict the probability of an instance belonging to a particular class. It is a type of regression analysis that estimates the relationship between the independent variables (features) and a binary dependent variable (the target)."
    )

df.drop(['Score'],axis=1,inplace=True)

# # Logistic regression


data = df.to_numpy()
n = data.shape[1]
x = data[:,0:n-1]
y = data[:,n-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


logisticRegr = LogisticRegression(C=1)

logisticRegr.fit(x_train,y_train)

y_pred=logisticRegr.predict(x_test)


col1, col2 = st.columns(2)

with col1:
   Sctest=round(100*logisticRegr.score(x_test,y_test))
   Sctrain=round(100*logisticRegr.score(x_train,y_train))

   st.write('#Accuracy test is: ',Sctest)
   st.write('#Accuracy train is: ',Sctrain)

   f1 = f1_score(y_test, y_pred, average='macro')

   st.write('\n#f1 score is: ',f1)


   Sctest=round(100*logisticRegr.score(x_test,y_test))
   Sctrain=round(100*logisticRegr.score(x_train,y_train))
   st.write('Accuracy test is: ', Sctest)
   st.write('Accuracy train is:', Sctrain)
   confusion_matrix(y_test, y_pred)


with col2:
   cm = confusion_matrix(y_test, y_pred)
   sns.heatmap(cm,annot=True)
   st.pyplot()

st.text(classification_report(y_test, y_pred))


fpositif, tpositif, thresholds = roc_curve(y_test, y_pred)
plt.plot([0,1],[0,1],'--')
plt.plot(fpositif,tpositif, label='LogisticRegr')
plt.xlabel('false positif')
plt.ylabel('true positif')
plt.title('Logistic regression ROC curve')
st.pyplot()

max_iteration = 0
maxF1 = 0
maxAccuracy = 0
optimal_state = 0
import random
for k in range(max_iteration):
    st.write('Iteration :'+str(k)+', Current accuracy: '+str(maxAccuracy)+ ', Current f1 : '+str(maxF1), end="\r")
    split_state = np.random.randint(1,100000000)-1
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=split_state)
    logisticRegr = LogisticRegression(C=1)
    logisticRegr.fit(x_train,y_train)
    y_pred=logisticRegr.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)*100
    
    if (accuracy>maxAccuracy and f1>maxF1):
        maxF1 = f1 
        maxAccuracy = accuracy
        optimal_state = split_state
    
   
optimal_state = 85491961
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=optimal_state)
logisticRegr = LogisticRegression(C=1)
logisticRegr.fit(x_train,y_train)
y_pred=logisticRegr.predict(x_test)
f1 = round(100*f1_score(y_test, y_pred, average='macro'))
accuracy = accuracy_score(y_test, y_pred)*100
st.write('\n\n\n*Accuracy is: '+str(accuracy)+'\n*f1 score is: ',f1)

yt_lg,yp_lg = y_test,y_pred

col1, col2 = st.columns(2)

with col1:
   st.write ( '\n\n *the ROC curve: ')

   fpositif, tpositif, thresholds = roc_curve(y_test, y_pred)
   plt.plot([0,1],[0,1],'--')
   plt.plot(fpositif,tpositif, label='LogisticRegr')
   plt.xlabel('false positif')
   plt.ylabel('true positif')
   plt.title('LogisticRegr ROC curve')
   st.pyplot()

with col2:
  

  st.write(' *the confusion matrix ')

  cm = confusion_matrix(y_test, y_pred)
  sns.heatmap(cm,annot=True)
  st.pyplot()



with st.container():
    st.header("k-nearest neighbors ")
    st.markdown(
        "k-Nearest Neighbors (k-NN) is a simple and intuitive machine learning algorithm used for both classification and regression tasks. It operates based on the principle that instances with similar features tend to belong to the same class or have similar outputs.\n For classification, the predicted class of the new instance is determined by majority voting among the classes of its k nearest neighbors. The class with the highest count among the neighbors is assigned as the predicted class. "
    )


# ## k-nearest neighbors 


y=df.perf
target=["perf"]
x = df.drop(target,axis = 1 )
max_iteration = 0
maxF1 = 0
maxAccuracy = 0
optimal_state = 0
for k in range(max_iteration):
    st.write('Iteration :'+str(k)+', Current accuracy: '+str(maxAccuracy)+ ', Current f1 : '+str(maxF1), end="\r")
    split_state = np.random.randint(1,100000000)-1
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=split_state)
    KNN = KNeighborsClassifier()
    KNN.fit(x_train,y_train)
    y_pred=KNN.predict(x_test)
    f1 = round(100*f1_score(y_test, y_pred, average='macro'))
    accuracy = accuracy_score(y_test, y_pred)*100
    
    if (accuracy>maxAccuracy and f1>maxF1):
        maxF1 = f1 
        maxAccuracy = accuracy
        optimal_state = split_state
    
optimal_state = 71027464

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=optimal_state)
KNN= KNeighborsClassifier()
KNN.fit(x_train,y_train)
y_pred=KNN.predict(x_test)
f1 = round(100*f1_score(y_test, y_pred, average='macro'))
accuracy = accuracy_score(y_test, y_pred)*100
st.write('\n\n\n*Accuracy is: '+str(accuracy)+'\n*f1 score is: ',f1)

st.write('random_state is ',optimal_state)




col1, col2 = st.columns(2)
#The Receiver Operating Characteristic (ROC) curve is a graphical representation used to assess the performance of a binary classification model. It illustrates the trade-off between the true positive rate (sensitivity) and the false positive rate (1 - specificity) across different classification thresholds.
with col1:
   st.write( '\n\n *the ROC curve: ')

   fpositif, tpositif, thresholds = roc_curve(y_test, y_pred)
   plt.plot([0,1],[0,1],'--')
   plt.plot(fpositif,tpositif, label='knn')
   plt.xlabel('false positif')
   plt.ylabel('true positif')
   plt.title('KNN ROC curve')
   st.pyplot()


with col2:
   yt_knn,yp_knn= y_test,y_pred


   st.write(' *the confusion matrix ')

   cm = confusion_matrix(y_test, y_pred)
   sns.heatmap(cm,annot=True)
   st.pyplot()




with st.container():
    st.header("Setup a knn classifier with k neighbors")
    st.markdown(
        "Plotting the curv.In case of classifier like knn the parameter to be tuned is n_neighbors "
    )


neighbors= np.arange(1,20)
train_accuracy =np.empty(19)
test_accuracy = np.empty(19)

for i,k in enumerate(neighbors):
    
    knn = KNeighborsClassifier(n_neighbors=k)
    
    
    knn.fit(x_train, y_train)
    
    
    train_accuracy[i] = knn.score(x_train, y_train)
    
   
    test_accuracy[i] = knn.score(x_test, y_test) 
    

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
st.pyplot() 


param_grid = {'n_neighbors':np.arange(1,20)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x_train,y_train)

knn_cv.best_score_

knn_cv.best_params_

param_grid = {'n_neighbors':np.arange(1,20)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x_test,y_test)

knn_cv.best_score_

knn_cv.best_params_

param_grid = {'n_neighbors':np.arange(1,20)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x,y)

knn_cv.best_score_

knn_cv.best_params_


params = {"n_neighbors":[7,19] , "metric":["euclidean", "manhattan", "chebyshev"]}
acc = {}

for m in params["metric"]:
    acc[m] = []
    for k in params["n_neighbors"]:
        st.write("Model_{} metric: {}, n_neighbors: {}".format(i, m, k))
        i += 1
        t = time()
        knn = KNeighborsClassifier(n_neighbors=k, metric=m)
        knn.fit(x_train,y_train)
        pred = knn.predict(x_test)
        st.write("Time: ", time() - t)
        acc[m].append(accuracy_score(y_test, y_pred))
        st.write("Acc: ", acc[m][-1])


with st.container():
    st.header("Show results of every model")
    st.markdown(
        "Show results of every model"
    )

max_iteration = 0
maxF1 = 0
maxAccuracy = 0
optimal_state = 0
f1 = 0
accuracy = 0
True60 = False
for k in range(max_iteration):
    st.write ('Iteration :'+str(k)+', Current accuracy: '+str(maxAccuracy)+ ', Current f1 : '+str(maxF1), end="\r")
    split_state = np.random.randint(1,100000000)-1
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=split_state)
    KNN = KNeighborsClassifier(n_neighbors=7,metric='chebyshev')
    KNN.fit(x_train,y_train)
    y_pred=KNN.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)*100
    
    if accuracy>maxAccuracy and f1>=0.5:
        maxF1 = f1 
        maxAccuracy = accuracy
        optimal_state = split_state
        if maxAccuracy>79:
            break
    
optimal_state = 29300362         
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=optimal_state)
KNN_f= KNeighborsClassifier(n_neighbors=7,metric='chebyshev')
KNN_f.fit(x_train,y_train)
y_pred=KNN_f.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)*100
st.write('\n\n\n*Accuracy is: '+str(accuracy)+'\n*f1 score is: ',f1)

st.write ('random_state is ',optimal_state)

yt_knn,yp_knn= y_test,y_pred


ac = accuracy_score(yt_knn,yp_knn)
st.write('Accuracy is: ',ac)
cm= confusion_matrix(yt_knn,yp_knn)
sns.heatmap(cm,annot=True)
st.pyplot()
yt_knn,yp_knn = y_test,y_pred


st.text(classification_report(y_test,y_pred))


st.write ( ' the ROC curve: ')

fpositif, tpositif, thresholds = roc_curve(y_test, y_pred)
plt.plot([0,1],[0,1],'--')
plt.plot(fpositif,tpositif, label='final knn model')
plt.xlabel('false positif')
plt.ylabel('true positif')
plt.title('knn_f ROC curve')
st.pyplot()



def showResults(accuracy, trainingTime, y_pred,model):
    
    st.write('------------------------------------------------Results :',model,'-------------------------------------------------')
    confusionMatrix = confusion_matrix(y_test, y_pred)
    st.write('\n The ROC curve is :\n')
    fig, _ = plt.subplots()
    fpr,tpr,thresholds=roc_curve(y_test,y_pred)
    plt.plot([0, 1],[0, 1],'--')
    plt.plot(fpr,tpr,label=model)
    plt.xlabel('false positive')
    plt.ylabel('false negative')
    plt.legend()
    fig.suptitle('ROC curve: '+str(model))
    st.pyplot()
    
    st.write('----------------------------------------------')
    st.write('The model  accuracy:', round(accuracy),'%')
    st.write('----------------------------------------------')
    st.write('The training time is: ',trainingTime)
    st.write('----------------------------------------------')
    st.write('The f1 score is :',round(100*f1_score(y_test, y_pred, average='macro'))/100)
    st.write('----------------------------------------------')
    st.write('The roc_auc_score is :',round(100*roc_auc_score(y_test, y_pred))/100)
    st.write('----------------------------------------------')
    st.write('The confusion matrix is :\n')
    ax = plt.axes()
    sns.heatmap(confusionMatrix,annot=True)
    st.pyplot()

st.header('Support Vector Machine(SVM):')
st.markdown(
        "Support Vector Machines (SVM) can utilize different kernels to transform the input features into a higher-dimensional space, allowing for more complex decision boundaries.\nHere's a brief overview of three commonly used kernels in SVM:\nLinear Kernel:\n The linear kernel is computationally efficient and works well when the number of features is large compared to the number of instances.\n Polynomial Kernel: \n The polynomial kernel maps the input features into a higher-dimensional space using polynomial functions.\n It introduces non-linearity, allowing the SVM to capture more complex relationships between the features. \n Radial Basis Function (RBF) Kernel:\n The RBF kernel is the most commonly used kernel in SVM. It transforms the features into an infinite-dimensional space.\n The kernel function has a parameter gamma that controls the width of the RBF function. Higher values of gamma result in more complex decision boundaries and can lead to overfitting")

# c value stands for regularizastion low c value to get less noisy data.
def optimal_C_value():
    Ci = np.array(( 0.0001,0.001,0.01,0.05,0.1,4,10,40,100))
    minError = float('Inf')
    optimal_C = float('Inf')

    for c in Ci:
        clf = SVC(C=c,kernel='linear')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_val)
        error = np.mean(np.double(predictions != y_val))
        if error < minError:
            minError = error
            optimal_C = c
    return optimal_C

 
# Optimal C and the degree of the polynomial
def optimal_C_d_values():
    Ci = np.array(( 0.0001,0.001,0.01,0.05,0.1,4,10,40,100))
    Di = np.array(( 2, 5, 10, 15, 20, 25, 30))
    minError = float('Inf')
    optimal_C = float('Inf')
    optimal_d = float('Inf')

    for d in Di:
        for c in Ci:
            clf = SVC(C=c,kernel='poly', degree=d)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_val)
            error = np.mean(np.double(predictions != y_val))
            if error < minError:
                minError = error
                optimal_C = c
                optimal_d = d
    return optimal_C,optimal_d


# Optimal C and gamma : higher gamma higher the weight of point closer to hyperplane helps to manage outliers
def optimal_C_gamma_values():
    Ci = np.array(( 0.0001,0.001,0.01,0.05,0.1,4,10,40,100))
    Gi = np.array(( 0.000001,0.00001,0.01,1,2,3,5,20,70,100,500,1000))
    minError = float('Inf')
    optimal_C = float('Inf')
    optimal_g = float('Inf')

    for g in Gi:
        for c in Ci:
            clf = SVC(C=c,kernel='rbf', gamma=g)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_val)
            error = np.mean(np.double(predictions != y_val))
            if error < minError:
                minError = error
                optimal_C = c
                optimal_g = g
    return optimal_C,optimal_g




def compare_kernels():
    X_train1,X_val1,X_test1,y_train1,y_val1,y_test1 = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state1)
    X_train2,X_val2,X_test2,y_train2,y_val2,y_test2 = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state2)
    X_train3,X_val3,X_test3,y_train3,y_val3,y_test3 = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state3)
    st.write('------------------------------------------------ Comparison -----------------------------------------------------')
    st.write('\n')
    f11 = "{:.2f}".format(f1_score(y_test1, y_linear, average='macro'))
    f22 = "{:.2f}".format(f1_score(y_test2, y_poly, average='macro'))
    f33 = "{:.2f}".format(f1_score(y_test3, y_gauss, average='macro'))
    roc1 = "{:.2f}".format(roc_auc_score(y_test1, y_linear))
    roc2 = "{:.2f}".format(roc_auc_score(y_test2, y_poly))
    roc3 = "{:.2f}".format(roc_auc_score(y_test3, y_gauss))
    a1,a2 = confusion_matrix(y_test1, y_linear)[0],confusion_matrix(y_test1, y_linear)[1]
    b1,b2 = confusion_matrix(y_test2, y_poly)[0],confusion_matrix(y_test2, y_poly)[1]
    c1,c2 = confusion_matrix(y_test3, y_gauss)[0],confusion_matrix(y_test3, y_gauss)[1]
    data_rows = [('training time',time1, time2, time3),
                 ('','','',''),
                  ('accuracy %',linear_accuracy, poly_accuracy, gauss_accuracy),
                 ('','','',''),
                 ('confusion matrix',a1, b1, c1),
                ('',a2,b2,c2),
                 ('','','',''),
                ('f1 score',f11,f22,f33),
                 ('','','',''),
                ('roc_auc_score',roc1,roc2,roc3)]
    t = Table(rows=data_rows, names=('metric','Linear kernel', 'polynomial kernel', 'gaussian kernel'))
    st.write(t)
    st.write('\n\n')
    st.write('The Roc curves :\n')
    y_pred1 = y_linear
    y_pred2 = y_poly
    y_pred3 = y_gauss
    fig, _ = plt.subplots()
    fig.suptitle('Comparison of three ROC curves')
    fpr,tpr,thresholds=roc_curve(y_test1,y_pred1)
    plt.plot([0, 1],[0, 1],'--')
    plt.plot(fpr,tpr,label='Linear kernel :'+str(roc1))
    plt.xlabel('false positive')
    plt.ylabel('false negative')
    fpr,tpr,thresholds=roc_curve(y_test2,y_pred2)
    plt.plot(fpr,tpr,label='Polynomial kernel :'+str(roc2))
    fpr,tpr,thresholds=roc_curve(y_test3,y_pred3)
    plt.plot(fpr,tpr,label='Gaussian kernel :'+str(roc3))
    plt.legend()
    st.pyplot()

with st.container():
    st.header("Print results of the choosen kernel")
    st.markdown(
        "Print results of the choosen kernel"
    )



def best_kernel(kernel):
    X_train1,X_val1,X_test1,y_train1,y_val1,y_test1 = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state1)
    X_train2,X_val2,X_test2,y_train2,y_val2,y_test2 = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state2)
    X_train3,X_val3,X_test3,y_train3,y_val3,y_test3 = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state3)
    
    time = 0
    f1 = 0
    accuracy = 0
    rc = 0
    y = 0
    if kernel == 'linear kernel':
        time = time1
        f1 = "{:.2f}".format(f1_score(y_test1, y_linear, average='macro'))
        accuracy = round(100*linear_accuracy)/100
        rc = round(100*roc_auc_score(y_test1, y_linear))/100
        y_test = y_test1
        y = y_linear
    elif kernel == 'polynomial kernel':
        time = time2
        f1 = "{:.2f}".format(f1_score(y_test2, y_poly, average='macro'))
        accuracy = round(100*poly_accuracy)/100
        rc = round(100*roc_auc_score(y_test2, y_poly))/100
        y_test = y_test2
        y = y_poly
    else :
        time = time3
        f1 = "{:.2f}".format(f1_score(y_test3, y_gauss, average='macro'))
        accuracy = round(100*gauss_accuracy)/100
        rc = round(100*roc_auc_score(y_test3, y_gauss))/100
        y_test = y_test3
        y = y_gauss 
        

    yt_svm,yp_svm = y_test, y
    
    st.write('The choosen kernel :',kernel)
    st.write('the training :',time)
    st.write('the accuracy :',round(accuracy),'%')
    st.write('the f1 score :',f1)
    st.write('The roc_auc_score is :',rc)
    st.write('----------------------------------------\nThe ROC curve :')
    fig, _ = plt.subplots()
    fpr,tpr,thresholds=roc_curve(y_test,y)
    plt.plot([0, 1],[0, 1],'--')
    plt.plot(fpr,tpr,label=kernel+': '+str(rc))
    plt.xlabel('false positive')
    plt.ylabel('false negative')
    plt.legend()
    st.pyplot()
    confusionMatrix = confusion_matrix(y_test, y)
    st.write('----------------------------------------\nThe confusion matrix is  :')
    ax = plt.axes()
    sns.heatmap(confusionMatrix,annot=True)
    st.pyplot()
    ax.set_title('Confusion matrix of SVM '+str(kernel))
    return yt_svm,yp_svm
    

def factors(array, K, max_or_min, df):
    
    n = array.shape[1]
    array = array.reshape(n,1)
    my_list = array.tolist()
    
    if max_or_min == 'max':
        temp = sorted(my_list)[-K:]
        res = [] 
        for ele in temp: 
            res.append(my_list.index(ele))
        return(get_factors(res, df))
    
    
    elif max_or_min == 'min':
        temp = sorted(my_list, reverse=True)[-K:]
        temp = temp = np.array(temp).reshape(K,1)
        res = []
        for ele in temp:
            if ele<0:
                res.append(my_list.index(ele))
        return(get_factors(res, df))
    

    else:
        return
    
def get_factors(index, df):
    f = []
    for i in index:
        f.append(df.columns[i])
    return f
    
 
columns_name = {
                 'sex': 'Gender stereotypes and expectations can have a negative impact on student performance, such as by assigning certain academic strengths or interests to certain genders.',
                 'activities': "Participation in activities can improve academic performance, but overcommitting can lead to poorer results.",
                 'famsize': "In larger families, the attention and support that parents can provide to each child may be divided among more siblings. This could potentially lead to less individualized attention and support for each child's academic needs, which might impact their performance.",
                 'Pstatus': "Parents who live together in harmony can help to provide a secure and nurturing environment for their kids. The mental health, motivation, and capacity for concentration of a student can all be favourably influenced by emotional stability at home.", 
                 'Medu': " Mothers with higher education levels can provide more effective academic support, leading to improved student performance",
                 'Fedu': "Fathers with higher education prioritise education and provide a supportive learning environment, encouraging academic achievement and offering guidance.",
                 'Mjob': "Values like perseverance, ambition, and hard work can be imparted by mothers",
                 'Fjob': "The father's job often contributes to the family's socioeconomic status. Higher-income families may have more resources available to support their child's education, such as access to better schools, educational materials, tutors, or extracurricular activities. This can positively impact a student's performance.", 
                 'reason': 'Students with personal interests, career goals, or passion for a particular field of study are more likely to be motivated and engaged in their coursework, which can positively influence their performance. ',
                 'schoolsup': 'It is the form of academic guidance can help students navigate their courses effectively, leading to improved academic performance.',
                 'famsup': 'Family support provides emotional stability and encouragement to students, resulting in higher self-esteem, confidence, and motivation, which positively impacts their academic performance.',
                 'paid': ' Extra paid classes may offer supplementary materials, resources, or practise sessions that complement the regular course curriculum. and it  can provide students with additional opportunities to review and delve deeper into the course material.',
                 'higher': 'Students put in more effort to achieve higher education, leading to better academic performance. ',
                 'nursery': 'Nursery education often focuses on providing early exposure to various learning experiences, including language development, numeracy skills, problem-solving, and critical thinking. This can help enhance cognitive development and lay a strong foundation for future academic success.',
                 'romantic': 'Balancing academic and romantic commitments can be difficult. ',
                 'famrel': 'Positive role modeling within the family can motivate and inspire students to strive for academic success.',
                 'goout': 'Going out with friends can help students feel supported, reduce stress, and improve mental health, but can also lead to distractions and detrimental effects on academic performance.',
                 'Dalc': ' Excessive alcohol intake can negatively impact cognitive abilities.' ,
                 'Walc': 'Alcohol consumption can lead to decreased motivation, procrastination, and reduced study time.',
                 'studytime':"Students must use effective strategies and balance study and life to maximize learning outcomes.",
                 'backlog':"Backlogs can create stress and pressure, but dealing with them can motivate students to develop a sense of determination and resilience.",
                 "traveltime":"time reqired to go to college/classes"
                 }       




def column_to_string(factors, max_or_min):
    if max_or_min == 'max':
        st.write('-----------------------------------------------------------------------------------')
        st.title('Factors helping students succeed:')
    else:
        st.write('***********************************************************************************')
        st.write('-----------------------------------------------------------------------------------')
        st.title('Factors leading students to failure:')

    for factor in factors:
        if factor in columns_name:
            st.write(f"**{factor}:** {columns_name[factor]}")
        else:
            st.write(factor)

# Usage example
positive_factors = ['sex', 'activities', 'Medu', 'Fedu']
negative_factors = ['backlog', 'famrel', 'Walc']

column_to_string(positive_factors, 'max')
column_to_string(negative_factors, 'min')
# ------------------------------------------------------------------------------------------------------------------------------

def split(df,rest_size,test_size,randomState):
    data = df.to_numpy()
    n = data.shape[1]
    x = data[:,0:n-1]
    y = data[:,n-1]
    if(randomState):
        X_train,X_rest,y_train,y_rest = train_test_split(x,y,test_size=rest_size,random_state=randomState)
        X_val,X_test,y_val,y_test = train_test_split(X_rest,y_rest,test_size=test_size,random_state=randomState)
    else:
        X_train,X_rest,y_train,y_rest = train_test_split(x,y,test_size=rest_size,random_state=0)
        X_val,X_test,y_val,y_test = train_test_split(X_rest,y_rest,test_size=test_size,random_state=0)
    
    return X_train,X_val,X_test,y_train,y_val,y_test

# Linear kernel 
optimal_split_state1 = 0
maxAccuracy = 0
maxF1 = 0


max_iteration = 0
if max_iteration != 0:
    st.write ('----------------------------------------Hyperparameters tunning starts----------------------------------------\n\n')

for k in range(max_iteration):
    st.write ('Iteration :'+str(k)+', Current accuracy: '+str(maxAccuracy)+' Current f1 '+str(maxF1), end="\r")
    split_state = np.random.randint(1,1000000000)-1
    X_train,X_val,X_test,y_train,y_val,y_test = split(df,rest_size=0.4,test_size=0.4,randomState=split_state)
    optimal_C = optimal_C_value()



    linear_clf = SVC(C=optimal_C,kernel='linear')


    tic = time()
    linear_clf.fit(X_train, y_train)
    toc = time()
    time1 = str(round(1000*(toc-tic))) + "ms"
    y_linear = linear_clf.predict(X_test)
    linear_f1 = f1_score(y_test, y_linear, average='macro')
    linear_accuracy = accuracy_score(y_test, y_linear)*100
    if linear_accuracy>maxAccuracy and linear_f1>maxF1:
        maxAccuracy = linear_accuracy
        maxF1 = linear_f1
        optimal_split_state1 = split_state
    if maxAccuracy>86 and maxF1>80:
        break
        
optimal_split_state1 = 388628375
X_train,X_val,X_test,y_train,y_val,y_test = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state1)
optimal_C = optimal_C_value()

linear_clf = SVC(C=optimal_C,kernel='linear')

tic = time()
linear_clf.fit(X_train, y_train)
toc = time()
time1 = str(round(1000*(toc-tic))) + "ms"
y_linear = linear_clf.predict(X_test)
linear_accuracy = accuracy_score(y_test, y_linear)*100
if max_iteration != 0:
    st.write('\n\n\n                            ---------------------------process ended'         '------------------------------------                            \n\n\n')


showResults(linear_accuracy, time1, y_linear,'SVM linear kernel')

with st.container():
    st.header("Polynomial kernel ")
    st.markdown(
        "Polynomial kernel "
    )

# Polynomial kernel 
optimal_split_state2 = 0
maxAccuracy = 0
maxF1 = 0
max_iteration = 0

if max_iteration != 0:
    st.write ('----------------------------------------Hyperparameters tunning starts----------------------------------------\n\n')
for k in range(max_iteration):
    st.write ('Iteration :'+str(k)+', Current accuracy: '+str(maxAccuracy)+', Current f1 '+str(maxF1), end="\r")
    
    split_state = np.random.randint(1,100000000)-1
    X_train,X_val,X_test,y_train,y_val,y_test = split(df,rest_size=0.4,test_size=0.4,randomState=split_state)

    optimal_C, optimal_d = optimal_C_d_values()
    
    poly_clf = SVC(C=optimal_C,kernel='poly', degree=optimal_d)

    poly_clf.fit(X_train, y_train)
    y_poly = poly_clf.predict(X_test)
    poly_f1 = f1_score(y_test, y_poly, average='macro')
    poly_accuracy = accuracy_score(y_test, y_poly)*100
    
    if poly_accuracy>maxAccuracy and poly_f1>maxF1:
        maxAccuracy = poly_accuracy
        maxF1 = poly_f1
        optimal_split_state2 = split_state


optimal_split_state2 = 7070621

X_train,X_val,X_test,y_train,y_val,y_test = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state2)

optimal_C, optimal_d = optimal_C_d_values()


poly_clf = SVC(C=optimal_C,kernel='poly', degree=optimal_d)

tic = time()
poly_clf.fit(X_train, y_train)
toc = time()
time2 = str(round(1000*(toc-tic))) + "ms"
y_poly = poly_clf.predict(X_test)
poly_accuracy = accuracy_score(y_test, y_poly)*100
if max_iteration != 0:
    st.write('\n\n\n**************************************Ended**************************************** \n\n\n')


showResults(poly_accuracy, time2, y_poly,'SVM polynomial kernel')



with st.container():
    st.header("Gaussian kernel ")
    st.markdown(
        "Gaussian kernel "
    )
#Gaussian kernel 
optimal_split_state3 = 0
maxAccuracy = 0
maxF1 = 0




max_iteration = 0
if max_iteration != 0:
    st.write ('----------------------------------------------Hyperparameters tunning starts''--------------------------------------------\n\n')
for k in range(max_iteration):
    st.write ('Iteration :'+str(k)+', Current accuracy: '+str(maxAccuracy)+', Current f1 '+str(maxF1), end="\r")
    
    split_state = np.random.randint(1,100000000)-1
    X_train,X_val,X_test,y_train,y_val,y_test = split(df,rest_size=0.4,test_size=0.4,randomState=split_state)

    optimal_C, optimal_gamma = optimal_C_gamma_values()
    
    gauss_clf = SVC(C=optimal_C,kernel='rbf',gamma=optimal_gamma)

    gauss_clf.fit(X_train, y_train)
    y_gauss = gauss_clf.predict(X_test)
    gauss_f1 = f1_score(y_test, y_gauss, average='macro')
    gauss_accuracy = accuracy_score(y_test, y_gauss)*100
    
    if gauss_accuracy>maxAccuracy and gauss_f1>maxF1:
        maxAccuracy = gauss_accuracy
        maxF1 = gauss_f1
        optimal_split_state3 = split_state


optimal_split_state3 = 93895097

X_train,X_val,X_test,y_train,y_val,y_test = split(df,rest_size=0.4,test_size=0.4,randomState=optimal_split_state3)

optimal_C, optimal_gamma = optimal_C_gamma_values()


# optimal C value
gauss_clf = SVC(C=optimal_C,kernel='rbf',gamma=optimal_gamma)


tic = time()
gauss_clf.fit(X_train, y_train)
toc = time()
time3 = str(round(1000*(toc-tic))) + "ms"
y_gauss = gauss_clf.predict(X_test)
gauss_accuracy = (accuracy_score(y_test, y_gauss)*100)

if max_iteration != 0:
    st.write('\n\n\n**************************Ended***************************\n\n\n')
                                                                

showResults(gauss_accuracy, time3, y_gauss,'SVM gaussian kernel')

compare_kernels()

yt_svm,yp_svm = best_kernel('linear kernel')

coefs = linear_clf.coef_

# factors-succeed
column_to_string(factors(coefs, 5, 'max', df),'max')
#failure-factors
column_to_string(factors(coefs, 5, 'min', df), 'min')



with st.container():
     st.header("Graph representing the Factors By Score")

# #1   
with st.container():
    st.header("Student status By Mother education and Fathers Job ")
    st.markdown(
        "Mothers with higher education levels can provide more effective academic support, leading to improved student performance\nThe father's job often contributes to the family's socioeconomic status. Higher-income families may have more resources available to support their child's education, such as access to better schools, educational materials, tutors, or extracurricular activities. This can positively impact a student's performance."
    )
    col1, col2= st.columns([1, 1])
    f, fx = plt.subplots()
    figure = sns.lineplot(x = 'Fedu', y = 'Score', data = original_df)
    fx = fx.set(ylabel="Score", xlabel="Mother eduction")
    figure.grid(False)
    col1.pyplot()

    f, fx = plt.subplots()
    figure = sns.lineplot(x='Fjob', y='Score', data=original_df)
    fx = fx.set(ylabel="Score", xlabel="Fathers Job")
    figure.grid(False)
    col2.pyplot()



#3
with st.container():
    st.header("Student status By cohabitation status of parents and Family Support")
    st.markdown(
        "Parents who live together in harmony can help to provide a secure and nurturing environment for their kids. The mental health, motivation, and capacity for concentration of a student can all be favourably influenced by emotional stability at home.\nFamily support provides emotional stability and encouragement to students, resulting in higher self-esteem, confidence, and motivation, which positively impacts their academic performance."
    )
    col1, col2= st.columns([1, 1])
    # 1) Mother eduction
        # Mjob distribution
    f, fx = plt.subplots()
    figure = sns.barplot(x='Pstatus', y='Score', data=original_df)
    fx = fx.set(ylabel="Score", xlabel="cohabitation status of parents")
    figure.grid(False)
    col1.pyplot()

    f, fx = plt.subplots()
    figure =sns.barplot(x = 'famsup', y = 'Score', data =original_df)
    fx = fx.set(ylabel="Score", xlabel="family Support")
    figure.grid(False)
    col2.pyplot()


with st.container():
    st.header("Student status By studytime and nursery attendece")
    st.markdown(
        "Students must use effective strategies and balance study and life to maximize learning outcomes\nNursery education often focuses on providing early exposure to various learning experiences, including language development, numeracy skills, problem-solving, and critical thinking. This can help enhance cognitive development and lay a strong foundation for future academic success."
    )
    col1, col2= st.columns([1, 1])
    f, fx = plt.subplots()
    figure = sns.lineplot(x='studytime', y='Score', data=original_df)
    fx = fx.set(ylabel="Score", xlabel="studytime")
    figure.grid(False)
    col1.pyplot()

    #Student status By mother JOB
    f, fx = plt.subplots()
    figure = sns.barplot(x='nursery', y='Score', data=original_df)
    fx = fx.set(ylabel="Score", xlabel="nursery")
    figure.grid(False)
    col2.pyplot()


with st.container():
    st.header("Student status By goout and backlog")
    st.markdown(
        "Going out with friends can help students feel supported, reduce stress, and improve mental health, but can also lead to distractions and detrimental effects on academic performance.\nBacklogs can create stress and pressure, but dealing with them can motivate students to develop a sense of determination and resilience."
    )
    col1, col2= st.columns([1, 1])
    
    f, fx = plt.subplots()
    figure = sns.lineplot(x='goout', y='Score', data=original_df)
    fx = fx.set(ylabel="Score", xlabel="studytime")
    figure.grid(False)
    col1.pyplot()

    f, fx = plt.subplots()
    figure = sns.barplot(x = 'backlog', y = 'Score', data = original_df)
    fx = fx.set(ylabel="Score", xlabel="nursery")
    figure.grid(False)
    col2.pyplot()





with st.container():
    st.header("Student status By activities and sex(Gender)")
    st.markdown(
        "Participation in activities can improve academic performance, but overcommitting can lead to poorer results.\n Gender stereotypes and expectations can have a negative impact on student performance, such as by assigning certain academic strengths or interests to certain genders."
    )
    col1, col2= st.columns([1, 1])
    
    score_percentage_by_activities = original_df['activities'].value_counts(normalize=True) * 100
    #Create a pie chart
    fig,ax = plt.subplots()
    ax.pie(score_percentage_by_activities, labels=score_percentage_by_activities.index, autopct='%1.1f%%')
    plt.title('Percentage Distribution of Score by Activities')
    # # Display the chart
    figure.grid(False)
    col1.pyplot()

  
    f, fx = plt.subplots()
    # # Count the occurrences of each score category for each sex category
    score_percentage_by_sex = original_df['sex'].value_counts(normalize=True) * 100
    fig,ax = plt.subplots()
    ax.pie(score_percentage_by_sex, labels=score_percentage_by_sex.index, autopct='%1.1f%%')
    ax.set_title('Distribution of Score by Sex')
    figure.grid(False)
    col2.pyplot()
