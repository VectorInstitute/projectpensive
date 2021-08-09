import streamlit as st

from problem_framing import problem_framing
from demo import demo
from design import design
from evaluation import evaluation
from failed_attempts import failed_attempts
from next_steps import next_steps

st.set_page_config(page_title="Project Pensive",
                       page_icon="images/vertical_logo.jpg",
                       layout='wide',
                       initial_sidebar_state='expanded')

# Main header
col1, col2, col3, col4 = st.columns([2, 1, 0.1, 2])  
col2.title("Project Pensive")
col3.markdown(":thought_balloon:")

# Brief description of project
st.write(
    "Project Pensive is open source software that leverages the latest AI research to benefit those offering "
    "recommendations."
)
st.write(
    "The goal of Project Pensive is to build better Recommender Engines that incorporate content conscious filters to "
    "improve content quality and user experience."
)

# Sidebar
st.sidebar.header("Vector Institute: AI Engineering and Technology")
sidebar = st.sidebar.selectbox(
    "Demo Section",
    ("Problem Framing", "Demo", "Design", "Evaluation", "Failed Attempts", "Next Steps")
)

if sidebar == "Problem Framing":
    problem_framing()
elif sidebar == "Demo":
    demo()
elif sidebar == "Design":
    design()
elif sidebar == "Evaluation":
    evaluation()
elif sidebar == "Failed Attempts":
    failed_attempts()
elif sidebar == "Next Steps":
    next_steps()

col1, col2, col3 = st.columns([1, 1, 1])    
col1.image("images/horizontal_logo.jpg")