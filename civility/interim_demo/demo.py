import streamlit as st

from civility_classifier import civility_classifier
from introduction import introduction
from problem_framing import problem_framing
from tools_explored import tools_explored


# Demo title
st.title("Project Pensive: Building Recommender Systems with Civil Filters")

# Brief description of project
st.write(
    "The goal of Project Pensive is to build content conscious Recommender Systems."
)

# Sidebar
st.sidebar.header("Vector Institute: AI Engineering and Technology")
st.sidebar.image("images/vector_logo.jpeg", width=300)
sidebar = st.sidebar.selectbox(
    "Demo Section",
    ("Introduction", "Problem Framing", "Tools Explored", "Civility Classifier")
)

# Main page
if sidebar == "Introduction":
    introduction()

# Problem framing and steps
if sidebar == "Problem Framing":
    problem_framing()

# Tools explored
if sidebar == "Tools Explored":
    tools_explored()

# Civility classifier
if sidebar == "Civility Classifier":
    civility_classifier()

