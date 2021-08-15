import streamlit as st

from problem_framing import problem_framing
from demo import demo
from design import design
from evaluation import evaluation
from failed_attempts import failed_attempts
from next_steps import next_steps
from conclusion import conclusion

st.set_page_config(
    page_title="Project Pensive",
    page_icon="images/vertical_logo.jpg",
    layout='wide',
    initial_sidebar_state='expanded'
)

# Main header
st.title("Project Pensive")

# Brief description of project
st.write(
    "Project Pensive is open source software that leverages the latest AI research to benefit those offering "
    "recommendations. The goal of Project Pensive is to build better Recommender Engines that incorporate content conscious filters to "
    "improve content quality and user experience."
)

# Sidebar
st.sidebar.header("Vector Institute: AI Engineering and Technology")
st.sidebar.image("images/vector_logo.jpeg", width=300)
sidebar = st.sidebar.selectbox(
    "Demo Section",
    ("Problem Framing", "Design", "Demo", "Evaluation", "Failed Attempts", "Next Steps", "Conclusion")
)

if sidebar == "Problem Framing":
    problem_framing()
elif sidebar == "Design":
    design()
elif sidebar == "Demo":
    demo()
elif sidebar == "Evaluation":
    evaluation()
elif sidebar == "Failed Attempts":
    failed_attempts()
elif sidebar == "Next Steps":
    next_steps()
elif sidebar == "Conclusion":
    conclusion()

if sidebar != "Conclusion":
    st.write("")  # Blank line
    st.markdown(
        "Made by [Michael Nasello](https://ca.linkedin.com/in/michael-nasello) and "
        "[Sheen Thusoo](https://ca.linkedin.com/in/sheenthusoo): Applied Machine Learning Interns, Summer 2021"
    )
