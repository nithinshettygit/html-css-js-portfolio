import streamlit as st
import google.generativeai as genai
import os
import dotenv # Make sure you've installed 'python-dotenv' via 'pip install python-dotenv'

# --- Configuration & API Setup ---
# Load environment variables from .env file.
# This ensures sensitive API keys are not hardcoded in the script.
dotenv.load_dotenv()

# Retrieve the Gemini API key from environment variables
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Stop the app if the API key is not found, providing a clear error message.
if not gemini_api_key:
    st.error("Gemini API key not found. Please set the GOOGLE_API_KEY environment variable in your .env file.")
    st.stop()

# Configure the Gemini API client.
# The 'client_options' are crucial for specifying the correct API endpoint,
# which often resolves '404 model not found' errors in certain regions or setups.
genai.configure(
    api_key=gemini_api_key,
    transport="rest", # 'rest' transport is generally robust for web applications
    client_options={"api_endpoint": "generativelanguage.googleapis.com"}
)

# Initialize the Gemini model for conversational AI.
# Using 'gemini-1.5-flash' for broad availability and efficiency.
# If you have access to 'gemini-1.5-pro' and need more complex reasoning, you can try that.
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    # Or for more advanced reasoning (if available):
    # model = genai.GenerativeModel('gemini-1.5-pro')
except Exception as e:
    st.error(f"Failed to initialize Gemini model. "
             f"Please ensure your GOOGLE_API_KEY is correct, the chosen model ('gemini-1.5-flash') "
             f"is available in your region, and you have a stable network connection. Error: {e}")
    st.stop() # Stop the script if the model cannot be initialized


# --- Portfolio Knowledge Base ---
# This multiline string serves as the information source for the chatbot.
# For larger portfolios, consider loading this data from an external file (e.g., .txt, .md, .json)
# for better organization and easier updates without modifying code.
portfolio_knowledge_base = """
NITHIN SHETTY M
shettyn517@gmail.com | +91 77601 52732
https://linkedin.com/in/nithin-shetty-m-530274265 | https://github.com/nithinshettygit
Dakshina Kannada, Karnataka, 574325

EXECUTIVE SUMMARY
AI & ML engineering student with hands-on experience in deep learning, LLM-based agents, and real-time systems.
Skilled in deploying solutions using LangChain, FAISS, and Python. Seeking AI/ML/GenAI Engineer roles to contribute
to impactful, intelligent applications.

EDUCATION
Vivekananda College of Engineering and Technology, Puttur, Dakshina Kannada, Karnataka
B.E. – Artificial Intelligence and Machine Learning, Final Year
Affiliated to Visvesvaraya Technological University, Belagavi, Karnataka
Graduation: Expected June 2026 | CGPA: 8.60 / 10.00 (First 6 Semesters)
PUC (PCMB) – Sri Rama Pre-University College Kalladka , Dakshina Kannada, Karnataka
Year of Completion: 2022 | Percentage: 88.00%
SSLC – Shri Ramachandra High School Perne, Dakshina Kannada, Karnataka
Year of Completion: 2020 | Percentage: 96.36%

EXPERIENCE & PROJECTS
AIRA – AI Powered Smart Teaching Robot  Mar 2025 – Jun 2025
Major Project VCET
 Built RAG & LLM based AI teaching agent enabled real-time Q&A with interruption-resume logic.
 Enhanced response time and answer relevance by integrating FAISS vector search.
 Tools: Langchain, GeminiFlash LLM , Sentence-Transformers, FAISS, Python, React.js, FastAPI

Autonomous Wheelchair using Deep Learning  Jul 2024 – Oct 2024
Mini Project VCET
 Developed CNN-based Deep learning model for real-time direction control of wheelchair prototype.
 Integrated with ESP8266 hardware & Flask UI for dual mode navigation.
 Achieved reliable navigation in a controlled hospital-like environment.
 Tools: Python, PyTorch, OpenCV, Flask, Arduino, NodeMCU

TECHNICAL SKILLS
 Languages: Python, Java
 AI/ML: PyTorch, scikit-learn, OpenCV, NLP, Generative AI
 LLM & GenAI Tools: LangChain, OpenAI API, LLaMA, FAISS, RAG, Sentence-Transformers
 Web & UI: Flask, Streamlit, React.js

SOFT SKILLS
 Communication
 Teamwork
 Problem solving

CERTIFICATIONS
 Udemy: AI & LLM Engineering MasteryGenAI, RAG Complete Guide
"""

# --- Chatbot Logic ---
def get_chatbot_response(user_question: str) -> str:
    """
    Generates a response using the configured Gemini model based on the provided portfolio data.
    The model is instructed to only use the information within the 'portfolio_knowledge_base'.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        str: The AI assistant's response or an error message.
    """
    # Defensive check: Although we stop the script if the key is missing globally,
    # this adds another layer for robustness in case of a very unusual state.
    if not genai.api_key:
        return "Gemini API key is not configured. Please ensure it's set correctly."

    # Construct the prompt for the LLM.
    # The prompt includes detailed instructions, the portfolio data as context,
    # and the user's specific question.
    prompt = f"""
    You are a helpful AI assistant tasked with answering questions about Nithin Shetty M's professional profile.
    Only use the information provided below to answer the questions. If the information is not available,
    state that you don't have enough information.
    Keep your answers concise and directly related to the question.

    Nithin Shetty M's Portfolio Data:
    ---
    {portfolio_knowledge_base}
    ---

    User Question: {user_question}
    AI Assistant:
    """
    try:
        # Generate content using the Gemini model.
        # Stream=True is useful for larger responses, but not strictly necessary for simple Q&A.
        # Setting generation_config and safety_settings can help control output quality and safety.
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,  # Lower temperature makes output more deterministic/factual
                max_output_tokens=200 # Limit response length
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        # Access the generated text. Check if parts exist to handle partial responses.
        return response.text if response.candidates else "Could not generate a response based on the provided information."
    except Exception as e:
        # Provide a more informative error message in case of API communication failure.
        return (f"An error occurred while communicating with the Gemini API. This could be due to "
                f"network issues, an invalid prompt, or a temporary API problem. Error details: {e}")

# --- Streamlit UI Setup ---
st.set_page_config(
    layout="wide", # Use full width of the browser
    page_title="Nithin Shetty M - AI Portfolio & Chatbot",
    # Add an icon for the browser tab
    # page_icon="💡" # You can use an emoji or path to an image file
)

# Custom CSS for a more structured and attractive design
# This CSS is injected directly into the Streamlit app.
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f0f2f6; /* Light gray background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.05); /* Subtle shadow for depth */
    }
    .stApp {
        background-color: #f0f2f6; /* Match main background */
    }

    /* Header styling for the portfolio title */
    .header {
        background-color: #264653; /* Dark blue/green */
        color: white; /* White text */
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Shadow for header */
    }
    .header h1 {
        font-size: 3.2em; /* Larger main title */
        margin-bottom: 8px;
        font-weight: 700; /* Bold font */
    }
    .header p {
        font-size: 1.3em;
        opacity: 0.9; /* Slightly transparent for subheading */
    }

    /* Section titles within the portfolio */
    .section-title {
        color: #264653; /* Match header primary color */
        font-size: 2em; /* Prominent section titles */
        border-bottom: 3px solid #e76f51; /* Coral/orange underline */
        padding-bottom: 10px;
        margin-top: 40px; /* More space above sections */
        margin-bottom: 25px;
        font-weight: 600;
    }

    /* Content blocks for sections like summary, education, etc. */
    .content-block {
        background-color: white; /* White background for content */
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08); /* Soft shadow */
        margin-bottom: 25px;
        line-height: 1.6; /* Improved readability */
    }
    .content-block p, .content-block ul {
        margin-bottom: 10px;
    }
    .content-block ul {
        list-style-type: disc;
        margin-left: 20px;
    }

    /* Styling for individual skill tags */
    .skill-item {
        background-color: #e9c46a; /* Vibrant orange/yellow */
        color: #333; /* Dark text for contrast */
        padding: 7px 12px;
        border-radius: 20px; /* Pill-shaped */
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block; /* Allows items to flow on one line */
        font-size: 0.95em;
        font-weight: 500;
        transition: background-color 0.3s ease; /* Smooth hover effect */
    }
    .skill-item:hover {
        background-color: #f4a261; /* Slightly darker on hover */
    }

    /* Project title specific styling */
    .project-title {
        color: #e76f51; /* Distinct coral/orange for project titles */
        font-size: 1.4em;
        margin-top: 15px; /* Space above each project */
        margin-bottom: 8px;
        font-weight: 600;
    }
    .project-title small {
        color: #666; /* Lighter color for date range */
        font-size: 0.8em;
    }

    /* Contact information styling */
    .contact-info {
        font-size: 1.1em;
        margin-bottom: 15px;
    }
    .contact-info p {
        margin-bottom: 8px;
    }

    /* Social links styling */
    .social-links a {
        color: #264653; /* Match primary color */
        margin-right: 20px;
        text-decoration: none; /* No underline */
        font-weight: bold;
        transition: color 0.3s ease; /* Smooth hover effect */
    }
    .social-links a:hover {
        color: #e9c46a; /* Change color on hover */
    }

    /* Chatbot container styling */
    .chatbot-container {
        background-color: #ffffff; /* White background */
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1); /* More prominent shadow for chatbot */
        margin-top: 40px;
        border: 1px solid #ddd; /* Subtle border */
    }

    /* Individual chat message styling */
    .chat-message {
        background-color: #f8f8f8; /* Light gray for messages */
        padding: 12px 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 5px solid #4a90e2; /* Blue accent for chat messages */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Light shadow for messages */
    }
    </style>
    """, unsafe_allow_html=True)


# --- Portfolio Content Rendering ---
# Main header for the portfolio
st.markdown('<div class="header">', unsafe_allow_html=True)
st.markdown("<h1>NITHIN SHETTY M</h1>", unsafe_allow_html=True)
st.markdown("<p>AI & ML Engineering Student | Generative AI Enthusiast</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Create two columns for a side-by-side layout (e.g., main content vs. sidebar)
col1, col2 = st.columns([2, 1]) # Column 1 takes 2/3 width, Column 2 takes 1/3

with col1: # Content for the wider left column
    st.markdown('<h2 class="section-title">Executive Summary</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content-block">
        AI & ML engineering student with hands-on experience in deep learning, LLM-based agents, and real-time systems.
        Skilled in deploying solutions using LangChain, FAISS, and Python. Seeking AI/ML/GenAI Engineer roles to contribute
        to impactful, intelligent applications.
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Education</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content-block">
        <p><strong>Vivekananda College of Engineering and Technology, Puttur</strong><br>
        B.E. – Artificial Intelligence and Machine Learning, Final Year<br>
        Graduation: Expected June 2026 | CGPA: 8.60 / 10.00 (First 6 Semesters)</p>
        <p><strong>Sri Rama Pre-University College Kalladka</strong><br>
        PUC (PCMB) – Year of Completion: 2022 | Percentage: 88.00%</p>
        <p><strong>Shri Ramachandra High School Perne</strong><br>
        SSLC – Year of Completion: 2020 | Percentage: 96.36%</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Experience & Projects</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content-block">
        <p class="project-title"><strong>AIRA – AI Powered Smart Teaching Robot</strong> <br>
        <small>Mar 2025 – Jun 2025 | Major Project VCET</small></p>
        <ul>
            <li>Built RAG & LLM based AI teaching agent enabled real-time Q&A with interruption-resume logic.</li>
            <li>Enhanced response time and answer relevance by integrating FAISS vector search.</li>
            <li>Tools: Langchain, GeminiFlash LLM , Sentence-Transformers, FAISS, Python, React.js, FastAPI</li>
        </ul>

        <p class="project-title"><strong>Autonomous Wheelchair using Deep Learning</strong> <br>
        <small>Jul 2024 – Oct 2024 | Mini Project VCET</small></p>
        <ul>
            <li>Developed CNN-based Deep learning model for real-time direction control of wheelchair prototype.</li>
            <li>Integrated with ESP8266 hardware & Flask UI for dual mode navigation.</li>
            <li>Achieved reliable navigation in a controlled hospital-like environment.</li>
            <li>Tools: Python, PyTorch, OpenCV, Flask, Arduino, NodeMCU</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

with col2: # Content for the narrower right column (sidebar-like)
    st.markdown('<h2 class="section-title">Contact</h2>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="content-block contact-info">
        <p>📧 shettyn517@gmail.com</p>
        <p>📞 +91 77601 52732</p>
        <p>📍 Dakshina Kannada, Karnataka, 574325</p>
        <div class="social-links">
            <a href="https://linkedin.com/in/nithin-shetty-m-530274265" target="_blank">LinkedIn</a>
            <a href="https://github.com/nithinshettygit" target="_blank">GitHub</a>
        </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Technical Skills</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content-block">
        <p><strong>Languages:</strong> <span class="skill-item">Python</span> <span class="skill-item">Java</span></p>
        <p><strong>AI/ML:</strong> <span class="skill-item">PyTorch</span> <span class="skill-item">scikit-learn</span> <span class="skill-item">OpenCV</span> <span class="skill-item">NLP</span> <span class="skill-item">Generative AI</span></p>
        <p><strong>LLM & GenAI Tools:</strong> <span class="skill-item">LangChain</span> <span class="skill-item">OpenAI API</span> <span class="skill-item">LLaMA</span> <span class="skill-item">FAISS</span> <span class="skill-item">RAG</span> <span class="skill-item">Sentence-Transformers</span></p>
        <p><strong>Web & UI:</strong> <span class="skill-item">Flask</span> <span class="skill-item">Streamlit</span> <span class="skill-item">React.js</span></p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Soft Skills</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content-block">
        <span class="skill-item">Communication</span>
        <span class="skill-item">Teamwork</span>
        <span class="skill-item">Problem Solving</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Certifications</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content-block">
        <ul>
            <li>Udemy: AI & LLM Engineering MasteryGenAI, RAG Complete Guide</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)


# --- Chatbot Section ---
st.markdown('---', unsafe_allow_html=True) # Horizontal line for visual separation
st.markdown('<h2 class="section-title">Ask Nithin\'s Portfolio!</h2>', unsafe_allow_html=True)
st.markdown('<div class="chatbot-container">', unsafe_allow_html=True)
st.write("Hello! I'm a chatbot designed to answer questions about Nithin's professional background. Feel free to ask anything about his skills, projects, education, or experience!")

# Initialize chat history in Streamlit's session state.
# This ensures that the conversation persists across user interactions.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history on app rerun.
# Each message is rendered as a chat bubble based on its role (user/assistant).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input for new questions.
# The ':=` (walrus operator) assigns the input to 'prompt' if it's not empty.
if prompt := st.chat_input("What would you like to know about Nithin?"):
    # Add the user's message to the chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's message in the chat interface.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the chatbot's response.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): # Show a spinner while the LLM generates a response.
            response = get_chatbot_response(prompt)
        st.markdown(response) # Display the assistant's response.
    # Add the assistant's response to the chat history.
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown('</div>', unsafe_allow_html=True)