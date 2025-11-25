import streamlit as st
import re
import io
import json
from datetime import datetime
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Viral Shorts Title Generator 🚀",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Styling ---
st.markdown("""
<style>
    .stApp { background-color: #1a1a1a; color: #e6e6e6; }
    .stButton>button {
        background-color: #ff4b4b; color: white; border-radius: 8px; border: none;
        padding: 10px 20px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #ff6a6a; }
    .stSelectbox div[data-baseweb="select"] { background-color: #333333; }
    .stTextArea textarea, .stTextInput input { background-color: #333333; color: #e6e6e6; }
    h1, h2, h3 { color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
def parse_srt(file_content: bytes) -> str | None:
    """Parses an SRT file content and extracts only the dialogue."""
    try:
        srt_text = file_content.decode('utf-8', errors='ignore')
        # Remove sequence numbers + timecodes (handles , or . for ms)
        text_no_ts = re.sub(
            r'\d+\s*\n\d{2}:\d{2}:\d{2}[,.]\d{3}\s-->\s\d{2}:\d{2}:\d{2}[,.]\d{3}\s*',
            '', srt_text
        ).strip()
        # Strip HTML tags
        text_no_tags = re.sub(r'<[^>]+>', '', text_no_ts)
        # Remove leftover empty lines and join
        dialogue = " ".join([ln.strip() for ln in text_no_tags.splitlines() if ln.strip()])
        return dialogue
    except Exception as e:
        st.error(f"Error parsing SRT file: {e}")
        return None

def parse_json_from_response(text: str) -> list:
    """Extracts and parses a JSON array from a string, handling markdown code fences."""
    match = re.search(r'```json\s*([\s\S]+?)\s*```', text)
    if match:
        json_str = match.group(1)
    else:
        # Assume the whole string is the JSON array if no fences are found
        json_str = text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        st.error("AI response was not valid JSON. Could not parse tones.")
        return []


# --- Prompt Engineering Functions ---

def get_tone_prompt(srt_raw_text: str) -> str:
    """Builds the prompt for extracting tones from a transcript."""
    # This prompt remains the same
    return f"""
ROLE
You are a YouTube Shorts tone extractor. Read an SRT, understand its meaning and audience, then output ONLY the tone names. Do NOT generate titles or any other fields.

PLAIN-ENGLISH NAMING (must follow)
- Each tone name is 1–3 simple words a 12-year-old can understand.
- Keep nuance but stay clear (e.g., "Myth Busting", "Calm Guide", "Soft Warning", "Process Review", "Women First").
- No slashes, emojis, colons, or jargon. Use spaces; Title Case preferred.

EXHAUSTIVE MODE (very important)
- Return ALL distinct tones that the SRT clearly supports (no arbitrary cap).
- Dedupe near-duplicates; keep the clearest name.
- Only include tones you can justify from SRT evidence; do not invent without cues.

INPUTS
- srt_raw: raw SRT text with timestamps.
- config (optional):
  - max_tones: "max" (default) for exhaustive; or an integer to hard-cap.
  - language: "en" by default
  - audience_hint (optional)

AVAILABLE_STRATEGIES (for internal reasoning only; DO NOT output)
["Punchline / Reveal","Controversial Opinion","Clear Outcome / Result","Problem Statement",
 "Contradiction / Irony","Curiosity Hook","Secret / Hidden Strategy","Urgency / FOMO",
 "List or Framework","Transformation / Before-After","Emotional Trigger","Direct Question",
 "Surprising / Unexpected","Motivational","Nostalgic / Sentimental","Aspirational / Luxurious",
 "Intriguing / Mysterious","Urgent / Timely"]

METHOD (deterministic)
1) CLEAN & READ: Strip timestamps for analysis; keep order. Lowercase for matching; preserve entities and groups.
2) CONTEXT: Internally summarize what the clip says, for whom, and the stance (claim/challenge/neutral).
3) SIGNALS → TONES: Mine stance/emotion cues (claims, myth-challenge, uncertainty markers like "still", identity labels like "women", cadence words like "quarterly", advice verbs like "understand/explain").
4) NAME: Propose tones that best capture meaning and emotion. If needed, create new simple names.
5) SELECTION:
   - If config.max_tones = "max": include all justified tones with confidence ≥ 0.20, deduped and sorted by importance.
   - If an integer is provided: include up to that many highest-confidence tones after deduping.

QUALITY GUARDS
- Use plain words but keep essence. Be identity-sensitive. No outside facts.
- If SRT is very short and evidence is weak, return fewer tones.
- Absolutely no extra text or keys beyond the JSON array.

OUTPUT (STRICT JSON ONLY)
- Print ONLY a JSON array of strings (tone names). No wrapper object, no comments, no trailing text.
- Example format (for shape only; do not print this example literally):
  ["Educational","Myth Busting","Calm Guide"]

NOW DO THE TASK
SRT:
{srt_raw_text}

config:
{{ "max_tones": "max", "language": "en" }}
"""

def get_header_prompt(transcript_text: str, chosen_tone: str, header_count: int, custom_angle: str) -> str:
    """Builds the prompt for generating headers based on a chosen tone."""
    # This prompt remains the same
    angle_block = ""
    if custom_angle and custom_angle.strip():
        angle_block = f"""
# [CUSTOM ANGLE — STRICT]
All generated outputs MUST align with this angle/domain:
\"\"\"{custom_angle.strip()}\"\"\"
Stay tightly on-theme.
"""

    return f"""
[ROLE & EXPERTISE]
You are a top-tier viral copywriter and social media strategist. You specialize in creating high-performing hooks for YouTube Shorts, Instagram Reels, and TikToks. You understand retention psychology and what works for the Indian audience as of October 2025.

[PRIMARY TASK]
Analyze the provided transcript and generate a list of viral Headers for on-screen text and thumbnails. The language must be simple, emotionally engaging, and curiosity-inducing.

[INPUT TRANSCRIPT]
---
{transcript_text}
---

[GENERATION GUIDELINES & CONSTRAINTS]
## PRIMARY TONE & STYLE
**Chosen Tone:** {chosen_tone}

**Instruction:** All generated headers MUST strictly adhere to the Chosen Tone specified above. Use one of the following options for the placeholder:
- **Shocking/Intriguing:** Create a sense of disbelief or an urge to know more.
- **Knowledge-Based:** Frame as a secret, hack, or little-known fact.
- **Aspirational:** Connect with the audience's desires and future goals.
- **Reverse-Psychology:** Use a challenge or a "don't do this" approach.
- **Relatable Emotion:** Tap into a common, shared feeling or struggle.

{angle_block}

## GUIDING PRINCIPLES (You MUST follow these):
- **BE HYPER-SPECIFIC:** Incorporate specific names, numbers, or unique concepts from the transcript.
- **FOCUS ON TRANSFORMATION & OUTCOME:** Frame headers around a clear "before & after" or a tangible result.
- **LEVERAGE AUTHORITY/PERSONALITY:** If a specific person is mentioned, use their name.
- **THINK VISUALLY:** Imagine the text on a thumbnail.

## HEADER FORMATTING RULES:
- **Length:** STRICTLY 3–5 words.
- **Form:** Punchy phrase (not a full sentence).
- **Emojis:** Include 1–2 strong emojis (e.g., 🤫, 🤯, 🚨, 💰, 🚩).
- **Examples:** “The 12-Hour Lie 🤯”, “Their Secret Pay Trick 🤫”, “Stop Chasing Happiness 🚩”.

[OUTPUT FORMAT — NO EXTRA TEXT]
Respond ONLY with this section and nothing else. Generate EXACTLY {header_count} headers.

**Headers (3–5 words max)**
- [Header 1]
- [Header 2]
...
- [Header {header_count}]
"""

def get_title_prompt(transcript_text: str, chosen_tone: str, title_count: int) -> str:
    """Builds the prompt for generating titles based on a chosen tone."""
    # This prompt remains the same
    return f"""
ROLE AND GOAL:
You are an expert viral content strategist based in Noida, specializing in writing high-engagement, "scroll-stopping" titles for YouTube Shorts. Your goal is to generate {title_count} powerful titles based on the provided video context and strategic parameters.

CONTEXT OF THE VIDEO:
---
{transcript_text}
---

STRATEGIC PARAMETERS:
- **Desired Tone/Style:** {chosen_tone}

TITLE GENERATION STRATEGIES TO USE:
1. **Punchline / Reveal:** Drop a surprising or bold fact early (e.g., “50% of My Income Comes from Social Media?!”)
2. **Controversial Opinion:** Spark debate or strong reactions (e.g., “Freelancing Is Dead – Here's Why”)
3. **Clear Outcome / Result:** Show tangible success or transformation (e.g., “How I Made ₹10L in 6 Months Freelancing”)
4. **Problem Statement:** Call out a relatable pain point (e.g., “Struggling to Get Clients? Watch This.”)
5. **Contradiction / Irony:** Challenge common assumptions (e.g., “Clients Pay Less Than My Instagram Posts Do”)
6. **Curiosity Hook:** Create an information gap people want to close (e.g., “I Did THIS Before Every Big Client Deal”)
7. **Secret / Hidden Strategy:** Tease insider tips or unknown hacks (e.g., “The Tool No Freelancer Talks About”)
8. **Urgency / FOMO:** Build pressure to act now or miss out (e.g., “Do This Before It’s Too Late!”)
9. **List or Framework:** Use structure like steps, tips, or tools (e.g., “3 Steps to Build a High-Income Side Hustle”)
10. **Transformation / Before-After:** Show clear change over time or effort (e.g., “From ₹0 to ₹1L/Month in 90 Days”)
11. **Emotional Trigger:** Use words that evoke strong feelings (e.g., “My Biggest Failure”)
12. **Direct Question:** Ask a question the audience wants answered (e.g., “Is This The Future?”)
13. **Surprising/Unexpected:** Surprise the audience with a surprising fact or statement (e.g., “I’m a Mentalist”)
14. **Motivational:** Motivate the audience to take action (e.g., “Don’t Let Fear Hold You Back”)
15. **Nostalgic/Sentimental:** Evoke nostalgia or sentimentality (e.g., “The Best Advice I Ever Got”)
16. **Aspirational / Luxurious:** Inspire the audience to aspire to something (e.g., “The Best Way to Make Money”)
17. **Intriguing/Mysterious:** Intrigue the audience with a mysterious or intriguing statement (e.g., “The Secret to Success”)
18. **Urgent/Timely:** Create a sense of urgency or timeliness (e.g., “Do This Before It’s Too Late!”)

INSTRUCTIONS:
Your final output must be ONLY a Markdown table with two columns: "Strategy" and "Suggested Title". Do not include any other text, explanation, or introduction. Generate exactly {title_count} titles.
"""

# --- Sidebar: Config & Model ---
with st.sidebar:
    st.header("⚙️ Configuration")

    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception:
        st.error("`secrets.toml` missing or `GOOGLE_API_KEY` not set.")
        st.info("Create `.streamlit/secrets.toml` and add `GOOGLE_API_KEY = \"...\"`")
        st.stop()

    # --- CORRECTED MODEL SELECTION (NEW) ---
    st.subheader("🤖 AI Model")
    
    # Create a mapping from user-friendly names to the required API model IDs
    model_map = {
        "Gemini 2.5 Pro": "gemini-2.5-pro", # Hypothetical ID, assuming standard naming
        "Gemini 2.5 Flash": "gemini-2.5-flash", # Hypothetical ID
        "Gemini Pro (Latest)": "gemini-pro" # 'gemini-pro' is the identifier for the latest stable version
    }
    
    # The options shown to the user are the keys of the dictionary
    display_names = list(model_map.keys())
    
    # Set "Gemini 2.5 Pro" as the default
    default_index = display_names.index("Gemini 2.5 Pro")

    # Let the user select the display name
    selected_display_name = st.selectbox(
        "Choose AI Model",
        display_names,
        index=default_index,
        help="Select the AI model for content generation."
    )
    
    # Get the actual model ID from the map to use in the API call
    actual_model_id = model_map[selected_display_name]
    
    model = genai.GenerativeModel(actual_model_id)
    # --- END OF CORRECTED MODEL SELECTION ---


    st.markdown("---")
    st.subheader("🧪 Output Settings")
    colA, colB = st.columns(2)
    with colA:
        header_count = st.slider("Headers to generate", min_value=5, max_value=30, value=15, step=1)
    with colB:
        title_count = st.slider("Titles to generate", min_value=5, max_value=30, value=10, step=1)

    st.markdown("---")
    st.subheader("🎯 Custom Angle (Optional)")
    use_custom_angle = st.checkbox("Constrain headers to a custom angle/domain")
    custom_angle = ""
    if use_custom_angle:
        custom_angle = st.text_area(
            "Describe your angle",
            placeholder="e.g., Contrarian advice for first-time founders",
            height=100
        )

    st.markdown("---")
    st.caption("Powered by Google Gemini")


# --- Main Content Area ---
st.title("🎬 Viral Shorts Title & Headline Generator")
st.markdown("A multi-step tool to find the perfect angle for your content. Provide a transcript, analyze its tones, select one, and generate!")

# Initialize session state variables
if "transcript_input" not in st.session_state:
    st.session_state.transcript_input = ""
if "generated_tones" not in st.session_state:
    st.session_state.generated_tones = []
if "selected_tone" not in st.session_state:
    st.session_state.selected_tone = None
if "last_result_md" not in st.session_state:
    st.session_state.last_result_md = ""
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = ""


# --- Step 1: Input Transcript ---
st.header("1) Provide Your Video Transcript")
input_method = st.radio(
    "Choose input method:",
    ("Paste Text", "Upload .txt File", "Upload .srt File"),
    horizontal=True,
    key="input_method_radio"
)

transcript_input_area = ""
if input_method == "Paste Text":
    transcript_input_area = st.text_area(
        "Paste your full transcript here:",
        height=200,
        placeholder="When I was doing TV, we used to get 4 days for a 12 hour shift..."
    )
elif input_method == "Upload .txt File":
    uploaded_txt = st.file_uploader("Upload a .txt file", type=['txt'])
    if uploaded_txt:
        transcript_input_area = uploaded_txt.read().decode('utf-8', errors='ignore')
        st.success("✅ TXT file uploaded and processed!")
elif input_method == "Upload .srt File":
    uploaded_srt = st.file_uploader("Upload a .srt file", type=['srt'])
    if uploaded_srt:
        parsed = parse_srt(uploaded_srt.getvalue())
        if parsed:
            transcript_input_area = parsed
            st.success("✅ SRT file uploaded and dialogue extracted!")
        else:
            st.error("Could not parse SRT. Please check the file formatting.")

# Update session state if input changes
if transcript_input_area and st.session_state.transcript_input != transcript_input_area:
    st.session_state.transcript_input = transcript_input_area
    # Reset downstream states if input changes
    st.session_state.generated_tones = []
    st.session_state.selected_tone = None
    st.session_state.last_result_md = ""


# --- Step 2: Analyze Tone ---
if st.session_state.transcript_input:
    st.header("2) Analyze Transcript for Tones")
    if st.button("🔍 Analyze Tones"):
        with st.spinner("Analyzing tones from transcript..."):
            try:
                prompt = get_tone_prompt(st.session_state.transcript_input)
                response = model.generate_content(prompt)
                tones = parse_json_from_response(response.text)
                st.session_state.generated_tones = tones
                if not tones:
                    st.warning("Analysis complete, but no distinct tones were found. The transcript might be too short or generic.")
            except Exception as e:
                st.error(f"An error occurred during tone analysis: {e}")


# --- Step 3: Select Tone ---
if st.session_state.generated_tones:
    st.header("3) Select a Tone")
    st.session_state.selected_tone = st.radio(
        "Choose the tone that best fits your video:",
        st.session_state.generated_tones,
        horizontal=True
    )

# --- Step 4: Generate Content ---
if st.session_state.selected_tone:
    st.header("4) Generate Your Content")
    if st.button("🚀 Generate Titles & Headlines"):
        with st.spinner("🧠 Crafting the perfect hooks..."):
            try:
                # Generate Headers
                header_prompt = get_header_prompt(
                    transcript_text=st.session_state.transcript_input,
                    chosen_tone=st.session_state.selected_tone,
                    header_count=header_count,
                    custom_angle=(custom_angle.strip() if use_custom_angle else "")
                )
                header_response = model.generate_content(header_prompt)

                # Generate Titles
                title_prompt = get_title_prompt(
                    transcript_text=st.session_state.transcript_input,
                    chosen_tone=st.session_state.selected_tone,
                    title_count=title_count
                )
                title_response = model.generate_content(title_prompt)

                # Combine results
                final_md = f"## Results for Tone: *{st.session_state.selected_tone}*\n\n"
                final_md += f"{header_response.text.strip()}\n\n"
                final_md += f"## Titles (under 10 words)\n\n"
                final_md += f"{title_response.text.strip()}"

                st.session_state["last_result_md"] = final_md
                st.session_state["last_run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")

# --- Show Results ---
if st.session_state["last_result_md"]:
    st.header("✅ Generated Results")
    st.markdown(st.session_state["last_result_md"])

    # Prepare downloadable file
    file_buf = io.StringIO()
    file_buf.write(f"# Viral Shorts Output\n")
    file_buf.write(f"Generated at: {st.session_state['last_run_time']}\n\n")
    file_buf.write(st.session_state["last_result_md"])
    file_str = file_buf.getvalue().encode("utf-8")

    st.download_button(
        label="⬇️ Download as .md",
        data=file_str,
        file_name=f"viral_shorts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )
