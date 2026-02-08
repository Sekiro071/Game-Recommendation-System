import re
import pandas as pd
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ------------------------------
# Load data
# ------------------------------
CSV_PATH = r"C:\Users\Shreyansh\ML\Projects\Game Reccomendation\augmented_with_images_alternative_api.csv"
df = pd.read_csv(CSV_PATH, encoding="utf-8")

# Ensure required columns exist
for col in ["name", "genres", "categories", "steamspy_tags", "developer", "header_image_url"]:
    if col not in df.columns:
        df[col] = ""

# Fill NAs
df["genres"] = df["genres"].fillna("")
df["categories"] = df["categories"].fillna("")
df["steamspy_tags"] = df["steamspy_tags"].fillna("")
df["developer"] = df["developer"].fillna("")
df["header_image_url"] = df["header_image_url"].fillna("")

# ------------------------------
# Helpers / Cleaning
# ------------------------------
def clean_developer_name(dev_string):
    if pd.isna(dev_string):
        return ""
    if not isinstance(dev_string, str):
        dev_string = str(dev_string)
    dev_string = dev_string.replace("|", ";").replace(",", ";").replace("/", ";")
    parts = [re.sub(r"\s+", "", p.strip()) for p in dev_string.split(";") if p.strip()]
    return ";".join(parts)

df["developer_tags"] = df["developer"].apply(clean_developer_name)

df["name"] = df["name"].astype(str).str.replace(r"[^a-zA-Z:]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
df["metadata"] = (df["genres"] + ";" + df["categories"] + ";" + df["steamspy_tags"] + ";" + df["developer_tags"])

# ------------------------------
# TF-IDF & similarity
# ------------------------------
def semicolon_analyzer(text):
    return list(set(str(text).lower().split(";")))

tfidf_vectorizer = TfidfVectorizer(analyzer=semicolon_analyzer)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["metadata"])
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

def get_best_match(partial_name: str):
    if not partial_name: return None
    matches = df[df["name"].str.contains(partial_name, case=False, na=False)]
    if matches.empty: return None
    return matches.iloc[0]["name"]

# ------------------------------
# Image resolver
# ------------------------------
def get_image_url(idx: int, name: str) -> str:
    url = df.at[idx, 'header_image_url']
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return url
    return f"https://placehold.co/300x168?text={quote(name)}"

def recommend_game(title: str, top_n: int = 5):
    best_match = get_best_match(title)
    if not best_match:
        return gr.Label(f"No game found for input: '{title}'. Try another title."), ""

    index = df.index[df["name"] == best_match][0]
    sim_scores = cosine_sim[index].toarray().ravel() if hasattr(cosine_sim[index], "toarray") else cosine_sim[index]
    ranked = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    ranked = [(i, s) for i, s in ranked if i != index]

    top_n = max(1, min(int(top_n), len(ranked)))
    top_indices = [i for i, _ in ranked[:top_n]]
    
    recs = []
    for i in top_indices:
        game_name = df["name"].iat[i]
        image_url = get_image_url(i, game_name)
        recs.append({"name": game_name, "image_url": image_url})
    
    # Generate the HTML for the gallery
    gallery_html = ""
    for rec in recs:
        gallery_html += f"""
        <div class="game-card">
            <img src="{rec['image_url']}" alt="{rec['name']}" class="game-image">
            <div class="game-title-overlay">
                <span>{rec['name']}</span>
            </div>
        </div>
        """
    
    new_label_text = f"Showing recommendations for : {best_match}"
    return gr.Label(new_label_text), f'<div class="gallery-container">{gallery_html}</div>'

# ------------------------------
# UI Function
# ------------------------------
initial_games = df.sample(5, replace=False).to_dict('records')
initial_gallery_html = ""
for game in initial_games:
    idx = df[df['name'] == game['name']].index[0]
    initial_gallery_html += f"""
    <div class="game-card">
        <img src="{get_image_url(idx, game['name'])}" alt="{game['name']}" class="game-image">
        <div class="game-title-overlay">
            <span>{game['name']}</span>
        </div>
    </div>
    """

with gr.Blocks(
    theme=gr.themes.Soft(), 
    css="""
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Rajdhani', 'Orbitron', monospace;
        }
        
        footer {visibility: hidden} 
        
        /* Gaming dark background with subtle matrix effect */
        .gradio-container {
            background: 
                radial-gradient(circle at 25% 25%, #0a0a0a 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, #111111 0%, transparent 50%),
                linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
            min-height: 100vh;
            position: relative;
        }
        
        /* Subtle grid overlay for gaming feel */
        .gradio-container::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: -1;
            animation: gridPulse 4s ease-in-out infinite;
        }
        
        @keyframes gridPulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.1; }
        }
        
        /* Main gaming container */
        .main-content {
            background: rgba(13, 17, 23, 0.85);
            border: 2px solid #00ff88;
            border-radius: 12px;
            margin: 20px;
            padding: 40px;
            box-shadow: 
                0 0 30px rgba(0, 255, 136, 0.2),
                inset 0 1px 0 rgba(0, 255, 136, 0.1);
            position: relative;
        }
        
        /* Corner brackets for gaming UI feel */
        .main-content::before,
        .main-content::after {
            content: '';
            position: absolute;
            width: 30px;
            height: 30px;
            border: 3px solid #ff6b35;
        }
        
        .main-content::before {
            top: 15px;
            left: 15px;
            border-right: none;
            border-bottom: none;
        }
        
        .main-content::after {
            bottom: 15px;
            right: 15px;
            border-left: none;
            border-top: none;
        }
        
        /* Gaming title */
        #main_title {
            text-align: center; 
            font-family: 'Orbitron', monospace !important;
            font-size: 4.5rem !important;
            font-weight: 900 !important;
            margin-bottom: 10px !important;
            color: #00ff88 !important;
            position: relative;
        }
        
        #subtitle {
            text-align: center; 
            color: #ff6b35 !important;
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin-bottom: 40px !important;
            letter-spacing: 1px;
        }
        
        /* Gaming input styling */
        .gradio-textbox input {
            background: rgba(22, 27, 34, 0.9) !important;
            border: 2px solid #30363d !important;
            border-radius: 8px !important;
            color: #00ff88 !important;
            font-size: 16px !important;
            font-family: 'Rajdhani', monospace !important;
            font-weight: 600 !important;
            padding: 16px 20px !important;
            transition: all 0.3s ease !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5) !important;
        }
        
        .gradio-textbox input:focus {
            background: rgba(22, 27, 34, 1) !important;
            border-color: #00ff88 !important;
            box-shadow: 
                inset 0 2px 4px rgba(0, 0, 0, 0.5),
                0 0 0 3px rgba(0, 255, 136, 0.1),
                0 0 15px rgba(0, 255, 136, 0.3) !important;
        }
        
        .gradio-textbox input::placeholder {
            color: #7d8590 !important;
            font-style: italic;
        }
        
        .gradio-textbox label {
            color: #f0f6fc !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px !important;
        }
        
        /* Gaming slider styling */
        .gradio-slider {
            background: rgba(22, 27, 34, 0.8) !important;
            border: 2px solid #30363d !important;
            border-radius: 8px !important;
            padding: 20px !important;
        }
        
        .gradio-slider label {
            color: #f0f6fc !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Gaming button with hexagonal inspiration */
        .gradio-button {
            background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%) !important;
            border: 2px solid #ff6b35 !important;
            border-radius: 0 !important;
            color: #000 !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            font-family: 'Orbitron', monospace !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 18px 40px !important;
            position: relative !important;
            overflow: hidden !important;
            clip-path: polygon(10px 0%, 100% 0%, calc(100% - 10px) 100%, 0% 100%) !important;
            box-shadow: 
                0 0 20px rgba(255, 107, 53, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        
        .gradio-button:hover {
            background: linear-gradient(135deg, #ff8c42 0%, #ffad5a 100%) !important;
            box-shadow: 
                0 0 30px rgba(255, 107, 53, 0.6),
                0 5px 15px rgba(0, 0, 0, 0.3) !important;
            transform: translateY(-2px) !important;
        }
        
        .gradio-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.6s;
        }
        
        .gradio-button:hover::before {
            left: 100%;
        }
        
        /* Gaming label styling */
        .gradio-label {
            background: rgba(22, 27, 34, 0.9) !important;
            border: 2px solid #00ff88 !important;
            border-radius: 8px !important;
            padding: 20px !important;
            margin: 20px 0 !important;
            box-shadow: 
                0 0 20px rgba(0, 255, 136, 0.2),
                inset 0 1px 0 rgba(0, 255, 136, 0.1) !important;
        }
        
        .gradio-label .wrap {
            color: #00ff88 !important;
            font-size: 22px !important;
            font-weight: 700 !important;
            font-family: 'Orbitron', monospace !important;
            text-align: center !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Gaming gallery container */
        .gallery-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 25px;
            padding: 30px 0;
            animation: fadeInUp 0.8s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Gaming-style game cards */
        .game-card {
            position: relative;
            width: 280px;
            height: 160px;
            background: rgba(22, 27, 34, 0.9);
            border: 2px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.3s ease;
            animation: cardSlideIn 0.6s ease-out forwards;
            opacity: 0;
            transform: translateX(-50px);
            box-shadow: 
                0 4px 15px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        @keyframes cardSlideIn {
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .game-card:nth-child(1) { animation-delay: 0.1s; }
        .game-card:nth-child(2) { animation-delay: 0.2s; }
        .game-card:nth-child(3) { animation-delay: 0.3s; }
        .game-card:nth-child(4) { animation-delay: 0.4s; }
        .game-card:nth-child(5) { animation-delay: 0.5s; }
        
        .game-card:hover {
            border-color: #ff6b35;
            transform: translateY(-5px) scale(1.03);
            box-shadow: 
                0 8px 25px rgba(0, 0, 0, 0.7),
                0 0 20px rgba(255, 107, 53, 0.3),
                inset 0 1px 0 rgba(255, 107, 53, 0.2);
        }

        /* Cyberpunk corner brackets for cards */
        .game-card::before,
        .game-card::after {
            content: '';
            position: absolute;
            width: 15px;
            height: 15px;
            border: 2px solid #00ff88;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 3;
        }
        
        .game-card::before {
            top: 8px;
            left: 8px;
            border-right: none;
            border-bottom: none;
        }
        
        .game-card::after {
            bottom: 8px;
            right: 8px;
            border-left: none;
            border-top: none;
        }
        
        .game-card:hover::before,
        .game-card:hover::after {
            opacity: 1;
        }

        .game-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: brightness(0.8) contrast(1.1);
            transition: all 0.3s ease;
        }
        
        .game-card:hover .game-image {
            filter: brightness(1) contrast(1.2);
            transform: scale(1.05);
        }

        .game-title-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(to top, rgba(13, 17, 23, 0.95), rgba(13, 17, 23, 0.7), transparent);
            color: #f0f6fc;
            padding: 15px 12px 12px;
            text-align: center;
            font-weight: 700;
            font-size: 15px;
            font-family: 'Rajdhani', monospace;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
            transition: all 0.3s ease;
            border-top: 1px solid rgba(0, 255, 136, 0.2);
        }
        
        .game-card:hover .game-title-overlay {
            background: linear-gradient(to top, rgba(13, 17, 23, 0.95), rgba(255, 107, 53, 0.1), transparent);
            color: #00ff88;
            border-top-color: #ff6b35;
        }
        
        /* Gaming row styling */
        .gradio-row {
            background: rgba(22, 27, 34, 0.6) !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            padding: 20px !important;
            margin: 15px 0 !important;
        }
        
        /* Responsive gaming design */
        @media (max-width: 768px) {
            #main_title {
                font-size: 3rem !important;
            }
            
            #subtitle {
                font-size: 1.2rem !important;
            }
            
            .game-card {
                width: 240px;
                height: 140px;
            }
            
            .main-content {
                margin: 10px;
                padding: 20px;
            }
        }
        
        /* Gaming loading animation */
        @keyframes scanline {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
    """
) as demo:
    with gr.Column(elem_classes="main-content"):
        gr.Markdown(
            """
            <h1 id="main_title">Respawned</h1>
            <h3 id="subtitle">Find. Play. Repeat. | The ultimate stop for game discovery</h3>
            """,
            elem_id="header_section"
        )
        with gr.Row():
            title_in = gr.Textbox(label="Enter a Game Title", placeholder="e.g. Dark Souls III", scale=3)
            k_in = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of recommendations", scale=1)
        go_btn = gr.Button(" Get Recommendations", variant="primary")
        
        # CHANGED LINE: Updated the initial label text
        output_title = gr.Label("Your Next Adventure Awaits") 
        
        gallery_html = gr.HTML(f'<div class="gallery-container">{initial_gallery_html}</div>')
        go_btn.click(fn=recommend_game, inputs=[title_in, k_in], outputs=[output_title, gallery_html])

demo.launch()