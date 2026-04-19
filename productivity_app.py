import gradio as gr
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("model.pkl")

QUARTERS    = ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
DEPARTMENTS = ["finishing", "sweing"]
DAYS        = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:     #00130d;
    --card:   #051a10;
    --input:  #0d2b1a;
    --border: #1a4a30;
    --accent: #34d399;
    --grad1:  #10b981;
    --grad2:  #06b6d4;
    --text:   #6ba88a;
    --heading: white;
}

body.light-mode {
    --bg:     #f0fdf4 !important;
    --card:   #ffffff !important;
    --input:  #dcfce7 !important;
    --border: #86efac !important;
    --accent: #059669 !important;
    --grad1:  #10b981 !important;
    --grad2:  #0891b2 !important;
    --text:   #065f46 !important;
    --heading: #064e3b !important;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    transition: background 0.4s ease !important;
}
.gradio-container {
    max-width: 100vw !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 24px !important;
}
.main, .wrap, .contain { max-width: 100% !important; width: 100% !important; }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px; letter-spacing: 3px;
    color: var(--accent); text-transform: uppercase;
    margin-bottom: 12px !important;
    padding: 0 !important; background: none !important; border: none !important;
}
label { color: var(--text) !important; font-size: 13px !important; font-weight: 500 !important; }
input[type=number], input, textarea {
    background: var(--input) !important;
    border: 1px solid var(--border) !important;
    color: var(--heading) !important; border-radius: 8px !important;
    transition: all 0.4s ease !important;
}
.gr-box, .gr-panel, .gr-form {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    transition: all 0.4s ease !important;
}
select {
    background: var(--input) !important;
    border: 1px solid var(--border) !important;
    color: var(--heading) !important; border-radius: 8px !important;
}
button[class*="primary"] {
    background: linear-gradient(135deg, var(--grad1), var(--grad2)) !important;
    border: none !important; color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; font-size: 14px !important;
    letter-spacing: 1px !important; border-radius: 10px !important;
    padding: 14px 32px !important;
}
button[class*="primary"]:hover { opacity: 0.85 !important; }
button[class*="secondary"] {
    background: var(--input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important;
}

/* Toggle button styling */
#toggle-btn button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 999px !important;
    font-size: 11px !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 1px !important;
    padding: 5px 14px !important;
    min-width: unset !important;
    width: auto !important;
    transition: all 0.3s ease !important;
    opacity: 0.7 !important;
}
#toggle-btn button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    opacity: 1 !important;
}

footer { display: none !important; }
"""

# This JS runs on page load via gr.Blocks(js=...)
HEAD_JS = """
function() {
    // Restore saved theme on page load
    const saved = localStorage.getItem('prod_theme');
    if (saved === 'light') {
        document.body.classList.add('light-mode');
    }
    return [];
}
"""

# This JS runs when toggle button is clicked
TOGGLE_JS = """
function() {
    const isLight = document.body.classList.toggle('light-mode');
    localStorage.setItem('prod_theme', isLight ? 'light' : 'dark');
    return [isLight ? '🌙 Dark Mode' : '☀️ Light Mode'];
}
"""

def predict_productivity(
    team, targeted_productivity, smv, wip, over_time,
    incentive, idle_time, idle_men, no_of_style_change,
    no_of_workers, quarter, department, day
):
    q2 = 1 if quarter == "Quarter2" else 0
    q3 = 1 if quarter == "Quarter3" else 0
    q4 = 1 if quarter == "Quarter4" else 0
    q5 = 1 if quarter == "Quarter5" else 0
    dept_finishing = 1 if department == "finishing" else 0
    dept_sweing    = 1 if department == "sweing"    else 0
    day_saturday  = 1 if day == "Saturday"  else 0
    day_sunday    = 1 if day == "Sunday"    else 0
    day_thursday  = 1 if day == "Thursday"  else 0
    day_tuesday   = 1 if day == "Tuesday"   else 0
    day_wednesday = 1 if day == "Wednesday" else 0

    features = np.array([[
        team, targeted_productivity, smv, wip, over_time,
        incentive, idle_time, idle_men, no_of_style_change, no_of_workers,
        q2, q3, q4, q5, dept_finishing, dept_sweing,
        day_saturday, day_sunday, day_thursday, day_tuesday, day_wednesday
    ]])

    prediction = float(np.clip(model.predict(features)[0], 0, 1))

    if prediction >= 0.75:
        label, color, tip = "🟢 High Performer",     "#22c55e", "Excellent output! This team is crushing targets."
    elif prediction >= 0.50:
        label, color, tip = "🟡 Moderate Performer", "#f59e0b", "Decent productivity. Some optimization headroom remains."
    else:
        label, color, tip = "🔴 Needs Attention",    "#ef4444", "Below average. Consider reviewing workload or incentives."

    return f"""
    <div style="font-family:'Inter',sans-serif;
      background:linear-gradient(135deg,var(--card),var(--bg));
      border-radius:18px; padding:36px; text-align:center;
      border:1px solid {color}55;
      box-shadow:0 0 50px {color}22, 0 0 0 1px var(--border);
      margin-top:8px; transition:all 0.4s ease;">
      <div style="font-size:13px;color:var(--text);letter-spacing:4px;text-transform:uppercase;margin-bottom:14px;">
        Predicted Productivity
      </div>
      <div style="font-size:72px;font-weight:800;line-height:1;margin-bottom:10px;
        background:linear-gradient(135deg,var(--grad1),var(--grad2));
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        {prediction:.2%}
      </div>
      <div style="font-size:22px;color:var(--heading);margin-bottom:18px;font-weight:600;">{label}</div>
      <div style="background:{color}15;border:1px solid {color}44;border-radius:12px;
        padding:14px 22px;color:var(--text);font-size:14px;margin-bottom:22px;">{tip}</div>
      <div style="background:var(--input);border-radius:12px;height:12px;
        overflow:hidden;border:1px solid var(--border);">
        <div style="width:{prediction*100:.1f}%;height:100%;
          background:linear-gradient(90deg,var(--grad1),var(--grad2));border-radius:12px;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;color:var(--text);font-size:11px;margin-top:5px;">
        <span>0%</span><span>50%</span><span>100%</span>
      </div>
    </div>
    """


with gr.Blocks(css=CSS, title="Employee Productivity Predictor", js=HEAD_JS) as demo:

    # ── Header with toggle tucked top-right ──────────────────────────────────
    gr.HTML("""
    <div style="
        position:relative; text-align:center; padding:40px 20px 24px;
        background:linear-gradient(135deg,#10b98122,#06b6d422);
        border-bottom:1px solid var(--border);
        margin-bottom:20px; border-radius:0 0 20px 20px;">
        <h1 style="
            font-family:'Space Mono',monospace;
            font-size:2.6rem; font-weight:700;
            background:linear-gradient(135deg,#10b981,#06b6d4);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text; margin:0; letter-spacing:-1px;">
            ⚡ Productivity Predictor
        </h1>
        <p style="color:var(--text); margin:10px 0 0; font-size:15px;">
            Decision Tree model · Garment Manufacturing Analytics
        </p>
    </div>
    """)

    # ── Toggle — small, top-right aligned ────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=8): pass
        with gr.Column(scale=1, min_width=130):
            toggle_btn = gr.Button("☀️ Light", elem_id="toggle-btn", variant="secondary", size="sm")

    toggle_btn.click(fn=None, inputs=[], outputs=[toggle_btn], js=TOGGLE_JS)

    gr.HTML("<div style='margin-bottom:16px'></div>")

    # ── Inputs ────────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<p class="section-label">👥 Team Info</p>')
            team               = gr.Number(label="Team Number",          value=0,    precision=0)
            no_of_workers      = gr.Number(label="No. of Workers",       value=0)
            no_of_style_change = gr.Number(label="No. of Style Changes", value=0,    precision=0)

            gr.HTML('<p class="section-label">🎯 Targets & Time</p>')
            targeted_productivity = gr.Slider(0.0, 1.0, value=0, step=0.01,
                                              label="Targeted Productivity (0–1)")
            over_time = gr.Number(label="Overtime (mins)",  value=0)
            idle_time = gr.Number(label="Idle Time (mins)", value=0.0)
            idle_men  = gr.Number(label="Idle Men Count",   value=0, precision=0)

        with gr.Column(scale=1):
            gr.HTML('<p class="section-label">🏭 Work Details</p>')
            smv       = gr.Number(label="Standard Minute Value (SMV)", value=0)
            wip       = gr.Number(label="Work-In-Progress (WIP)",      value=0)
            incentive = gr.Number(label="Incentive (BDT)",             value=0, precision=0)

            gr.HTML('<p class="section-label">📅 Schedule</p>')
            quarter    = gr.Dropdown(QUARTERS,    label="Quarter"   )
            department = gr.Dropdown(DEPARTMENTS, label="Department")
            day        = gr.Dropdown(DAYS,        label="Day of Week")

    with gr.Row():
        clear_btn  = gr.ClearButton(variant="secondary", value="🔄 Reset")
        submit_btn = gr.Button("🔍 Predict Productivity", variant="primary")

    output = gr.HTML()

    submit_btn.click(
        fn=predict_productivity,
        inputs=[team, targeted_productivity, smv, wip, over_time,
                incentive, idle_time, idle_men, no_of_style_change,
                no_of_workers, quarter, department, day],
        outputs=output
    )
    clear_btn.add([team, targeted_productivity, smv, wip, over_time,
                   incentive, idle_time, idle_men, no_of_style_change,
                   no_of_workers, quarter, department, day, output])

    gr.HTML("""
    <div style="text-align:center;color:var(--border);font-size:12px;padding:20px 0 8px;font-family:monospace;">
        Built with Gradio · Decision Tree Regressor · sklearn
    </div>
    """)

if __name__ == "__main__":
    demo.launch()