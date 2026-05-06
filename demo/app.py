"""
FactEval Gradio Demo – Interactive factuality checker.

Run locally:  python demo/app.py
Run on Colab: Upload facteval/ folder, then run this file.
"""

import json
import gradio as gr
from facteval import analyze, fast_check

EXAMPLES = [
    [
        "Patient presents with acute appendicitis. Given 500mg Amoxicillin. Discharge scheduled for tomorrow.",
        "Patient was diagnosed with acute appendicitis and underwent successful appendectomy. Post-operative care includes IV fluids and rest. No antibiotics were administered. Patient will remain under observation for 48 hours."
    ],
    [
        "Tesla's Q3 revenue reached $25 billion, a 40% year-over-year increase. The company delivered 500,000 vehicles in the quarter.",
        "Tesla reported Q3 revenue of $23.35 billion, representing a 9% year-over-year increase. Vehicle deliveries for the quarter totaled 435,059."
    ],
    [
        "To start a React project, run `npm init react-app my-app` in your terminal. This will install React v17 by default.",
        "To create a new React single-page application, the recommended command is `npx create-react-app my-app`. This installs the latest stable version of React, currently v18."
    ]
]


def run_check(answer: str, contexts: str, calibrator_path: str = ""):
    """Run FactEval pipeline and format results for Gradio."""
    if not answer.strip():
        return "⚠️ Please enter an answer to check.", "", "", ""

    context_list = [c.strip() for c in contexts.strip().split("\n") if c.strip()]
    if not context_list:
        return "⚠️ Please enter at least one context passage.", "", "", ""

    cal_path = calibrator_path.strip() if calibrator_path.strip() else None
    result = analyze(answer, context_list, calibrator_path=cal_path)

    # 1. Highlighted answer (the viral feature)
    highlighted_html = f"""
    <div style="font-family: Inter, sans-serif; font-size: 18px; line-height: 2;
                padding: 20px; border-radius: 12px; background: #0f172a; color: #e2e8f0;">
        {result.get("highlighted_answer", answer)}
    </div>
    """

    # 2. Per-claim verdicts with reasons
    details_parts = []
    for c in result["claims"]:
        label = c["label"]
        colors = {"supported": "#22c55e", "contradicted": "#ef4444", "unverifiable": "#f59e0b"}
        emojis = {"supported": "✅", "contradicted": "❌", "unverifiable": "❓"}
        color = colors.get(label, "#94a3b8")
        emoji = emojis.get(label, "")
        conf = c.get("calibrated_confidence", c["confidence"])

        diag = c.get("diagnostics", {})
        diag_type = diag.get("failure_type", "")
        diag_badge_colors = {
            "verified": "#22c55e", "hallucination": "#ef4444", "possible_hallucination": "#f97316",
            "no_evidence": "#6b7280", "retrieval_gap": "#8b5cf6", "inconclusive": "#f59e0b",
        }
        badge_color = diag_badge_colors.get(diag_type, "#64748b")
        suggestion = diag.get("suggestion", "")

        details_parts.append(f"""
        <div style="padding: 12px; margin: 8px 0; border-left: 4px solid {color};
                    background: {color}10; border-radius: 0 8px 8px 0; font-family: Inter, sans-serif;">
            <div style="font-weight: 600; font-size: 15px; color: #f1f5f9;">
                {emoji} {c["claim"]}
                <span style="font-size: 11px; padding: 2px 8px; border-radius: 12px;
                             background: {badge_color}30; color: {badge_color}; margin-left: 8px;">
                    {diag_type.replace("_", " ")}
                </span>
            </div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">
                {c.get("reason", "")}
            </div>
            {'<div style="font-size: 12px; color: #f59e0b; margin-top: 4px; font-style: italic;">💡 ' + suggestion + '</div>' if suggestion else ''}
            <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                Confidence: {conf:.1%}
                {"• Evidence score: " + f"{c['evidence_score']:.3f}" if c.get("evidence_score") else ""}
                • Retrieval: {diag.get("retrieval_quality", "n/a")}
            </div>
        </div>
        """)

    details_html = '<div>' + ''.join(details_parts) + '</div>'

    # 3. Summary card
    s = result["summary"]
    summary_html = f"""
    <div style="font-family: Inter, sans-serif; padding: 16px; border-radius: 12px;
                background: linear-gradient(135deg, #1e293b, #334155); color: white;">
        <h3 style="margin: 0 0 12px 0; color: #e2e8f0;">📊 Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
            <div style="padding: 8px; background: #ffffff10; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold;">{s['total_claims']}</div>
                <div style="font-size: 12px; color: #94a3b8;">Total Claims</div>
            </div>
            <div style="padding: 8px; background: #22c55e20; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; color: #22c55e;">{s['supported']}</div>
                <div style="font-size: 12px; color: #94a3b8;">Supported</div>
            </div>
            <div style="padding: 8px; background: #ef444420; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; color: #ef4444;">{s['contradicted']}</div>
                <div style="font-size: 12px; color: #94a3b8;">Contradicted</div>
            </div>
            <div style="padding: 8px; background: #f59e0b20; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; color: #f59e0b;">{s['unverifiable']}</div>
                <div style="font-size: 12px; color: #94a3b8;">Unverifiable</div>
            </div>
        </div>
        <div style="margin-top: 12px; padding: 8px; background: #ffffff10; border-radius: 8px; text-align: center;">
            <span style="font-size: 14px; color: #94a3b8;">Hallucination Rate</span><br>
            <span style="font-size: 28px; font-weight: bold;
                         color: {'#22c55e' if s['hallucination_rate'] < 0.3 else '#ef4444'};">
                {s['hallucination_rate']:.0%}
            </span>
        </div>
        <div style="margin-top: 8px; font-size: 11px; color: #64748b; text-align: right;">
            ⏱ {result['pipeline_time_seconds']:.1f}s
            {'• 📐 calibrated' if result.get('calibrated') else '• raw scores'}
        </div>
    </div>
    """

    # 4. Raw JSON
    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    return highlighted_html, details_html, summary_html, json_output


# ── Gradio Interface ─────────────────────────────────────────────────────────

with gr.Blocks(
    title="FactEval – Hallucination Detector",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
        .gradio-container { max-width: 1400px !important; }
        footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🔍 FactEval – Find Exactly Which Parts Are Hallucinated
        Paste an LLM-generated answer and reference contexts.
        FactEval highlights ✅ **supported**, ❌ **contradicted**, and ❓ **unverifiable** claims.
        """
    )

    with gr.Row():
        # LEFT COLUMN: Inputs & Examples
        with gr.Column(scale=1):
            answer_input = gr.Textbox(
                label="LLM Answer",
                placeholder="Enter the text to fact-check...",
                value=EXAMPLES[0][0],
                lines=4,
            )
            context_input = gr.Textbox(
                label="Reference Contexts (one per line)",
                placeholder="Enter ground truth passages, one per line...",
                value=EXAMPLES[0][1],
                lines=5,
            )
            calibrator_input = gr.Textbox(
                label="Calibrator Path (optional)",
                placeholder="Path to calibrator.pkl",
                lines=1,
            )
            check_btn = gr.Button("🔍 Check Factuality", variant="primary", size="lg")
            
            gr.Examples(
                examples=EXAMPLES,
                inputs=[answer_input, context_input],
                label="Try these examples",
            )

        # RIGHT COLUMN: Outputs
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Highlighted Answer")
            highlighted_output = gr.HTML()

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Claim Details")
                    details_output = gr.HTML()
                with gr.Column(scale=1):
                    summary_output = gr.HTML()

            with gr.Accordion("Raw JSON Output", open=False):
                json_output = gr.Code(language="json")

    check_btn.click(
        fn=run_check,
        inputs=[answer_input, context_input, calibrator_input],
        outputs=[highlighted_output, details_output, summary_output, json_output],
    )

    demo.load(
        fn=run_check,
        inputs=[answer_input, context_input, calibrator_input],
        outputs=[highlighted_output, details_output, summary_output, json_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
