import gradio as gr
import torch
import re
import unicodedata
import math
import pickle
import os
# from tankri_inference_code import TransformerSeq2Seq, translate  # name of the file above (without .py)
import tankri_inference_code as tankri
translate = tankri.translate

# Example: from transformer_infer import translate

# Optional: Title and description for UI
TITLE = "English ↔ Tankri Translator"
DESCRIPTION = """

Enter a sentence and get the translation instantly.
"""

# Build Gradio interface
iface = gr.Interface(
    fn=translate,  # directly use your translate() function
    inputs=gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text"),
    outputs=gr.Textbox(label="Translated Text"),
    title=TITLE,
    description=DESCRIPTION,
    theme="default",
    examples=[
        ["How are you?"],
        ["This is a beautiful script."]
    ]
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)

# demo.launch(server_name="127.0.0.1", server_port=7860)
