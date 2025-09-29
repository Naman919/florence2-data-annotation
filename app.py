import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
import copy
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import tempfile
import xml.etree.ElementTree as ET

models = {
    'microsoft/Florence-2-base': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to("cpu").eval(),
}

processors = {
    'microsoft/Florence-2-base': AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True),
}

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'red','lime', 'indigo', 'yellow', 'aqua','gold', 'tan']

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def run_example(task_prompt, image, text_input=None, model_id='microsoft/Florence-2-base'):
    model = models[model_id]
    processor = processors[model_id]
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    device = "cpu"  # or "cuda" if GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig

def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image

def convert_to_od_format(data):
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    return od_results

def create_xml(data, task, image=None):
    root = ET.Element('annotation')

    bboxes = data.get('bboxes') or data.get('bbox') or []
    labels = data.get('labels') or data.get('bboxes_labels') or []

    if not bboxes or not labels:
        print("Warning: No bounding boxes or labels found. XML will have only <annotation> root.")
    else:
        for bbox, label in zip(bboxes, labels):
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'label').text = str(label)
            bbox_elem = ET.SubElement(obj, 'bbox')
            ET.SubElement(bbox_elem, 'x1').text = str(int(bbox[0]))
            ET.SubElement(bbox_elem, 'y1').text = str(int(bbox[1]))
            ET.SubElement(bbox_elem, 'x2').text = str(int(bbox[2]))
            ET.SubElement(bbox_elem, 'y2').text = str(int(bbox[3]))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xml', prefix=f"{task.replace(' ','_')}_")
    tree = ET.ElementTree(root)
    tree.write(tmp.name, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return tmp.name


def process_image(image, task_prompt, text_input=None, model_id='microsoft/Florence-2-base'):
    image = Image.fromarray(image)
    if task_prompt == 'Caption':
        task_prompt = '<CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None, None

    elif task_prompt == 'Detailed Caption':
        task_prompt = '<DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None, None

    elif task_prompt == 'More Detailed Caption':
        task_prompt = '<MORE_DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None, None

    elif task_prompt == 'Object Detection':
        task_prompt = '<OD>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<OD>'])
        #od_datao = convert_to_od_format(results['<OD>'])
        xml_output = create_xml(results['<OD>'], task_prompt)
        return results, fig_to_pil(fig), xml_output

    elif task_prompt == 'XML Generator':
        task_prompt = '<OD>'
        results = run_example(task_prompt, image, model_id=model_id)
        #od_data = convert_to_od_format(results['<OD>'])
        xml_output = create_xml(results['<OD>'], task_prompt)
        return results, None, xml_output

    elif task_prompt == 'Dense Region Caption':
        task_prompt = '<DENSE_REGION_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<DENSE_REGION_CAPTION>'])
        return results, fig_to_pil(fig), None

    elif task_prompt == 'Caption to Phrase Grounding':
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        xml_output = create_xml(results['<CAPTION_TO_PHRASE_GROUNDING>'], task_prompt)
        return results, fig_to_pil(fig), None

    else:
        return "", None, None

css = """
  .gr-group {
    margin-bottom: 20px;
    padding: 20px;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
"""


single_task_list = [
    'Caption', 'Detailed Caption', 'More Detailed Caption', 'Object Detection',
    'Dense Region Caption', 'Caption to Phrase Grounding', 'XML Generator',
]

def update_task_dropdown(choice):
    return gr.Dropdown(choices=single_task_list, value='Caption')

with gr.Blocks(css=css) as demo:
    with gr.Tab(label="Data Annotation"):

        # Input card
        with gr.Group():
            gr.Markdown("### INPUT")
            input_img = gr.Image(label="Input Picture")
            model_selector = gr.Dropdown(choices=list(models.keys()), label="Model", value='microsoft/Florence-2-base')
            task_type = gr.Radio(choices=['Single task'], label='Task type', value='Single task')
            task_prompt = gr.Dropdown(choices=single_task_list, label="Task Prompt", value="Caption")
            task_type.change(fn=update_task_dropdown, inputs=task_type, outputs=task_prompt)
            submit_btn = gr.Button(value="Submit")

        # Output card
        with gr.Group():
            gr.Markdown("### OUTPUT")
            output_text = gr.Textbox(label="Output Text")
            output_img = gr.Image(label="Output Image")
            download_btn = gr.File(label="Generated XML", type="filepath")

        # Connect button
        submit_btn.click(process_image, [input_img, task_prompt, text_input, model_selector],
                         [output_text, output_img, download_btn])


demo.launch(debug=True)
