# Florence-2 Data Annotation 🖼️

An interactive **Gradio web app** built using Microsoft’s [Florence-2](https://huggingface.co/microsoft/Florence-2-base) model for **image captioning, object detection, phrase grounding, and XML export**.

---

## 🚀 Features

* **Image Captioning** (`<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`)
* **Object Detection** with bounding boxes
* **Dense Region Captioning**
* **Caption-to-Phrase Grounding**
* **XML export** for detected objects

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/Naman919/florence2-data-annotation.git
cd florence2-data-annotation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open the Gradio link in your browser.

---

## 🛠️ Tech Stack

| Technology               | Role                               |
| ------------------------ | ---------------------------------- |
| Gradio                   | Web App Interface                  |
| HuggingFace Transformers | Florence-2 Model Inference         |
| PyTorch                  | ML Backend                         |
| PIL / Matplotlib         | Image Processing & Visualization   |
| XML (ElementTree)        | Generating annotation/export files |
| Python                   | Core Development                   |

---

## 📂 Project Structure

```
/florence2-data-annotation
├── app.py              # Main Gradio app
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
├── .gitignore           # Ignore cache/temp files
```

---

## 📸 Example Outputs

**1. Image Captioning**

*Input*:  <img width="1530" height="604" alt="input" src="https://github.com/user-attachments/assets/ecee38a9-b8cc-4501-a93c-910ba491f52c" />

*Output*: `"A man holding a dog in his arms"`

**2. Object Detection**
*Bounding boxes & labels drawn on the image*

<img width="620" height="420" alt="input" src="https://github.com/user-attachments/assets/3932b70b-7db5-4665-b23f-4fbff49bf6e9" />


**3. XML Export**

```xml
<annotation>
<object>
<label>dog</label>
<bbox>
<x1>228</x1>
<y1>188</y1>
<x2>417</x2>
<y2>388</y2>
</bbox>
</object>
<object>
<label>man</label>
<bbox>
<x1>31</x1>
<y1>9</y1>
<x2>453</x2>
<y2>413</y2>
</bbox>
</object>
</annotation>
```
---

## 🔮 Future Improvements

* Add support for more Florence-2 model variants
* Export to **COCO JSON** and **YOLO TXT** formats
* Batch image processing
* Better error handling 
---
