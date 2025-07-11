Transformers-Deep-Learning
A curated collection of hands-on Jupyter notebooks demonstrating state-of-the-art Transformer architectures for Natural Language Processing and Computer Vision tasks.
Transformers have revolutionized deep learning by introducing self-attention mechanisms that excel at modeling long-range dependencies in data. This repository provides end-to-end examples:

Vision Transformers: Understanding and implementing self-attention in image patches.

BERT Fine-Tuning: Applying BERT to Named Entity Recognition tasks.

Question Answering: Building an extractive QA system using pretrained Transformers.

PyTorch Implementations: Practical code examples with PyTorch and Hugging Face’s transformers library.

🧰Prerequisites

Python 3.7 or higher
Jupyter Notebook
Git
For GPU acceleration:
NVIDIA GPU with CUDA support
CUDA Toolkit and cuDNN installed

⚙️ Installation
Clone the repository
git clone https://github.com/VimlenduSharma/Transformers-Deep-Learning-.git
cd Transformers-Deep-Learning-
Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
pip install --upgrade pip
pip install transformers torch torchvision datasets seqeval jupyter

📊 Results
Each notebook contains visualizations and performance metrics. You’ll see:
Attention heatmaps over image patches (Vision Transformers)
Entity-level precision/recall for NER tasks
Exact match (EM) and F1 scores for question answering
Feel free to modify hyperparameters, swap datasets, or integrate with your own data!

📚 References
Vaswani et al., "Attention Is All You Need"
Dosovitskiy et al., "An Image is Worth 16x16 Words"
Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
PyTorch Documentation: https://pytorch.org/docs/stable/index.html

Made with ❤️ by Vimlendu Sharma



