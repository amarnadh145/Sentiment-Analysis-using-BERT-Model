### Abstract
Sentiment analysis is a natural language processing (NLP) task that involves analysing text
data to determine the sentiment expressed in it. With the increasing use of social media and
online communication, sentiment analysis has become an essential tool for businesses to
understand customer feedback and make data-driven decisions. In recent years, deep learning
models, such as the Bidirectional Encoder Representations from Transformers (BERT) model,
have shown remarkable performance in sentiment analysis tasks. In this project, we propose to
use the BERT model for sentiment analysis on a dataset of amazon reviews of books. We will
fine-tune the pre-trained BERT model on the dataset of amazon reviews of books, which
involves adapting the pre-trained model to the specific domain of customer reviews. We will
evaluate the performance of the BERT model using metrics such as accuracy, precision, recall,
and F1-score. The outcome of this project will be a sentiment analysis model that can accurately
classify the sentiment of amazon reviews, which can be used by businesses to improve their
products and services.
### Requirements
1. Streamlit Framework
2. Transformers
3. Torch
### Build
Run the code of model.txt file in a notebook to build the model which gives you a bin file named 'best_state_model.bin'
after training.
### Execution
Place the files app.py, module.py, img.png and the generated bin file in a editor.
Run the command `streamlit run app.py` in editor, browser will automatically opens and asks for a text.
### Checking for results
Place a text in the textbox and click `Analyze` to get the sentiment of text.

