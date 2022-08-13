from itertools import islice, cycle

import docx2txt as docx2txt
import numpy as np
import streamlit as st
import torch
from annotated_text import annotated_text
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer

# Define label, pretrained model and device used
# We prefer to use torch.cuda however if cpu is also an option

ner_tags = {0: 'O',
            1: 'B-degree',
            2: 'I-degree',
            3: 'B-organization',
            4: 'I-organization',
            5: 'B-skill',
            6: 'B-name',
            7: 'I-name',
            8: 'B-job_position',
            9: 'I-job_position',
            10: 'B-email',
            11: 'I-skill',
            12: 'B-phone',
            13: 'I-phone', }

pre_model_name_ner = 'xlm-roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function: Load saved model
def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')

    model = model.to(device)
    model.eval()
    return model


# Function: Load tokenizer from the pretrained model - xlm-roberta-based
def load_tokenizer_ner(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


# Function: make the predictions and print all highlight text
# The highlight text will have label and color viewed along.
# Limit from the pre-trained-model: <= 514 tokens
def make_predictions(text, model, model_name):
    tokenizer = load_tokenizer_ner(model_name)
    tokenized_sentences = tokenizer.encode_plus(text, return_tensors='pt')
    input_ids = tokenized_sentences

    with torch.no_grad():
        # focus on our tensor output
        output = model(**input_ids)["logits"]

    labels = np.argmax(output[0].to('cpu').numpy(), axis=1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids["input_ids"].to('cpu').numpy()[0])  #

    # define the token and label dict to use
    new_tokens, new_labels = [], []

    # Provide value inside the token and label dict
    for token, label in zip(tokens, labels):
        if token.startswith('##'):
            new_tokens[-1] = new_tokens[-1] + token[2:]

        else:
            new_labels.append(ner_tags.get(label))
            new_tokens.append(token)

    # Color cycle option used for the annotated_text
    # Some variable for our field will also be defined as well
    color = list(islice(cycle(['#8ef', '#faa', '#afa', '#fea', '#8ef', '#afa']), len(new_tokens)))
    result = []
    name = ''
    email = ''
    phone = ''

    for token, label, color in zip(new_tokens, new_labels, color):
        # View the space of our text
        # Indentify the labelled text and display with labels and color
        token = token.replace('▁', ' ')

        if token == '[CLS]' or token == '[SEP]':
            continue
        if label == 'O':
            label = ' '
            result.append(token.lower() + ' ')

        else:
            if label == 'B-name' or label == 'I-name':

                name += token.upper()

            elif label == 'B-email':

                email += token.upper()

            elif label == 'B-phone' or label == 'I-phone':
                phone += token.upper()
            elif label == 'B-skill' or label == 'I-skill':

                result.append((token, 'Skill', '#fea'))

            elif label == 'I-organization' or label == 'B-organization':
                result.append((token, 'ORG', '#00d'))

            elif label == 'B-degree' or label == 'I-degree':
                result.append((token, 'Degree', '#808'))

            elif label == 'B-job_position' or label == 'I-job_position':
                result.append((token, 'Job_Pos', '#f8e'))
            else:
                # result.append((token.upper(), label, color))
                result.append((token.lower()))

    result.append((name, 'Name', '#8ef'))
    result.append((email, 'Email', '#faa'))
    result.append((phone, 'Phone', '#afa'))

    return result


# Function: Load the document file (text, docx, pdf supported)
# Limit: 200 MB size
def load_file():
    doc_file = st.file_uploader("Upload Resume Document", type=["pdf", "docx", "txt"])

    if st.button("Process"):

        if doc_file is not None:

            file_details = {"filename": doc_file.name, "filetype": doc_file.type}
            st.write(file_details)

            raw_text = ''

            if doc_file.type == "text/plain":
                # Read as string (decode bytes to string)
                raw_text = str(doc_file.read(), "utf-8")

            elif doc_file.type == "application/pdf":
                try:
                    raw_text = extract_text(doc_file)
                except:
                    st.write("None")

            else:
                raw_text = docx2txt.process(doc_file)

            model = load_model("res/NER_model_full.pt")

            # Run the prediction function if satisfy the condition of text length
            # Else print out the error
            if len(raw_text) > 0:
                st.success("Result out:")
                result = make_predictions(raw_text, model, pre_model_name_ner)
                annotated_text(*result)

            else:
                st.error("Please check the input again or ask Administrator!")


def main():
    st.title("CV Extractor")

    # Navigation
    menu = ["Extract CV", "About"]
    choice = st.sidebar.selectbox("", menu)

    if choice == "About":
        st.subheader("About")
        st.write(
            "A project from ICT30001 – Swinburne University of Technology Vietnam.")
    elif choice == "Extract CV":
        st.subheader("Extract CV")
        load_file()


if __name__ == '__main__':
    main()
