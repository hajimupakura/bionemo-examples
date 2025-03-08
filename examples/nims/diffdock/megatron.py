# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# @st.cache_resource
# def load_model():
#     # Option 1: Load directly from Hugging Face Hub
#     model_name_or_path = "EMBO/BioMegatron345mUncased"

#     # Option 2: Load from your local converted model directory
#     # model_name_or_path = "/home/hmpakula/models/biomegatron345muncasedsquadv1_0"

#     # Try forcing the slow tokenizer (if needed)
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
#     model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     return tokenizer, model, device

# tokenizer, model, device = load_model()

# st.title("Drug Discovery Research Q&A")
# st.write(
#     "This app answers questions related to drug discovery research, "
#     "focusing on the relationships among genes, diseases, and proteins."
# )

# # Refined default context with clear answer labeling
# default_context = """
# You are assisting in drug discovery research, focusing on the relationships among genes, diseases, and proteins.
# Researchers integrate data from genomics, proteomics, and bioinformatics to uncover how gene mutations and protein dysfunction contribute to disease.
# Here are some examples:
# - **BRCA1:** A gene involved in DNA repair; mutations in BRCA1 are linked to breast and ovarian cancers.
# - **TP53:** A tumor suppressor gene; mutations in TP53 are found in many types of cancer.
# - **EGFR:** A gene coding for a receptor involved in cell signaling; abnormalities in EGFR can lead to uncontrolled cell growth.
# This context is provided to help answer questions related to drug discovery and gene research.
# """

# # Suggest a more targeted question in the placeholder
# question = st.text_input("Question", "For example: Which gene is involved in DNA repair?")

# if st.button("Get Answer"):
#     if not question:
#         st.error("Please enter a question.")
#     else:
#         with st.spinner("Generating answer..."):
#             inputs = tokenizer.encode_plus(question, default_context, return_tensors="pt")
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             outputs = model(**inputs)
#             answer_start = torch.argmax(outputs.start_logits)
#             answer_end = torch.argmax(outputs.end_logits) + 1
#             answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
#             answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
#             # Fallback if no concise answer is found
#             if answer.strip() == "":
#                 answer = "No answer could be extracted from the provided context."
        
#         st.success("Answer generated!")
#         st.write("**Answer:**", answer)


# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# from langchain.retrievers import PubMedRetriever  # Requires langchain installation
# from langchain.chains import RetrievalQA

# @st.cache_resource
# def load_components():
#     # Base QA Model
#     qa_tokenizer = AutoTokenizer.from_pretrained("EMBO/BioMegatron345mUncased")
#     qa_model = AutoModelForQuestionAnswering.from_pretrained("EMBO/BioMegatron345mUncased")
    
#     # PubMed Retrieval
#     retriever = PubMedRetriever()  # Automatically handles API connections
    
#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     qa_model.to(device)
    
#     return qa_tokenizer, qa_model, retriever, device

# qa_tokenizer, qa_model, retriever, device = load_components()

# # Enhanced default context with update mechanism
# base_context = """
# Known biomarker-therapy associations:
# - EGFR T790M: Osimertinib resistance in NSCLC
# - BRCA1/2: PARP inhibitor sensitivity
# - PD-L1 >50%: Pembrolizumab response
# - HER2+: Trastuzumab efficacy
# - KRAS G12C: Sotorasib response
# """

# def get_dynamic_context(question):
#     """Retrieve relevant PubMed abstracts"""
#     docs = retriever.get_relevant_documents(
#         query=question,
#         max_results=5  # Get top 5 relevant abstracts
#     )
#     return "\n".join([d.page_content for d in docs])

# st.title("Dynamic Biomarker Discovery System")
# question = st.text_input("Enter your biomarker question:", 
#                        "What genetic markers predict response to CDK4/6 inhibitors?")

# if st.button("Analyze"):
#     with st.spinner("Searching biomedical literature..."):
#         try:
#             # Step 1: Retrieve dynamic context
#             dynamic_context = get_dynamic_context(question)
#             full_context = f"{base_context}\n\nRecent Findings:\n{dynamic_context}"
            
#             # Step 2: QA Processing
#             inputs = qa_tokenizer(
#                 question, 
#                 full_context,
#                 max_length=1024,
#                 truncation=True,
#                 return_tensors="pt"
#             ).to(device)
            
#             outputs = qa_model(**inputs)
#             answer_start = torch.argmax(outputs.start_logits)
#             answer_end = torch.argmax(outputs.end_logits) + 1
            
#             answer = qa_tokenizer.decode(
#                 inputs["input_ids"][0][answer_start:answer_end],
#                 skip_special_tokens=True
#             )
            
#             # Step 3: Validation and fallback
#             if not answer.strip():
#                 raise ValueError("No extractive answer found")
                
#             st.success("**Evidence-Based Answer:**")
#             st.markdown(f"```\n{answer}\n```")
#             st.caption("Source: Integrated analysis of PubMed literature and clinical databases")
            
#         except:
#             st.warning("No conclusive answer found in literature. Generating hypothesis...")
#             # Fallback to generative model (next level)
#             from transformers import BioGptForCausalLM, BioGptTokenizer
#             generative_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
#             generative_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
            
#             prompt = f"Based on biological knowledge: {question}"
#             inputs = generative_tokenizer(prompt, return_tensors="pt").to(device)
#             outputs = generative_model.generate(**inputs, max_length=150)
#             hypothesis = generative_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             st.markdown("**AI-generated Hypothesis:**")
#             st.info(hypothesis)
#             st.caption("Note: This is a speculative hypothesis requiring clinical validation")

# st.markdown("---")
# st.info("""
# **System Capabilities:**
# 1. Searches recent PubMed articles for biomarker evidence
# 2. Combines with established clinical knowledge
# 3. Generates hypotheses when no conclusive data exists
# """)

import torch
from transformers import pipeline, BertTokenizer, AutoModelForMaskedLM

# Define the checkpoint name or path.
checkpoint = "EMBO/BioMegatron345mUncased"

# Load the tokenizer and the model.
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)

# Create a fill-mask pipeline.
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Example question: you can ask any question by placing a [MASK] token where the answer should be.
# For instance, if you want to ask: "What is the capital of France?" you can write:
question = "The capital of France is [MASK]."

# Run the model.
results = fill_mask(question)

# Print the top predictions.
print("Results:")
for result in results:
    print(f"Token: {result['token_str']}, Score: {result['score']:.4f}")

# You can also create a function to ask multiple questions:
def ask_question(question_text):
    # Make sure to include a [MASK] token in the question_text where the answer should be.
    answers = fill_mask(question_text)
    print(f"\nQuestion: {question_text}")
    for ans in answers:
        print(f"Prediction: {ans['token_str']} (score: {ans['score']:.4f})")

# Example usage:
ask_question("The largest 2  planest in our solar system is [MASK].")
ask_question("The author of 'Coming of the Dry Season is [MASK].")
