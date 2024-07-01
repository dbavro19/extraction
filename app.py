import streamlit as st
from extract_pdf_to_json import pdf_processing
from q_and_a import answer_question


#Setup Streamlit
st.set_page_config(page_title="Extraction", page_icon=":tada", layout="wide")
st.title(f""":rainbow[Extract Financial KPI's from Earnings Report]""")
#Setup Tabs
tab1, tab2 = st.tabs(["Extract and Process", "Q+A"])

#
with tab1:
    st.write("---")
    uploaded_file = st.file_uploader('Upload a .pdf file', type="pdf")
    st.write("---")

    go=st.button("Go!")
    if go:
        st.balloons()
        pdf_processing(uploaded_file)

with tab2:
    st.write("---")
    query = st.text_input("Ask a Question")
    st.write("---")
    ask=st.button("Ask!")
    if ask:
        st.balloons()
        answer_question(query)
        st.success("Done!")