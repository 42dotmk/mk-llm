import os
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Translate check",
    layout="wide",
)

# Load the Parquet file into a DataFrame
@st.cache_data
def load_data(file_path):
    data = pd.read_parquet(file_path)
    return data


def main():
    st.title("Proofread translate files EN-MK")
    st.write("""Instructions:  
             - upload the parquet file  
             - you can navigate through rows via the "Edit Row" section
             - in the next 3 columns we have the text, the first two are the 2 texts in English and Macedonian and the 3rd one starts off with the Macedonian text as placeholder and can be edited to make the necessary changes.  
             - after you have made the changes or decided no changes are needed you click on the Make Changes button and move on to the next sample via the "Edit Row" section.  
             - when you click make changes it automatically creates or appends to a translate_v1.csv which will be in the docker volume.
             """)
    st.write("###")

    st.write("Upload File")
    uploaded_file = st.file_uploader("Choose a Parquet file", type="parquet")
    st.write("###")

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        edit_row_cols = st.columns(3)
        with edit_row_cols[1]:
            st.subheader("Edit Row")
            index = st.number_input(
                "Row index:", min_value=0, max_value=len(df) - 1, value=0, step=1
            )
            st.write("#")
        
        translate_cols = st.columns(3)
        with translate_cols[0]:
            st.write("**English text**:")
            st.write("###")
            st.write(df.iloc[index]["text"])
        with translate_cols[1]:
            st.write("**Macedonian text**:")
            st.write("###")
            st.write(df.iloc[index]["translated_text"])
        with translate_cols[2]:
            new_value = st.text_area(
                "**Make changes here**:", value=df.iloc[index]["translated_text"], height=2000
            )

        m = st.markdown(
            """
        <style>
        div.stButton > button:first-child {
            background-color: rgb(31, 52, 79);
            color:white;
        }
        </style>""",
            unsafe_allow_html=True,
        )
        if st.button("Make Changes"):
            new_row = df.iloc[index]
            new_row["transate_v1"] = new_value
            new_df = pd.DataFrame([new_row])
            new_csv = "./app/data/translate_v1.csv"
            if os.path.isfile(new_csv):
                new_df.to_csv(new_csv, mode="a", index=False, header=False)
                st.write("Changes appended successfully.")
            else:
                new_df.to_csv(new_csv, index=False)
                st.write("Changes appended successfully.")


if __name__ == "__main__":
    main()
