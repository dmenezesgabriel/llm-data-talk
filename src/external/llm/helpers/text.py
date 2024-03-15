from langchain.text_splitter import CharacterTextSplitter


class TextHelper:

    @staticmethod
    def get_text_chunks(raw_text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        return text_splitter.split_text(raw_text)
