import tiktoken


def num_of_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name=encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    question = "What kind of pets do I like?"
    num = num_of_tokens_from_string(question, "cl100k_base")
    print(num)
