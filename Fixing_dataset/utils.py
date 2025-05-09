import torch
import tensorflow as tf
import pprint
from datasets import load_dataset
import openai
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from IPython.display import display, HTML
import time
import psutil


class tools_local():

    def printer(self,printing_content):
        """
        Displays content in a formatted HTML output with word wrapping.

        This function takes a string input and displays it using IPython's display capabilities, ensuring that
        the text is wrapped appropriately (i.e., long lines of text won't overflow the display area). It uses 
        HTML's `pre-wrap` style to preserve spaces and newlines while also wrapping text for better readability.

        Args:
        -----
        printing_content (str):
            The content to be displayed. This can be any text or string that should be printed with word wrapping.
        
        Returns:
        --------
        None:
            The function outputs the wrapped content to the notebook using IPython's `display` and `HTML` functions,
            but it does not return any value.
        """
        # Display the output with word wrapping
        wrapped_output = f"<div style='white-space: pre-wrap;'>{printing_content}</div>"
        return display(HTML(wrapped_output))



    def searching(self, response, reference, data):
        """
        Isolates the output and runs a search through the dataset to find its corresponding index in the dataset.

        Args:
        -----
        response (str): 
            The response text to search for in the dataset. The phrase is extracted by removing the first and last 15 characters.

        reference (int): 
            The reference number associated with this search.

        Returns:
        --------
        list: 
            A list of tuples where each tuple contains the reference and the index of the matching row.
        """
        
        search_phrase = response[20:-15]  # Skip the first 15 characters and the last 15 characters
        
        # Search for the phrase in the 'query' column and store the row index
        filtered_rows_with_indices_query = [index for index, example in enumerate(data) if search_phrase.lower() in example['query'].lower()]

        # Only return reference and index, not the full example
        result = [(reference, index) for index in filtered_rows_with_indices_query]
        
        return result

    def resources(self):
        
        # Get the number of GPUs
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        print(f"Number of GPUs available: {num_gpus}")

        # List GPU details
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print(gpu)

        # Get CPU information
        print(f"CPU Count: {psutil.cpu_count(logical=True)}")
        print(f"CPU Usage (%): {psutil.cpu_percent(interval=1)}")

        # Get RAM information
        virtual_memory = psutil.virtual_memory()
        print(f"Total RAM: {virtual_memory.total / (1024 ** 3):.2f} GB")
        print(f"Available RAM: {virtual_memory.available / (1024 ** 3):.2f} GB")
        print(f"RAM Usage (%): {virtual_memory.percent}")

        # Disk Usage (Optional)
        disk_usage = psutil.disk_usage('/')
        print(f"Total Disk Space: {disk_usage.total / (1024 ** 3):.2f} GB")
        print(f"Available Disk Space: {disk_usage.free / (1024 ** 3):.2f} GB")

        # Get current device information (if using CUDA with PyTorch)
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print(f"Current device ID: {current_device}")
            print(f"Current device name: {torch.cuda.get_device_name(current_device)}")
        else:
            print("No CUDA-enabled device found.")




class llm_tools():
    
    def create_message_for_comparison(self,code_snippet, paragraphs): 
        """
        Compares the prompts with the highest cosine similarity to ultimately find the correct prompt for the code 

        Args:
        -----
        code_snippet (str):
            A string containing the Python code that will be compared against the provided paragraphs.
        
        paragraphs (list of str):
            A list of paragraphs (prompts) to be compared with the Python code. Each paragraph is a potential
            candidate that could have generated the provided code snippet.

        Returns:
        --------
        list:
            A list containing two dictionaries, formatted as messages:
            1. A system message that defines the role and behavior of the model.
            2. A user message that contains the code snippet and the paragraphs for comparison.
        """
        # System message to guide GPT's behavior
        system_message = {
            "role": "system",
            "content": "You are an assistant that compares prompts to find the one which most likely generated the provided code"
        }
                                        
        # User message that provides the code and paragraphs
        user_message_content = f"""
        Here is the Python code snippet:
        
        {code_snippet}

        Below are the prompts {len(paragraphs)}. Compare each prompt with the provided code snippet and determine which prompt is the most similar to the code. Provide the index of the most similar prompt and print that prompt under ** ** without paragraph index or anything else.
        """
        
        # Append each paragraph with its index
        for i, paragraph in enumerate(paragraphs, start=1):
            user_message_content += f"\nParagraph {i}: {paragraph}"
        
        user_message = {
            "role": "user",
            "content": user_message_content
        }
        
        return [system_message, user_message]



    def similarity_f(self,text1,text2, model):
        """
        Computes the cosine similarity between the embeddings of two text inputs.

        Args:
        -----
        text1 (str): 
            The first input text that will be encoded and compared.
        
        text2 (str): 
            The second input text that will be encoded and compared.

        Returns:
        --------
        float: 
            A floating-point value representing the cosine similarity between the two text embeddings. The value ranges
            between -1 and 1:
            - 1 indicates that the texts are identical in terms of the embedding space.
            - 0 indicates that the texts are orthogonal (no similarity).
            - -1 indicates maximum dissimilarity in terms of the embedding space.
            
        """
        # If the model is wrapped in DataParallel, use .module to access the actual model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)

        return util.pytorch_cos_sim(embedding1, embedding2).item()



    
    def similarity_ranking(self,data, model, open_gen, top_n): # Make sure the model here is 'all-MiniLM-L6-v2' and not gpt-4o-mini
        """
        Returns the top N most similar entries according to cosine similarity.

        Args:
            data (list of dict): A list of dictionaries containing queries to compare against.
                                Each dictionary should have a 'query' key.
            model: The model used to compute similarity scores.
            open_gen (str): The generated text to compare with the queries.
            top_n (int): The number of top similar entries to return.

        Returns:
            list of dict: A list of the top N similar queries, each represented as a 
                        dictionary with 'score', 'index', and 'query' keys.
        """

        top_similarities = []  # List to store (similarity, index) tuples

        for i in range(len(data)):
            query_response = data[i]['query']
            # open_gen_encoded = model.encode(open_gen, convert_to_tensor=True)
            score = self.similarity_f(open_gen, query_response, model=model)

            # Add the current score and index as a dictionary to the list
            top_similarities.append({'score': score, 'index': i, 'query': data[i]['query']})  # Storing the score and index in a dictionary

            # Sort the list by similarity score in descending order and keep only the top N
            top_similarities = sorted(top_similarities, key=lambda x: x['score'], reverse=True)[:top_n]

        return top_similarities



    def code_comparison(self, snippet1, snippet2):
        """
        Compares two Python code snippets to determine if they serve the same purpose.

        Args:
            snippet1 (str): The first Python code snippet to be compared.
            snippet2 (str): The second Python code snippet to be compared.

        Returns:
            None: The method prepares a system and user message for a language model 
            to analyze and compare the provided code snippets. The results are not 
            returned directly from this function but can be utilized in subsequent 
            processing.

        
        """

        # System message to guide GPT's behavior
        system_message = {
            "role": "system",
            "content": "You are an assistant that compares two code snippets to check if they serve the same purpose or not ? "
        }

        # User message that provides the code and paragraphs
        user_message_content = f"""
        Here are the Python code snippets:

        {snippet1, snippet2}
 
        """

        user_message = {
        "role": "user",
        "content": user_message_content
        }

        return [system_message, user_message]



    def tie_breaker_messages(self,code_snippet, paragraphs):
        """
        Compares the prompts with the highest cosine similarity to ultimately find the correct prompt for the code 

        Args:
        -----
        code_snippet (str):
        A string containing the Python code that will be compared against the provided paragraphs.

        paragraphs (list of str):
        A list of paragraphs (prompts) to be compared with the Python code. Each paragraph is a potential
        candidate that could have generated the provided code snippet.

        Returns:
        --------
        list:
        A list containing two dictionaries, formatted as messages:
        1. A system message that defines the role and behavior of the model.
        2. A user message that contains the code snippet and the paragraphs for comparison.
        """
        # System message to guide GPT's behavior
        system_message = {
            "role": "system",
            "content": "You are an assistant that compares prompts to find the one which most likely generated the provided code"
        }

        # User message that provides the code and paragraphs
        user_message_content = f"""
        Here is the Python code snippet:

        {code_snippet}

        Below are the prompts {len(paragraphs)}. 
        """

        # Append each paragraph with its index
        for i, paragraph in enumerate(paragraphs, start=1):
            user_message_content += f"\nParagraph {i}: {paragraph}"

        user_message = {
        "role": "user",
        "content": user_message_content
        }

        return [system_message, user_message]

    

    def tie_breaker_codes(self, prompt, snippet1, snippet2):
        """
        Compares two Python code snippets to determine which code was generated by the provided prompt.

        Args:
            prompt (str): The prompt used to generate the code snippets.
            snippet1 (str): The first Python code snippet to compare.
            snippet2 (str): The second Python code snippet to compare.

        Returns:
            list: A list containing two dictionaries:
                - The first dictionary is a system message that guides the assistant's behavior.
                - The second dictionary is a user message that includes the code snippets and the prompt.

        Example:
            tie_breaker_codes(
                prompt="Write a function to calculate the factorial of a number.",
                snippet1="def factorial(n): return 1 if n == 0 else n * factorial(n - 1)",
                snippet2="def fact(n): if n == 0: return 1 else: return n * fact(n - 1)"
            )
        """

        system_message = {
            "role": "system",
            "content": "You are an assistant that compares which of these codes were generated by the provided prompt"
        }

        # User message that provides the code and paragraphs
        user_message_content = f"""
        Here is the Python code snippets:

        {snippet1, snippet2}

        And this is the prompt {prompt}. 
        """
        
        user_message = {
        "role": "user",
        "content": user_message_content
        }

        return [system_message, user_message]


    def explainer(self, formatted_code_string, model="gpt-4o-mini", max_tokens=150, temperature=0.01):
        """
        Prepares messages for the Chat API and sends a request.

        Args:
            formatted_code_string (str): The code snippet to be analyzed.
            model (str): The model to use for the API call. Defaults to "gpt-4o-mini".
            max_tokens (int): The maximum number of tokens in the response. Defaults to 150.
            temperature (float): Sampling temperature. Defaults to 0.01.

        Returns:
            dict: The response from the OpenAI API, or None if an error occurs.
        """
        # Prepare the messages for the Chat API
        messages = [
            {"role": "user", "content": f"Explain the purpose of this code without getting into technical details. Here is the code: {formatted_code_string}"}
        ]

        # Make the API call
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            print(f"Error occurred while making the API call: {e}")
            return None


