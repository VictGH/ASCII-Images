import ollama

# Initialize a list to keep track of the conversation history
conversation_history = []

def send_message_to_ollama(user_message):
    # Append the user's message to the conversation history
    conversation_history.append({'role': 'user', 'content': user_message})

    # Get the streaming response from Ollama
    response_stream = ollama.chat(
        model='llama3',  # Change the model name if needed
        messages=conversation_history,
        stream=True  # Enable streaming responses
    )

    # Initialize an empty string to collect the full response
    response_content = ""
    print("Ollama:", end=" ")
    for part in response_stream:
        response_content += part['message']['content']
        print(part['message']['content'], end='', flush=True)

    # Add the complete response to the conversation history
    conversation_history.append({'role': 'system', 'content': response_content})
    print()  # Move to new line after the response

def main():
    print("Welcome to Ollama Chat! Type your messages below. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Exiting Ollama Chat. Goodbye!")
            break
        send_message_to_ollama(user_input)

if __name__ == "__main__":
    main()

