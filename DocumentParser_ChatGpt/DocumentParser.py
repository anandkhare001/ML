from openai import OpenAI
import os


def parse_document(prompt):
    client = OpenAI(api_key="")

    #  Step 1: Load the file
    uploaded_file = client.files.create(
        file=open("Welcome to our extensive MLOps Bootcamp.pdf", 'rb'),
        purpose='assistants'
    )

    # Step 2: Create an Assistant
    assistant = client.beta.assistants.create(
        name="Course Browser",
        instructions="You are an assistant to provide details about the course.",
        model="gpt-3.5-turbo",
        tools=[{"type": "retrieval"}],
        file_ids=[uploaded_file.id]
    )

    # Step 3: Create thread, message and run for assistant
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role='user',
        content=prompt
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Help the user on course query."
    )

    # Step 4: Run the assistant
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            latest_message = messages.data[0]
            text = latest_message.content[0].text.value
            print(text)
            break

    return text


if __name__ == '__main__':
    input_prompt = "What is the content of 'Crash Course on YAML'."
    text = parse_document(input_prompt)
    print(text)
