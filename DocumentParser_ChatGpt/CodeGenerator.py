import openai

api_key = ""  # Replace with your OpenAI API key


def generate_code(query):
    openai.api_key = api_key

    prompt = f"Generate Python code to calculate the factorial of a number: '{query}'."

    response = openai.Completion.create(

        engine="text-davinci-002",

        prompt=prompt,

        max_tokens=50,  # Adjust as needed

        n=1,

        stop=None,

        temperature=0.7  # Adjust for creativity vs. accuracy

    )

    code = response.choices[0].text.strip()

    return code


user_query = "Calculate the factorial of 5"

generated_code = generate_code(user_query)

print(f"User query: '{user_query}'")

print(f"Generated code:\n{generated_code}")
