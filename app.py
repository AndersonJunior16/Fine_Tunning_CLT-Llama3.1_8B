import mesop as me
import mesop.labs as mel
from mesop import stateclass
from llama_cpp import Llama
import os

model_directory = r"C:\Users\caixa\Downloads\IA_CLT"
model_filename = "unsloth.Q8_0 (1).gguf"
model_path = os.path.join(model_directory, model_filename)

lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,
    n_batch=512,
    n_gpu_layers=-1,
    n_ctx=2048,
)

# Template do prompt
prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

### Input:
{prompt}

### Response:
"""

def get_response(text):
    prompt = prompt_template.format(prompt=text)
    response = lcpp_llm(
        prompt,
        max_tokens=128,
        temperature=0.5,
        top_p=0.95,
        top_k=50,
        stop=['<|end_of_text|>'],
        echo=False
    )
    response_text = response['choices'][0]['text']
    
    # Verifica se o separador está presente
    if 'Assistant:\n' in response_text:
        return response_text.split('Assistant:\n')[1]
    else:
        return response_text

@stateclass
class State:
    pass

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/",
    title="Chatbot CLT"
)
def page():
    mel.chat(transform, title="Chatbot CLT", bot_user="Mesop Bot")

def transform(input: str, history: list[mel.ChatMessage]):
    content = get_response(input)
    yield content
