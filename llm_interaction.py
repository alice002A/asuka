from llama_cpp import Llama

modelPygmalion7B = 'exact/path/to/pygmalion-7B-ggml-model-q5_1.bin'

input_file = 'exact/path/to/input_file.txt'
memory_file = 'exact/path/to/memory_file.txt'
output_file = 'exact/path/to/output_file.txt'
personality_file = 'exact/path/to/personality_file.txt'
transcribed_text_file = 'exact/path/to/transcribed_text_file.txt'

class llmInteraction:
    def __init__(self):
        self.llm = Llama(model_path=modelPygmalion7B, n_threads=7,n_ctx=512 )

    def generate_response(self):

        max_response_tokens = 124  # Set the desired maximum number of response tokens
        stop_tokens = ["Me:", "\n",]  # stop generating when encountering

        with open(personality_file,'r', encoding='utf-8') as p:
            personality = p.read()
        with open(transcribed_text_file, 'r', encoding='utf-8') as f:
            transcribed = f.read()
        with open(memory_file, 'r', encoding='utf-8') as m:
            lines = m.readlines()
        last_x_lines = lines[-6:]

        history = ''.join(last_x_lines)

        with open(input_file, 'w', encoding='utf-8') as a:
            a.write(personality + history + transcribed + "Asuka:")

        with open(input_file, 'r', encoding='utf-8') as b:
            input_text = b.read()

        output = self.llm(input_text, max_tokens=max_response_tokens, stop=stop_tokens, echo=True)
        output_text = output['choices'][0]['text'].strip() + '\n'

        with open(output_file, 'w', encoding='utf-8') as c:
            c.write(output_text)

        with open(output_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        last_line = lines[-1].strip()
        
        with open(output_file, 'w', encoding='utf-8') as d:
            d.write(last_line)

        with open(memory_file, 'a', encoding='utf-8') as d:
            d.write(transcribed + last_line + '\n')

        print(last_line)