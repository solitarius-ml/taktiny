from taktiny import Maestro
from transformers import AutoTokenizer
import jax.numpy as jnp

repo = 'google/gemma-2-2b'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(repo)

print("Loading model...")
model = Maestro.from_pretrained(repo, dtype='bfloat16')

prompt = "Once upon a time\n"
print(f"\nPrompt: {prompt}")

input_ids = tokenizer.encode(prompt, return_tensors='np')
# input_ids = jnp.array(input_ids)

print("Generating...")
output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nOutput:\n{output_text}")
