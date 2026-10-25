from pyfly.dataset import load_dataset
from pyfly.tokenizer import (
    build_prompt_vocabulary,
    build_code_vocabulary,
    code_to_tokens,
    tokens_to_code,
)


examples = load_dataset("data/raw/python_mre.jsonl")

all_prompt_tokens = []

for example in examples:
    all_prompt_tokens.extend(example.prompt_tokens)


prompt_vocab = build_prompt_vocabulary(all_prompt_tokens)
code_vocab = build_code_vocabulary()


print("=== Prompt Vocabulary ===")
print(f"Size: {len(prompt_vocab.token_to_id)}")
print(prompt_vocab.token_to_id)

print()
print("=== Code Vocabulary ===")
print(f"Size: {len(code_vocab.token_to_id)}")
print(code_vocab.token_to_id)


print()
print("=== Tokenizer Test ===")

for example in examples:
    prompt_ids = prompt_vocab.encode(example.prompt_tokens)

    target_tokens = code_to_tokens(example.code)
    target_ids = code_vocab.encode(target_tokens)

    decoded_code = tokens_to_code(target_tokens)

    print()
    print("Prompt:", example.prompt_tokens)
    print("Prompt IDs:", prompt_ids)

    print("Code:", example.code)
    print("Target tokens:", target_tokens)
    print("Target IDs:", target_ids)

    assert decoded_code == example.code
    assert target_tokens[-1] == "EOS"


print()
print("✅ All tokenizer tests PASSED")